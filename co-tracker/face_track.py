import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment # Import the assignment solver
import os
import sys
import math

class FaceTracker:
    def __init__(self, video_path):
        # --- Configuration ---
        self.DEBUG = True # <-- Set to True to see all debug logs
        self.video_path = video_path
        self.similarity_threshold = 0.3  # Lowered from 0.5 to be more lenient
        self.max_pixel_distance = 200    # Increased from 150 to allow more movement
        self.feature_smoothing_alpha = 0.2
        self.max_frames_to_keep_track = 60  # Increased from 30 to keep tracks longer
        self.reidentification_threshold = 0.4  # New: threshold for re-identifying lost tracks
        
        # --- State and MediaPipe Setup ---
        self.tracked_persons = {}
        self.lost_tracks = {}  # New: store recently lost tracks for re-identification
        self.next_person_id = 1
        self.frame_count = 0
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.5, model_selection=1)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_writer = self._initialize_video_writer()
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def _initialize_video_writer(self):
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        base_name = os.path.splitext(self.video_path)[0]
        output_filename = f"{base_name}_face_detect_debug.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print(f"Saving output to: {output_filename}")
        return cv2.VideoWriter(output_filename, fourcc, fps, (self.width, self.height))

    def _get_color_histogram(self, frame, bbox):
        x, y, w, h = bbox
        face_roi = frame[y:y+h, x:x+w]
        if face_roi.size == 0: return None
        hist = cv2.calcHist([face_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()

    def _process_frame(self, frame):
        self.frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)

        current_detections = []
        if self.DEBUG: 
            print(f"\n--- Frame {self.frame_count} ---")
            if self.tracked_persons:
                print(f"  Currently tracking: {list(self.tracked_persons.keys())}")
            if self.lost_tracks:
                print(f"  Lost tracks pool: {list(self.lost_tracks.keys())}")

        if results.detections:
            if self.DEBUG: print(f"Found {len(results.detections)} faces:")
            for i, detection in enumerate(results.detections):
                bbox_relative = detection.location_data.relative_bounding_box
                bbox_cv = (int(bbox_relative.xmin * self.width), int(bbox_relative.ymin * self.height),
                           int(bbox_relative.width * self.width), int(bbox_relative.height * self.height))
                features = self._get_color_histogram(frame, bbox_cv)
                if features is not None:
                    current_detections.append({'id': i, 'bbox': bbox_cv, 'features': features})
                    if self.DEBUG:
                        min_f, max_f, mean_f = features.min(), features.max(), features.mean()
                        print(f"  -> New Detection id:{i} | bbox:{bbox_cv} | features (min/max/mean): {min_f:.3f}/{max_f:.3f}/{mean_f:.3f}")
        
        # --- Optimal Assignment Logic ---
        if not self.tracked_persons and not current_detections:
            pass # No tracks and no detections
        elif not self.tracked_persons and current_detections:
            pass # Handle first detections later
        elif self.tracked_persons and not current_detections:
            # No detections but we have tracks - just continue tracking
            if self.DEBUG: print(f"  No detections, continuing with {len(self.tracked_persons)} tracked persons")
            pass
        else:
            tracked_ids = list(self.tracked_persons.keys())
            num_tracked = len(tracked_ids)
            num_detected = len(current_detections)

            cost_matrix = np.full((num_tracked, num_detected), 100.0) # High cost default
            for i in range(num_tracked):
                person_data = self.tracked_persons[tracked_ids[i]]
                last_x, last_y, last_w, last_h = person_data['bbox']
                last_center = (last_x + last_w // 2, last_y + last_h // 2)
                for j in range(num_detected):
                    new_detection = current_detections[j]
                    new_x, new_y, new_w, new_h = new_detection['bbox']
                    new_center = (new_x + new_w // 2, new_y + new_h // 2)
                    
                    distance = math.dist(last_center, new_center)
                    if distance <= self.max_pixel_distance:
                        similarity = cosine_similarity([person_data['features']], [new_detection['features']])[0][0]
                        cost_matrix[i, j] = 1.0 - similarity

            if self.DEBUG:
                print("Cost Matrix (rows=tracked, cols=new):")
                print(np.round(cost_matrix, 2))

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if self.DEBUG: print(f"Optimal Pairs (tracked_idx, new_detection_idx): {list(zip(row_ind, col_ind))}")

            matched_indices = set()
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] < (1.0 - self.similarity_threshold):
                    person_id = tracked_ids[r]
                    matched_detection = current_detections[c]
                    person_data = self.tracked_persons[person_id]
                    if self.DEBUG: print(f"  MATCH: Tracked {person_id} -> New Detection {matched_detection['id']} (Cost: {cost_matrix[r,c]:.4f})")
                    
                    person_data['features'] = (self.feature_smoothing_alpha * matched_detection['features'] + (1 - self.feature_smoothing_alpha) * person_data['features'])
                    person_data['bbox'] = matched_detection['bbox']
                    person_data['last_seen'] = self.frame_count
                    x, y, w, h = matched_detection['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    matched_indices.add(c)

            unmatched_detections = set(range(num_detected)) - matched_indices
            for i in unmatched_detections:
                new_detection = current_detections[i]
                
                # Try to re-identify with lost tracks first
                best_reid_match = None
                best_reid_similarity = 0
                
                for lost_id, lost_data in self.lost_tracks.items():
                    # Check if the lost track is recent enough for re-identification
                    if (self.frame_count - lost_data['last_seen']) <= 90:  # Within 90 frames
                        similarity = cosine_similarity([lost_data['features']], [new_detection['features']])[0][0]
                        if similarity > best_reid_similarity and similarity > self.reidentification_threshold:
                            best_reid_similarity = similarity
                            best_reid_match = lost_id
                
                if best_reid_match:
                    # Re-identify the lost track
                    if self.DEBUG: print(f"  RE-IDENTIFIED: Detection {new_detection['id']} -> {best_reid_match} (similarity: {best_reid_similarity:.3f})")
                    self.tracked_persons[best_reid_match] = {
                        'features': new_detection['features'], 
                        'bbox': new_detection['bbox'], 
                        'last_seen': self.frame_count
                    }
                    del self.lost_tracks[best_reid_match]
                    x, y, w, h = new_detection['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for re-identified
                    cv2.putText(frame, best_reid_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    # Create new person ID
                    new_id = f"Person {self.next_person_id}"
                    self.next_person_id += 1
                    self.tracked_persons[new_id] = {'features': new_detection['features'], 'bbox': new_detection['bbox'], 'last_seen': self.frame_count}
                    if self.DEBUG: print(f"  NEW PERSON: Unmatched detection {new_detection['id']} becomes {new_id}")
                    x, y, w, h = new_detection['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        if not self.tracked_persons and current_detections:
            for new_detection in current_detections:
                # Try to re-identify with lost tracks first
                best_reid_match = None
                best_reid_similarity = 0
                
                for lost_id, lost_data in self.lost_tracks.items():
                    # Check if the lost track is recent enough for re-identification
                    if (self.frame_count - lost_data['last_seen']) <= 90:  # Within 90 frames
                        similarity = cosine_similarity([lost_data['features']], [new_detection['features']])[0][0]
                        if similarity > best_reid_similarity and similarity > self.reidentification_threshold:
                            best_reid_similarity = similarity
                            best_reid_match = lost_id
                
                if best_reid_match:
                    # Re-identify the lost track
                    if self.DEBUG: print(f"  RE-IDENTIFIED: Detection {new_detection['id']} -> {best_reid_match} (similarity: {best_reid_similarity:.3f})")
                    self.tracked_persons[best_reid_match] = {
                        'features': new_detection['features'], 
                        'bbox': new_detection['bbox'], 
                        'last_seen': self.frame_count
                    }
                    del self.lost_tracks[best_reid_match]
                    x, y, w, h = new_detection['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # Yellow for re-identified
                    # cv2.putText(frame, best_reid_match, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                else:
                    # Create new person ID
                    new_id = f"Person {self.next_person_id}"
                    self.next_person_id += 1
                    self.tracked_persons[new_id] = {'features': new_detection['features'], 'bbox': new_detection['bbox'], 'last_seen': self.frame_count}
                    if self.DEBUG: print(f"  FIRST PERSON: Detection {new_detection['id']} becomes {new_id}")
                    x, y, w, h = new_detection['bbox']
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Move lost tracks to re-identification pool instead of deleting immediately
        lost_ids = [pid for pid, pdata in self.tracked_persons.items() if (self.frame_count - pdata['last_seen']) > self.max_frames_to_keep_track]
        for pid in lost_ids:
            if self.DEBUG: print(f"  MOVING to re-id pool: {pid}")
            self.lost_tracks[pid] = self.tracked_persons[pid]
            del self.tracked_persons[pid]
        
        # Clean up old lost tracks (after 120 frames)
        old_lost_ids = [pid for pid, pdata in self.lost_tracks.items() if (self.frame_count - pdata['last_seen']) > 120]
        for pid in old_lost_ids:
            if self.DEBUG: print(f"  PERMANENTLY REMOVING old lost track: {pid}")
            del self.lost_tracks[pid]

        return frame

    def run(self):
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success: break
            processed_frame = self._process_frame(frame)
            self.video_writer.write(processed_frame)
            progress = (self.frame_count / self.total_frames) * 100
            sys.stdout.write(f"\r  Progress: {progress:.2f}%")
            sys.stdout.flush()
        self._cleanup()

    def _cleanup(self):
        self.detector.close()
        self.cap.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        print(f"\nProcessing complete.")

if __name__ == '__main__':
    VIDEO_FILE = 'vid_001__day_1__con_1__person_1_part1.MP4'
    if not os.path.exists(VIDEO_FILE):
        print(f"Error: Video file not found at '{VIDEO_FILE}'")
    else:
        tracker = FaceTracker(video_path=VIDEO_FILE)
        tracker.run()
# import cv2
# import mediapipe as mp
# import numpy as np
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
# from sklearn.metrics.pairwise import cosine_similarity
# import os

# # --- Configuration ---
# VIDEO_PATH = 'vid_001__day_1__con_1__person_1_part1.MP4'
# MODEL_PATH = 'pose_landmarker_heavy.task'
# SIMILARITY_THRESHOLD = 0.5
# # Set to 3 to handle a potential mirror image
# MAX_TRACKED_PERSONS = 3

# # --- Global Variables ---
# tracked_persons = {}
# next_person_id = 1

# # --- MediaPipe Setup ---
# BaseOptions = mp.tasks.BaseOptions
# PoseLandmarker = mp.tasks.vision.PoseLandmarker
# PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
# VisionRunningMode = mp.tasks.vision.RunningMode

# options = PoseLandmarkerOptions(
#     base_options=BaseOptions(model_asset_path=MODEL_PATH),
#     running_mode=VisionRunningMode.VIDEO,
#     num_poses=MAX_TRACKED_PERSONS,
#     min_pose_detection_confidence=0.5,
#     min_tracking_confidence=0.5)

# detector = PoseLandmarker.create_from_options(options)

# # --- Helper Functions ---

# def get_bounding_box(landmarks, frame_width, frame_height):
#     """Calculates a bounding box from pose landmarks."""
#     x_coords = [lm.x * frame_width for lm in landmarks]
#     y_coords = [lm.y * frame_height for lm in landmarks]
#     if not x_coords or not y_coords:
#         return None
#     x_min, x_max = int(min(x_coords)), int(max(x_coords))
#     y_min, y_max = int(min(y_coords)), int(max(y_coords))
#     x_min, y_min = max(0, x_min - 20), max(0, y_min - 20)
#     x_max, y_max = min(frame_width, x_max + 20), min(frame_height, y_max + 20)
#     return (x_min, y_min, x_max - x_min, y_max - y_min)

# def get_color_histogram(frame, bbox):
#     """Calculates a 3D color histogram for the person in the bounding box."""
#     x, y, w, h = bbox
#     person_roi = frame[y:y+h, x:x+w]
#     if person_roi.size == 0:
#         return None
#     hist = cv2.calcHist([person_roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()

# # --- Main Video Processing Loop ---
# cap = cv2.VideoCapture(VIDEO_PATH)
# frame_idx = 0

# # --- Video Saving Setup ---
# total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# base_name = os.path.splitext(VIDEO_PATH)[0]
# output_filename = f"{base_name}_face_detect.mp4"
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# video_writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

# print(f"Processing video: {VIDEO_PATH}")
# print(f"Saving output to: {output_filename}")

# while cap.isOpened():
#     success, frame = cap.read()
#     if not success:
#         break

#     # --- Print Progress ---
#     progress = (frame_idx / total_frames) * 100
#     print(f"  Progress: {progress:.2f}%", end='\r')

#     mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#     detection_result = detector.detect_for_video(mp_image, frame_idx)

#     current_detections = []
#     if detection_result.pose_landmarks:
#         for landmarks in detection_result.pose_landmarks:
#             bbox = get_bounding_box(landmarks, frame_width, frame_height)
#             if bbox:
#                 features = get_color_histogram(frame, bbox)
#                 if features is not None:
#                     current_detections.append({'bbox': bbox, 'features': features})

#     # --- Re-identification Logic ---
#     unmatched_detections = list(range(len(current_detections)))
#     for person_id, person_data in tracked_persons.items():
#         best_match_idx, max_similarity = -1, -1
#         for i in unmatched_detections:
#             similarity = cosine_similarity([person_data['features']], [current_detections[i]['features']])[0][0]
#             if similarity > max_similarity:
#                 max_similarity, best_match_idx = similarity, i
        
#         if max_similarity > SIMILARITY_THRESHOLD:
#             person_data['features'] = current_detections[best_match_idx]['features']
#             person_data['last_frame_seen'] = frame_idx
#             x, y, w, h = current_detections[best_match_idx]['bbox']
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#             cv2.putText(frame, person_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             unmatched_detections.remove(best_match_idx)
            
#     for i in unmatched_detections:
#         new_id = f"Person {next_person_id}"
#         next_person_id += 1
#         if next_person_id > MAX_TRACKED_PERSONS:
#             break
#         tracked_persons[new_id] = {'features': current_detections[i]['features'], 'last_frame_seen': frame_idx}
#         x, y, w, h = current_detections[i]['bbox']
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(frame, new_id, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     # --- Write Frame to Output Video ---
#     video_writer.write(frame)
#     frame_idx += 1

# # --- Cleanup ---
# cap.release()
# video_writer.release()
# cv2.destroyAllWindows()
# print(f"\nProcessing complete. Video saved successfully.")