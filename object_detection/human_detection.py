import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import argparse
import os
from collections import defaultdict

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.08

def map_to_ground_truth(detection_results, face_recognition_csv_path):
    """Map human detection results to ground truth face recognition data"""
    try:
        face_df = pd.read_csv(face_recognition_csv_path)
        print(f"Loaded face recognition data: {len(face_df)} records")
    except FileNotFoundError:
        print(f"Face recognition file not found: {face_recognition_csv_path}")
        return detection_results
    
    # Create mapping from body detections to face detections
    person_mapping = defaultdict(list)  # body_track_id -> list of (face_person_id, iou_score, frame_count)
    
    detection_df = pd.DataFrame(detection_results)
    
    for _, body_row in detection_df.iterrows():
        if 'person_id_' not in body_row['class_name']:
            continue
            
        body_frame = body_row['frame']
        body_box = [body_row['x1'], body_row['y1'], body_row['x2'], body_row['y2']]
        body_track_id = body_row['class_name']
        
        # Look for face detections in nearby frames (±5 frames tolerance)
        frame_tolerance = 5
        nearby_faces = face_df[
            (face_df['frame_number'] >= body_frame - frame_tolerance) & 
            (face_df['frame_number'] <= body_frame + frame_tolerance)
        ]
        
        for _, face_row in nearby_faces.iterrows():
            face_box = [face_row['x1'], face_row['y1'], face_row['x2'], face_row['y2']]
            iou = calculate_iou(body_box, face_box)
            
            if iou > 0.1:  # Minimum overlap threshold
                face_person_id = face_row['person_id']
                person_mapping[body_track_id].append((face_person_id, iou, 1))
    
    # Determine best mapping for each body track based on majority overlap
    final_mapping = {}
    for body_track_id, matches in person_mapping.items():
        if not matches:
            continue
            
        # Group by face_person_id and sum IoU scores
        person_scores = defaultdict(lambda: {'total_iou': 0, 'count': 0})
        for face_person_id, iou, count in matches:
            person_scores[face_person_id]['total_iou'] += iou
            person_scores[face_person_id]['count'] += count
        
        # Find person with highest average IoU and sufficient overlap
        best_person = None
        best_score = 0
        for person_id, data in person_scores.items():
            avg_iou = data['total_iou'] / data['count']
            if data['count'] >= 3 and avg_iou > best_score:  # Require at least 3 overlapping frames
                best_person = person_id
                best_score = avg_iou
        
        if best_person:
            final_mapping[body_track_id] = best_person
            print(f"Mapped {body_track_id} -> person_{best_person} (avg IoU: {best_score:.3f})")
    
    # Update detection results with mapped person IDs
    updated_results = []
    for result in detection_results:
        updated_result = result.copy()
        if result['class_name'] in final_mapping:
            mapped_person_id = final_mapping[result['class_name']]
            updated_result['class_name'] = f"{mapped_person_id}"
        
        updated_results.append(updated_result)
    
    print(f"Mapped {len(final_mapping)} body tracks to face recognition IDs")
    return updated_results

def is_face_inside_body(face_box, body_box):
    """Check if face is properly positioned within body box (face should be in upper portion)"""
    face_x1, face_y1, face_x2, face_y2 = face_box
    body_x1, body_y1, body_x2, body_y2 = body_box
    
    # Face center
    face_center_x = (face_x1 + face_x2) / 2
    face_center_y = (face_y1 + face_y2) / 2
    
    # Body dimensions
    body_width = body_x2 - body_x1
    body_height = body_y2 - body_y1
    
    # Check if face center is within body horizontally
    if not (body_x1 <= face_center_x <= body_x2):
        return False
    
    # Check if face is in upper portion of body (top 60% of body height - more lenient)
    upper_body_limit = body_y1 + (body_height * 0.6)
    if not (body_y1 <= face_center_y <= upper_body_limit):
        return False
    
    # Additional check: face should have reasonable overlap with body (more lenient)
    iou = calculate_iou(face_box, body_box)
    if iou < 0.005:  # More lenient overlap requirement
        return False
    
    return True

def apply_global_mapping(detection_results, face_df):
    """Apply global mapping based on strict face-body positioning"""
    track_mapping = defaultdict(lambda: defaultdict(lambda: {'matches': 0, 'frames': set()}))
    
    detection_df = pd.DataFrame(detection_results)
    
    print("Analyzing face-body overlaps...")
    
    # Collect all valid face-body matches across all frames
    for _, body_row in detection_df.iterrows():
        if 'person_id_' not in body_row['class_name']:
            continue
            
        body_frame = body_row['frame']
        body_box = [body_row['x1'], body_row['y1'], body_row['x2'], body_row['y2']]
        body_track_id = body_row['class_name']
        
        # Look for face detections in nearby frames (7 frames for better coverage)
        frame_tolerance = 7
        nearby_faces = face_df[
            (face_df['frame_number'] >= body_frame - frame_tolerance) & 
            (face_df['frame_number'] <= body_frame + frame_tolerance)
        ]
        
        for _, face_row in nearby_faces.iterrows():
            face_box = [face_row['x1'], face_row['y1'], face_row['x2'], face_row['y2']]
            
            # Use strict positioning check instead of just IoU
            if is_face_inside_body(face_box, body_box):
                face_person_id = face_row['person_id']
                track_mapping[body_track_id][face_person_id]['matches'] += 1
                track_mapping[body_track_id][face_person_id]['frames'].add(body_frame)
    
    # Determine final mapping with strict criteria
    final_mapping = {}
    for body_track_id, person_scores in track_mapping.items():
        if not person_scores:
            continue
            
        # Find person with most matches
        best_person = None
        best_match_count = 0
        best_frame_count = 0
        
        for person_id, data in person_scores.items():
            match_count = data['matches']
            frame_count = len(data['frames'])
            
            # More lenient requirements: at least 3 matches across at least 2 different frames
            if match_count >= 3 and frame_count >= 2:
                if match_count > best_match_count:
                    best_person = person_id
                    best_match_count = match_count
                    best_frame_count = frame_count
        
        # More lenient conflict resolution - allow multiple tracks per person
        if best_person:
            final_mapping[body_track_id] = best_person
            print(f"Global mapping: {body_track_id} -> person_{best_person} ({best_match_count} matches across {best_frame_count} frames)")
    
    # Update ALL detection results with global mapping
    updated_results = []
    for result in detection_results:
        updated_result = result.copy()
        if result['class_name'] in final_mapping:
            mapped_person_id = final_mapping[result['class_name']]
            updated_result['class_name'] = f"{mapped_person_id}"
        
        updated_results.append(updated_result)
    
    print(f"Applied global mapping to {len(final_mapping)} tracks with strict criteria")
    return updated_results

def recreate_video_with_mapped_labels(input_video_path, output_video_path, detection_results, model, device):
    """Recreate the video with the globally mapped labels"""
    print("Recreating video with mapped labels...")
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not reopen video file {input_video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Create frame-indexed detection results
    frame_detections = defaultdict(list)
    for result in detection_results:
        frame_detections[result['frame']].append(result)
    
    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw only mapped detections (exclude person_id_X labels)
        if frame_number in frame_detections:
            for detection in frame_detections[frame_number]:
                person_label = detection['class_name']
                
                # Only draw if it's a mapped person (not person_id_X or person_no_id)
                if not person_label.startswith('person_id_') and not person_label.startswith('person_no_id'):
                    x1, y1, x2, y2 = detection['x1'], detection['y1'], detection['x2'], detection['y2']
                    confidence = detection['confidence']
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{person_label} {confidence}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        out.write(frame)
        frame_number += 1
        
        if frame_number % 1000 == 0:
            print(f"Recreated {frame_number} frames")
    
    cap.release()
    out.release()
    print("Video recreation completed with mapped labels")

def process_video_with_yolo(video_path, output_video_path, output_csv_path, face_csv_path=None):
    try:
        model = YOLO("yolo11x.pt") 
    except Exception as e:
        print(f"Error loading YOLO model 'yolo11x.pt': {e}")
        print("Please ensure the model file exists in the current directory.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model.to(device)
    
    # Load face recognition data for real-time mapping
    face_df = None
    track_to_person_mapping = {}
    if face_csv_path and os.path.exists(face_csv_path):
        try:
            face_df = pd.read_csv(face_csv_path)
            print(f"Loaded face recognition data: {len(face_df)} records")
            print(f"Face CSV columns: {list(face_df.columns)}")
        except Exception as e:
            print(f"Error loading face recognition data: {e}")
            face_df = None
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    detection_results = []
    frame_number = 0
    
    # Store all detections with original track IDs first, map globally at the end
    def get_current_label(track_id):
        """Get current label - will be updated globally at the end"""
        return f"person_id_{track_id}"
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model.track(frame, classes=0, conf=0.5, verbose=False, persist=True, tracker="bytetrack.yaml")
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_with_conf = [(box, float(box.conf.item())) for box in result.boxes]
                boxes_with_conf.sort(key=lambda x: x[1], reverse=True)
                
                for box, confidence in boxes_with_conf[:3]:
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    body_box = [x1, y1, x2, y2]
                    
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id.item())
                        person_label = get_current_label(track_id)
                    else:
                        person_label = f"{class_name}_no_id"
                    
                    detection_results.append({
                        'frame': frame_number,
                        'class_name': person_label,
                        'confidence': f"{confidence:.2f}",
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{person_label} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")
            
    cap.release()
    out.release()
    
    if detection_results:
        # Apply global mapping to all detection results
        if face_df is not None:
            detection_results = apply_global_mapping(detection_results, face_df)
            
        # Re-create the video with updated labels
        recreate_video_with_mapped_labels(video_path, output_video_path, detection_results, model, device)
        
        # Filter to only include mapped persons (exclude person_id_X and person_no_id)
        filtered_results = [
            result for result in detection_results 
            if not result['class_name'].startswith('person_id_') and not result['class_name'].startswith('person_no_id')
        ]
        
        df = pd.DataFrame(filtered_results)
        df.to_csv(output_csv_path, index=False)
        print(f"Detection results saved to {output_csv_path} ({len(filtered_results)} mapped detections out of {len(detection_results)} total)")
    else:
        print("No person detections found in the video")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO and OC-SORT Object Tracking")
    parser.add_argument('--input', '-i', type=str, required=True, help="Input video file path")
    parser.add_argument('--output', '-o', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    video_filename_base = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_filename_base}_detected.mp4")
    output_csv_path = os.path.join(output_dir, f"{video_filename_base}_detections.csv")
    
    # Construct face recognition CSV path
    input_dir = os.path.dirname(input_path)
    face_csv_path = os.path.join(input_dir, "processed_face_recognition_videos", f"{video_filename_base}_global_gallery.csv")
    
    process_video_with_yolo(input_path, output_video_path, output_csv_path, face_csv_path)
