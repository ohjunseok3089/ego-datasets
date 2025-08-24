import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import hdbscan
import csv
from collections import defaultdict
import sys
import os
import glob
import time
import argparse

def load_ground_truth(ground_truth_path):
    """Load ground truth CSV file and return as DataFrame"""
    if not os.path.exists(ground_truth_path):
        print(f"    Warning: Ground truth file not found: {ground_truth_path}")
        return None
    
    try:
        df = pd.read_csv(ground_truth_path)
        print(f"    Loaded ground truth: {len(df)} records")
        return df
    except Exception as e:
        print(f"    Error loading ground truth: {e}")
        return None

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
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
    
    return intersection / union if union > 0 else 0.0

def match_with_ground_truth(data, ground_truth_df, iou_threshold=0.3):
    """Match detected faces with ground truth annotations"""
    if ground_truth_df is None:
        return data
    
    print(f"    Matching with ground truth (IoU threshold: {iou_threshold})...")
    
    matched_count = 0
    frame_num = data['frame_number']
    detected_bbox = data['bbox']
    
    # Find ground truth annotations for this frame
    frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_num]
    
    best_iou = 0
    best_person_id = None
    
    for _, gt_row in frame_gt.iterrows():
        gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
        iou = calculate_iou(detected_bbox, gt_bbox)
        
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_person_id = gt_row['person_id']
    
    if best_person_id is not None:
        data['person_id'] = str(int(best_person_id))  # Convert to string number
        print(f"    Matched {matched_count}/{len(data)} faces with ground truth")
        return data
    
    return data

def extract_embeddings(video_path, model, max_frames=None, ground_truth_df=None):
    print(f"  Extracting embeddings from: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return []
    
    if max_frames == "MAX":
        max_frames = 1e9 # infinity
    # Set maximum frames limit (20 minutes * 30 fps)
    if max_frames is None:
        max_frames = 20 * 30 * 60  # 36000 frames for 20 minutes at 30fps
    
    face_data = []
    frame_number = 0
    while cap.isOpened() and frame_number < max_frames:
        ret, frame = cap.read()
        if not ret: break
        faces = model.get(frame)
        for face in faces:
            iter_face_data = {
                'video_path': video_path,
                'frame_number': frame_number,
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding
            }
            iter_face_data = match_with_ground_truth(iter_face_data, ground_truth_df, iou_threshold=0.3)
            face_data.append(iter_face_data)
        frame_number += 1
    cap.release()
    return face_data

def save_outputs(video_path, face_data_for_video, output_dir):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_name}_global_gallery.csv")
    output_video_path = os.path.join(output_dir, f"{base_name}_global_gallery.mp4")

    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2'])
        for data in face_data_for_video:
            if data.get('person_id', 'unknown') != 'unknown':
                x1, y1, x2, y2 = data['bbox']
                writer.writerow([data['frame_number'], data['person_id'], x1, y1, x2, y2])
    print(f"    - Saved annotations to: {output_csv_path}")

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_to_faces = defaultdict(list)
    for data in face_data_for_video:
        frame_to_faces[data['frame_number']].append(data)

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        if frame_number in frame_to_faces:
            for data in frame_to_faces[frame_number]:
                if data.get('person_id', 'unknown') != 'unknown':
                    bbox = data['bbox']
                    person_id = data['person_id']
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, person_id, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    print(f"    - Saved labeled video to: {output_video_path}")

def main(args):
    print("Initializing InsightFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=[args.execution_provider])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    try:
        search_pattern = os.path.join(args.search_path, '*.MP4')
        all_mp4_files = glob.glob(search_pattern)
        video_files = sorted([f for f in all_mp4_files if args.pattern_to_match in os.path.basename(f)])
        if not video_files:
            print(f"Error: No video files found in '{args.search_path}' containing the pattern '{args.pattern_to_match}'")
            return
        print(f"\nFound {len(video_files)} matching video files for pattern '{args.pattern_to_match}':")
        for f in video_files: print(f"  - {os.path.basename(f)}")
    except Exception as e:
        print(f"Error finding video files: {e}")
        return

    print("\n--- Creating Global Gallery ---")
    start_time = time.time()
    
    ground_truth_df = None
    if args.ground_truth_dir:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        ground_truth_path = os.path.join(args.ground_truth_dir, f"{video_basename}.csv")
        ground_truth_df = load_ground_truth(ground_truth_path)
    first_video_path = video_files[0]
    print(f"Step 1: Creating gallery from '{os.path.basename(first_video_path)}'...")
    part1_data = extract_embeddings(first_video_path, app, max_frames=args.max_frames, ground_truth_df=ground_truth_df)
    
    gallery_embeddings = []
    gallery_ids = []
    if not part1_data:
        print("Error: No faces found in the first video to create a gallery. Aborting this group.")
        return

    part1_embeddings = np.array([data['embedding'] for data in part1_data])
    clusterer = hdbscan.HDBSCAN(min_cluster_size=args.min_cluster_size, metric='euclidean')
    labels = clusterer.fit_predict(part1_embeddings)
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        gallery_embeddings.append(np.mean(part1_embeddings[cluster_indices], axis=0))
        gallery_ids.append(f"person_{i+1}")
    
    print(f"Gallery created with {len(gallery_ids)} unique people.")

    for video_path in video_files:
        print(f"\nStep 2: Processing '{os.path.basename(video_path)}' using gallery...")
        video_data = extract_embeddings(video_path, app, max_frames="MAX")
        if not video_data: continue

        for data in video_data:
            if not gallery_embeddings:
                data['person_id'] = 'unknown'
                continue
            distances = 1 - np.dot(gallery_embeddings, data['embedding'])
            best_match_index = np.argmin(distances)
            
            if distances[best_match_index] < args.recognition_threshold:
                data['person_id'] = gallery_ids[best_match_index]
            else:
                data['person_id'] = 'unknown'
        
        save_outputs(video_path, video_data, args.output_dir)

    end_time = time.time()
    print(f"\nFinished processing group '{args.pattern_to_match}' in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition using a global gallery created from the first video of a group.")
    parser.add_argument('--search_path', type=str, required=True, help="Directory path to search for videos.")
    parser.add_argument('--pattern_to_match', type=str, required=True, help="Unique pattern to identify a group of videos (e.g., 'day_1__con_1__person_1').")
    parser.add_argument('--output_dir', type=str, default="processed_videos", help="Directory to save output files.")
    parser.add_argument('--min_cluster_size', type=int, default=75, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument('--recognition_threshold', type=float, default=0.8, help="Cosine distance threshold for recognition.")
    parser.add_argument('--execution_provider', type=str, default='CUDAExecutionProvider', help="Execution provider for ONNX Runtime (e.g., 'CoreMLExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider').")
    parser.add_argument('--ground_truth_dir', type=str, help="Directory containing ground truth CSV files (optional).")
    parser.add_argument('--max_frames', type=int, default=36000, help="Maximum frames to process (default: 36000 = 20 minutes at 30fps).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)



# Body pose extraction
# How to detect body + person recognition

# segmentation algorithm to find people's body and see where it moves
# co-tracking to see where it moved
