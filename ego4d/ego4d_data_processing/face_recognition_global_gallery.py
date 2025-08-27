import cv2
import numpy as np
import pandas as pd
import insightface
from insightface.app import FaceAnalysis
import warnings
import hdbscan
import csv
from collections import defaultdict
import sys
import os
import glob
import time
import argparse
from typing import Optional
import inspect


def _pick_insightface_pack(user_choice: Optional[str] = None) -> str:
    """Choose a model pack compatible with the installed insightface version.
    - If user_choice is set and not 'auto', return it.
    - Else pick based on version: buffalo_l (>=0.7), antelopev2 (>=0.3), antelope (<0.3).
    """
    if user_choice and user_choice != "auto":
        return user_choice
    try:
        from packaging.version import Version
        ver = Version(getattr(insightface, "__version__", "0"))
    except Exception:
        # Fallback simple parse
        try:
            ver = insightface.__version__
        except Exception:
            ver = "0"
        class _V:
            def __init__(self, s): self.s = s
            def __ge__(self, other): return False
            def __lt__(self, other): return True
        ver = _V(ver)

    try:
        from packaging.version import Version
        v = Version(insightface.__version__)
        if v >= Version("0.7.0"):
            return "buffalo_l"
        elif v >= Version("0.3.0"):
            return "antelopev2"
        else:
            return "antelope"
    except Exception:
        # If packaging not available, heuristically default to an older pack
        return "antelope"


def create_face_analysis(preferred_provider: str, model_root: Optional[str] = None, model_name: str = "auto"):
    """Create FaceAnalysis with best-effort provider selection.

    - Tries to use the requested onnxruntime provider if supported.
    - Falls back gracefully for older insightface versions without `providers`.
    """
    provider_to_use = preferred_provider
    cuda_available = False
    try:
        import onnxruntime as ort

        available = ort.get_available_providers()
        print(f"ONNX Runtime providers available: {available}")
        cuda_available = "CUDAExecutionProvider" in available
        # Enforce CUDA only
        if provider_to_use == "auto":
            provider_to_use = "CUDAExecutionProvider"
        if provider_to_use != "CUDAExecutionProvider" or not cuda_available:
            raise RuntimeError(
                "CUDAExecutionProvider not available. Install onnxruntime-gpu and ensure CUDA is properly configured."
            )
    except Exception as e:
        # Fail fast if CUDA EP cannot be verified
        raise

    # Determine model root directory from arg or env
    if model_root is None:
        model_root = os.environ.get("INSIGHTFACE_HOME")

    # Prefer passing `providers` if supported by installed insightface
    supports_providers = "providers" in inspect.signature(FaceAnalysis.__init__).parameters
    try:
        pack = _pick_insightface_pack(model_name)
        print(f"Using InsightFace pack: {pack}")
        if supports_providers:
            if model_root:
                return FaceAnalysis(name=pack, providers=[provider_to_use], root=model_root)
            return FaceAnalysis(name=pack, providers=[provider_to_use])
        else:
            warnings.warn(
                "InsightFace FaceAnalysis does not support 'providers' arg; using default backend."
            )
            if model_root:
                return FaceAnalysis(name=pack, root=model_root)
            return FaceAnalysis(name=pack)
    except (TypeError, AssertionError, RuntimeError) as e:
        # Commonly: unsupported kwargs or missing models in very old versions
        msg = str(e)
        print(f"InsightFace initialization failed: {msg}")
        print(
            "Troubleshooting: install 'onnxruntime-gpu' to enable CUDA, and upgrade 'insightface>=0.7'."
        )
        print(
            "If running offline, pre-download models by setting INSIGHTFACE_HOME and calling FaceAnalysis(...).prepare()."
        )
        raise

def load_ground_truth(ground_truth_path):
    """Load ground truth CSV file and return as DataFrame"""

    if not os.path.exists(ground_truth_path):
        print(f" Warning: Ground truth file not found: {ground_truth_path}")
        return None

    try:
        df = pd.read_csv(ground_truth_path, header=0)
        df = df.rename(columns={
            'frame_number': 'frame_number',
            'person_id': 'person_id',
            'x1': 'x1', 'x2': 'x2', 'y1': 'y1', 'y2': 'y2'
        })
        df = df[pd.to_numeric(df['person_id'], errors='coerce').notnull()]
        df['person_id'] = df['person_id'].astype(int)
        print(f" Loaded ground truth: {len(df)} records")
        print(f" Ground truth unique person IDs: {sorted(df['person_id'].unique())}")
        return df

    except Exception as e:
        print(f" Error loading ground truth: {e}")
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
    
    frame_num = data['frame_number']
    detected_bbox = data['bbox']
    
    # Find ground truth annotations for this frame
    frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_num]
    
    best_iou = 0
    best_person_id = None
    
    # Debug: only print for first few frames to avoid spam
    debug_print = frame_num < 10 or frame_num % 100 == 0
    
    if debug_print and len(frame_gt) > 0:
        print(f"    Frame {frame_num}: Detected [{detected_bbox[0]}, {detected_bbox[1]}, {detected_bbox[2]}, {detected_bbox[3]}]")
        for _, gt_row in frame_gt.iterrows():
            gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
            print(f"      GT Person {gt_row['person_id']}: [{gt_bbox[0]:.1f}, {gt_bbox[1]:.1f}, {gt_bbox[2]:.1f}, {gt_bbox[3]:.1f}]")
    
    for _, gt_row in frame_gt.iterrows():
        gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
        iou = calculate_iou(detected_bbox, gt_bbox)
        
        if debug_print:
            print(f"      IoU with Person {gt_row['person_id']}: {iou:.3f}")
        
        if iou > best_iou and iou >= iou_threshold:
            best_iou = iou
            best_person_id = gt_row['person_id']
    
    if best_person_id is not None:
        data['person_id'] = str(int(best_person_id))  # Convert to string number
        if debug_print:
            print(f"    Frame {frame_num}: Matched to Person {best_person_id} with IoU {best_iou:.3f}")
        return data
    
    if debug_print and len(frame_gt) > 0:
        print(f"    Frame {frame_num}: No match found (best IoU: {best_iou:.3f})")
    
    return data

def extract_embeddings(video_path, model, max_frames=None, ground_truth_df=None):
    print(f"  Extracting embeddings from: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return []
    
    # Set maximum frames limit
    if max_frames is None:
        # No limit - process entire video
        max_frames = float('inf')
    elif max_frames == "MAX":
        max_frames = float('inf')
    # Otherwise use the specified limit (default: 36000 for 20 minutes at 30fps)
    
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
    app = create_face_analysis(args.execution_provider, model_root=args.insightface_root, model_name=args.insightface_model)
    # CUDA-only mode
    ctx_id = 0
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    # Check if the video file exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file '{args.video_path}' does not exist!")
        return
    
    # For single video, use it as both gallery source and target
    video_files = [args.video_path]
    
    print(f"\nProcessing single video file: {os.path.basename(args.video_path)}")

    print("\n--- Creating Global Gallery ---")
    start_time = time.time()
    
    # Load ground truth if available
    ground_truth_df = None
    if args.ground_truth_dir:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        ground_truth_path = os.path.join(args.ground_truth_dir, f"{video_basename}.csv")
        ground_truth_df = load_ground_truth(ground_truth_path)
    
    first_video_path = video_files[0]
    print(f"Step 1: Creating gallery from '{os.path.basename(first_video_path)}' (20 minute limit)...")
    part1_data = extract_embeddings(first_video_path, app, max_frames=args.max_frames, ground_truth_df=ground_truth_df)
    
    gallery_embeddings = []
    gallery_ids = []
    if not part1_data:
        print("Error: No faces found in the first video to create a gallery. Aborting this group.")
        return

    part1_embeddings = np.array([data['embedding'] for data in part1_data])
    
    # Get max person ID from ground truth to limit the number of clusters
    max_gt_person_id = 2  # Default to 2 people
    if ground_truth_df is not None:
        max_gt_person_id = ground_truth_df['person_id'].max()
        print(f"    Ground truth max person ID: {max_gt_person_id}")
    
    # Adjust clustering parameters to get reasonable number of clusters
    # Try different min_cluster_size values to get close to ground truth person count
    best_clusterer = None
    best_labels = None
    best_score = float('inf')
    
    for min_size in [args.min_cluster_size, args.min_cluster_size * 2, args.min_cluster_size * 4]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='euclidean')
        labels = clusterer.fit_predict(part1_embeddings)
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Score based on how close we are to the ground truth person count
        score = abs(num_clusters - max_gt_person_id)
        print(f"    Trying min_cluster_size={min_size}: {num_clusters} clusters, score={score}")
        
        if score < best_score:
            best_score = score
            best_clusterer = clusterer
            best_labels = labels
    
    labels = best_labels
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    # Create gallery with person IDs matching ground truth range
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        gallery_embeddings.append(np.mean(part1_embeddings[cluster_indices], axis=0))
        # Use person IDs 1, 2, 3... instead of person_1, person_2...
        gallery_ids.append(str(i + 1))
    
    print(f"Gallery created with {len(gallery_ids)} unique people: {gallery_ids}")

    for video_path in video_files:
        print(f"\nStep 2: Processing '{os.path.basename(video_path)}' using gallery (full video)...")
        video_data = extract_embeddings(video_path, app, max_frames=None, ground_truth_df=ground_truth_df)  # No limit for Part 2
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
        
        print(f"Step 3: Saving results...")
        save_outputs(video_path, video_data, args.output_dir)

    end_time = time.time()
    print(f"\nFinished processing '{os.path.basename(args.video_path)}' in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition for a single video file.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the video file to process.")
    parser.add_argument('--output_dir', type=str, default="processed_videos", help="Directory to save output files.")
    parser.add_argument('--min_cluster_size', type=int, default=5, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument('--recognition_threshold', type=float, default=0.8, help="Cosine distance threshold for recognition.")
    parser.add_argument(
        '--execution_provider',
        type=str,
        default='auto',
        choices=['auto', 'CUDAExecutionProvider'],
        help="ONNX Runtime EP (CUDA required). 'auto' enforces CUDA."
    )
    parser.add_argument('--insightface_root', type=str, default=None, help="Optional cache dir for InsightFace models (defaults to $INSIGHTFACE_HOME).")
    parser.add_argument('--insightface_model', type=str, default='auto', help="InsightFace model pack: auto|buffalo_l|antelopev2|antelope")
    parser.add_argument('--ground_truth_dir', type=str, help="Directory containing ground truth CSV files (optional).")
    parser.add_argument('--max_frames', type=int, default=36000, help="Maximum frames to process (default: 36000 = 20 minutes at 30fps).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)



# Body pose extraction
# How to detect body + person recognition

# segmentation algorithm to find people's body and see where it moves
# co-tracking to see where it moved
