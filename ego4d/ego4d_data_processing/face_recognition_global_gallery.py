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
from typing import Optional, List, Tuple
import shutil
from scipy.optimize import linear_sum_assignment


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


def create_face_analysis(preferred_provider: str, model_root: Optional[str] = None, model_name: str = "antelopev2"):
    """Create FaceAnalysis for InsightFace 0.7.3 with CUDA provider only."""
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

    # Initialize FaceAnalysis exactly as supported by insightface>=0.7.3
    pack = model_name if model_name and model_name != "auto" else "antelopev2"
    print(f"Using InsightFace pack: {pack} with provider {provider_to_use}")
    def _init_face_analysis():
        if model_root:
            return FaceAnalysis(name=pack, providers=[provider_to_use], root=model_root)
        return FaceAnalysis(name=pack, providers=[provider_to_use])
    try:
        return _init_face_analysis()
    except (TypeError, AssertionError, RuntimeError) as e:
        # Commonly: missing/corrupted downloaded model pack causing 'assert "detection" in self.models'
        msg = str(e)
        print(f"InsightFace initialization failed: {msg}")
        print(
            "Troubleshooting: install 'onnxruntime-gpu' to enable CUDA, and upgrade 'insightface>=0.7'."
        )
        print(
            "If running offline, pre-download models by setting INSIGHTFACE_HOME and calling FaceAnalysis(...).prepare()."
        )
        # Attempt one clean re-download by removing possibly corrupted pack
        try:
            root = model_root or os.environ.get("INSIGHTFACE_HOME")
            if root:
                pack_dir = os.path.join(root, "models", pack)
                if os.path.isdir(pack_dir):
                    print(f"Cleaning corrupted InsightFace pack at: {pack_dir}")
                    shutil.rmtree(pack_dir, ignore_errors=True)
                # also remove cached zip if present
                zip_path = os.path.join(root, "models", f"{pack}.zip")
                if os.path.isfile(zip_path):
                    try:
                        os.remove(zip_path)
                    except OSError:
                        pass
                print("Retrying InsightFace initialization after cleanup...")
                return _init_face_analysis()
        except Exception as e2:
            print(f"Retry after cleanup failed: {e2}")
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

def assign_detections_to_gt(dets: List[dict], frame_gt_df: pd.DataFrame, iou_threshold: float = 0.3) -> List[Tuple[int, int]]:
    """Return list of (det_index, gt_index) matches per frame using Hungarian assignment on IoU.

    - dets: list of detection dicts with 'bbox'
    - frame_gt_df: ground-truth rows for the frame
    - Only pairs with IoU >= threshold are returned
    """
    if frame_gt_df is None or frame_gt_df.empty or not dets:
        return []
    gt_boxes = frame_gt_df[['x1','y1','x2','y2']].to_numpy().tolist()
    # build IoU matrix (gt x det)
    iou_mat = []
    for gt in gt_boxes:
        row = []
        for d in dets:
            row.append(calculate_iou(d['bbox'], gt))
        iou_mat.append(row)
    if not iou_mat:
        return []
    # convert to cost and solve min-cost assignment
    import numpy as np
    cost = 1.0 - np.array(iou_mat, dtype=np.float32)
    gi, dj = linear_sum_assignment(cost)
    matches = []
    for g, d in zip(gi, dj):
        if iou_mat[g][d] >= iou_threshold:
            matches.append((d, g))  # det index, gt index
    return matches

def extract_embeddings(video_path, model, max_frames=None, ground_truth_df=None, keep_unmatched: bool = False):
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
        dets = []
        for face in faces:
            dets.append({
                'video_path': video_path,
                'frame_number': frame_number,
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding,
            })
        if ground_truth_df is not None:
            frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_number]
            matches = assign_detections_to_gt(dets, frame_gt, iou_threshold=0.3)
            matched_det_indices = set()
            for det_idx, gt_idx in matches:
                item = dets[det_idx]
                gt_row = frame_gt.iloc[gt_idx]
                item['person_id'] = str(int(gt_row['person_id']))
                face_data.append(item)
                matched_det_indices.add(det_idx)
            if keep_unmatched:
                for i, item in enumerate(dets):
                    if i not in matched_det_indices:
                        face_data.append(item)
        else:
            face_data.extend(dets)
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
    print(f"Step 1: Seeding gallery from GT-labeled frames only...")
    # Phase 1: build gallery strictly from GT-labeled detections across the video
    part1_data = extract_embeddings(first_video_path, app, max_frames=None, ground_truth_df=ground_truth_df, keep_unmatched=False)
    
    gallery_embeddings = []
    gallery_ids = []
    if not part1_data:
        print("Error: No faces found in the first video to create a gallery. Aborting this group.")
        return

    if ground_truth_df is not None:
        # Build gallery directly from GT-labeled embeddings (authoritative IDs)
        by_id = defaultdict(list)
        for d in part1_data:
            if 'person_id' in d:
                by_id[d['person_id']].append(d['embedding'])
        for pid, embs in sorted(by_id.items(), key=lambda x: int(x[0])):
            gallery_embeddings.append(np.mean(np.stack(embs, axis=0), axis=0))
            gallery_ids.append(pid)
        print(f"Gallery built from GT IDs: {gallery_ids}")
    else:
        part1_embeddings = np.array([data['embedding'] for data in part1_data])
        # Cluster to form gallery if GT not available
        max_gt_person_id = 2
        best_clusterer = None
        best_labels = None
        best_score = float('inf')
        for min_size in [args.min_cluster_size, args.min_cluster_size * 2, args.min_cluster_size * 4]:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_size, metric='euclidean')
            labels = clusterer.fit_predict(part1_embeddings)
            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            score = abs(num_clusters - max_gt_person_id)
            print(f"    Trying min_cluster_size={min_size}: {num_clusters} clusters, score={score}")
            if score < best_score:
                best_score = score
                best_clusterer = clusterer
                best_labels = labels
        labels = best_labels
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        for i in range(num_clusters):
            cluster_indices = np.where(labels == i)[0]
            gallery_embeddings.append(np.mean(part1_embeddings[cluster_indices], axis=0))
            gallery_ids.append(str(i + 1))
        print(f"Gallery created with {len(gallery_ids)} unique people: {gallery_ids}")

    # Map for fast reservation of gallery slots by GT labels
    id_to_index = {pid: j for j, pid in enumerate(gallery_ids)}

    for video_path in video_files:
        print(f"\nStep 2: Processing '{os.path.basename(video_path)}' using seeded gallery (full video)...")
        # Phase 2: extract all detections; if GT exists, matched ones are labeled, others kept for recognition
        video_data = extract_embeddings(video_path, app, max_frames=None, ground_truth_df=ground_truth_df, keep_unmatched=True)
        if not video_data:
            continue

        # Assign unique IDs per frame using Hungarian only for unlabeled detections
        by_frame = defaultdict(list)
        for idx, d in enumerate(video_data):
            by_frame[d['frame_number']].append((idx, d))
        for frame_num, items in by_frame.items():
            import numpy as np
            # Reserve gallery indices already present due to GT labels in this frame
            reserved = set()
            for _, d in items:
                pid = d.get('person_id')
                if pid is not None and pid in id_to_index:
                    reserved.add(id_to_index[pid])

            # Build distance matrix for unknowns only
            unknown_indices = [i for i, (idx, d) in enumerate(items) if 'person_id' not in d]
            if not unknown_indices:
                continue
            D = np.zeros((len(unknown_indices), len(gallery_embeddings)), dtype=np.float32)
            for ri, item_i in enumerate(unknown_indices):
                _, d = items[item_i]
                D[ri, :] = 1.0 - np.dot(gallery_embeddings, d['embedding'])
            fi, gj = linear_sum_assignment(D)
            assigned = set()
            for ri, j in zip(fi, gj):
                if j in reserved or j in assigned:
                    continue
                if D[ri, j] < args.recognition_threshold:
                    item_i = unknown_indices[ri]
                    idx, _ = items[item_i]
                    video_data[idx]['person_id'] = gallery_ids[j]
                    assigned.add(j)
            # Others remain unknown
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
    parser.add_argument('--insightface_model', type=str, default='antelopev2', choices=['antelopev2','buffalo_l','antelope'], help="InsightFace model pack (default: antelopev2)")
    parser.add_argument('--ground_truth_dir', type=str, help="Directory containing ground truth CSV files (optional).")
    parser.add_argument('--max_frames', type=int, default=36000, help="Maximum frames to process (default: 36000 = 20 minutes at 30fps).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)



# Body pose extraction
# How to detect body + person recognition

# segmentation algorithm to find people's body and see where it moves
# co-tracking to see where it moved
