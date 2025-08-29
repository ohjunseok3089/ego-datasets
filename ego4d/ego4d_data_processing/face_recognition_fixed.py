import cv2
import numpy as np
import pandas as pd
import insightface
from insightface.app import FaceAnalysis
import warnings
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except Exception:
    HDBSCAN_AVAILABLE = False
    hdbscan = None
import csv
from collections import defaultdict
import sys
import os
import glob
import time
import argparse
from typing import Optional, List, Tuple, Dict
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
    """Create FaceAnalysis with a safe provider selection (CUDA if available, else CPU).

    - Tries CUDAExecutionProvider first when available; otherwise falls back to CPUExecutionProvider.
    - Honors explicit provider requests when possible.
    """
    provider_to_use = None
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
        print(f"ONNX Runtime providers available: {available}")
        has_cuda = "CUDAExecutionProvider" in available
        has_cpu = "CPUExecutionProvider" in available
        if preferred_provider == "CUDAExecutionProvider":
            if not has_cuda:
                raise RuntimeError("Requested CUDAExecutionProvider but it is not available.")
            provider_to_use = "CUDAExecutionProvider"
        else:
            # auto: prefer CUDA, else CPU
            provider_to_use = "CUDAExecutionProvider" if has_cuda else ("CPUExecutionProvider" if has_cpu else None)
        if provider_to_use is None:
            raise RuntimeError("No suitable ONNX Runtime provider found (need CUDA or CPU provider).")
    except Exception as e:
        print(f"ONNX Runtime provider selection error: {e}")
        # As a last resort, try default FaceAnalysis init without provider hint
        provider_to_use = None

    # Determine model root directory from arg or env
    if model_root is None:
        model_root = os.environ.get("INSIGHTFACE_HOME")

    # Initialize FaceAnalysis (insightface>=0.7.3 signature)
    pack = model_name if model_name and model_name != "auto" else "antelopev2"
    print(f"Using InsightFace pack: {pack} with provider {provider_to_use or 'default'}")

    def _init_face_analysis():
        kwargs = {}
        if provider_to_use is not None:
            kwargs["providers"] = [provider_to_use]
        if model_root:
            kwargs["root"] = model_root
        return FaceAnalysis(name=pack, **kwargs)

    try:
        return _init_face_analysis()
    except (TypeError, AssertionError, RuntimeError) as e:
        # Commonly: missing/corrupted downloaded model pack causing 'assert "detection" in self.models'
        msg = str(e)
        print(f"InsightFace initialization failed: {msg}")
        print("Troubleshooting: ensure 'insightface>=0.7' is installed.")
        print("If running offline, pre-download models by setting INSIGHTFACE_HOME and calling FaceAnalysis(...).prepare().")
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


def extract_gt_embeddings(video_path, model, ground_truth_df: pd.DataFrame) -> Dict[int, List[np.ndarray]]:
    """
    Phase 1: Extract embeddings only from ground truth frames and collect by person_id.
    Returns: Dict[person_id, List[embeddings]]
    """
    print(f"  Phase 1: Extracting GT embeddings from: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return {}
    
    gt_embeddings_by_person = defaultdict(list)
    gt_frames = set(ground_truth_df['frame_number'].unique())
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
            
        # Only process frames that have ground truth annotations
        if frame_number not in gt_frames:
            frame_number += 1
            continue
            
        faces = model.get(frame)
        dets = []
        for face in faces:
            dets.append({
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding,
            })
        
        frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_number]
        matches = assign_detections_to_gt(dets, frame_gt, iou_threshold=0.3)
        
        for det_idx, gt_idx in matches:
            item = dets[det_idx]
            gt_row = frame_gt.iloc[gt_idx]
            person_id = int(gt_row['person_id'])
            gt_embeddings_by_person[person_id].append(item['embedding'])
            
        frame_number += 1
    
    cap.release()
    
    print(f"    Collected GT embeddings for {len(gt_embeddings_by_person)} people:")
    for pid, embeddings in gt_embeddings_by_person.items():
        print(f"      Person {pid}: {len(embeddings)} embeddings")
    
    return gt_embeddings_by_person


def create_person_prototypes(gt_embeddings_by_person: Dict[int, List[np.ndarray]]) -> Tuple[List[np.ndarray], List[int]]:
    """
    Create prototype embeddings for each person from ground truth embeddings.
    Returns: (prototypes, person_ids)
    """
    prototypes = []
    person_ids = []
    
    for person_id in sorted(gt_embeddings_by_person.keys()):
        embeddings = gt_embeddings_by_person[person_id]
        if embeddings:
            # Average embeddings and L2 normalize
            proto = np.mean(np.stack(embeddings, axis=0), axis=0)
            norm = np.linalg.norm(proto) + 1e-8
            proto = proto / norm
            prototypes.append(proto)
            person_ids.append(person_id)
    
    print(f"  Created prototypes for persons: {person_ids}")
    return prototypes, person_ids


def assign_detections_to_prototypes_greedy(detections: List[dict], prototypes: List[np.ndarray], 
                                          person_ids: List[int], threshold: float = 0.4) -> List[Tuple[int, int, float]]:
    """
    Greedy assignment based on highest confidence first.
    Returns list of (detection_idx, prototype_idx, similarity_score)
    """
    if not detections or not prototypes:
        return []
    
    # Compute all similarities
    detection_embeddings = np.stack([d['embedding'] for d in detections])
    prototype_matrix = np.stack(prototypes)
    similarities = np.dot(prototype_matrix, detection_embeddings.T)  # (n_prototypes, n_detections)
    
    assignments = []
    used_prototypes = set()
    used_detections = set()
    
    # Greedy assignment: highest similarity first
    while True:
        # Mask out already used prototypes/detections
        masked_similarities = similarities.copy()
        for p_idx in used_prototypes:
            masked_similarities[p_idx, :] = -1
        for d_idx in used_detections:
            masked_similarities[:, d_idx] = -1
            
        # Find best remaining match
        max_sim = np.max(masked_similarities)
        if max_sim < threshold:
            break
            
        # Get indices of maximum similarity
        max_positions = np.where(masked_similarities == max_sim)
        p_idx, d_idx = max_positions[0][0], max_positions[1][0]  # Take first if multiple
        
        assignments.append((d_idx, p_idx, max_sim))
        used_prototypes.add(p_idx)
        used_detections.add(d_idx)
    
    return assignments


def process_all_frames(video_path, model, ground_truth_df: pd.DataFrame, 
                      prototypes: List[np.ndarray], person_ids: List[int], 
                      recognition_threshold: float = 0.6, max_frames=None):
    """
    Phase 2: Process all frames, using GT labels where available and matching against prototypes elsewhere.
    """
    print(f"  Phase 2: Processing all frames from: {os.path.basename(video_path)}...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open {video_path}")
        return []
    
    # Set maximum frames limit
    if max_frames is None:
        max_frames = float('inf')
    elif max_frames == "MAX":
        max_frames = float('inf')
    
    all_face_data = []
    frame_number = 0
    gt_frames = set(ground_truth_df['frame_number'].unique()) if ground_truth_df is not None else set()
    
    while cap.isOpened() and frame_number < max_frames:
        ret, frame = cap.read()
        if not ret: 
            break
        
        faces = model.get(frame)
        dets = []
        for face in faces:
            dets.append({
                'video_path': video_path,
                'frame_number': frame_number,
                'bbox': face.bbox.astype(int),
                'embedding': face.normed_embedding,
            })
        
        # Check if this frame has ground truth
        if frame_number in gt_frames:
            # Use ground truth labels
            frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_number]
            matches = assign_detections_to_gt(dets, frame_gt, iou_threshold=0.3)
            
            for det_idx, gt_idx in matches:
                item = dets[det_idx].copy()
                gt_row = frame_gt.iloc[gt_idx]
                item['person_id'] = str(int(gt_row['person_id']))
                item['source'] = 'ground_truth'
                all_face_data.append(item)
            
            # Keep unmatched detections for potential recognition
            matched_indices = set(det_idx for det_idx, _ in matches)
            unmatched_dets = [dets[i] for i in range(len(dets)) if i not in matched_indices]
        else:
            # No ground truth for this frame, all detections are unmatched
            unmatched_dets = dets
        
        # Try to recognize unmatched detections against prototypes using greedy assignment
        if unmatched_dets and prototypes:
            # Use greedy assignment for better multi-person handling
            assignments = assign_detections_to_prototypes_greedy(
                unmatched_dets, prototypes, person_ids, threshold=(1 - recognition_threshold)
            )
            
            for det_idx, proto_idx, similarity in assignments:
                det = unmatched_dets[det_idx].copy()
                det['person_id'] = str(person_ids[proto_idx])
                det['source'] = 'recognized'
                det['confidence'] = float(similarity)
                all_face_data.append(det)
                
            # Log assignment info for debugging
            if assignments:
                assigned_persons = [person_ids[proto_idx] for _, proto_idx, _ in assignments]
                confidences = [sim for _, _, sim in assignments]
                print(f"    Frame {frame_number}: assigned {len(assignments)} faces to persons {assigned_persons} "
                      f"(confidences: {[f'{c:.3f}' for c in confidences]})")
            # else: unassigned detections are left unmatched (don't add to results)
        
        frame_number += 1
    
    cap.release()
    print(f"    Processed {frame_number} frames, found {len(all_face_data)} labeled faces")
    
    # Print statistics
    by_source = defaultdict(int)
    by_person = defaultdict(int)
    for item in all_face_data:
        source = item.get('source', 'unknown')
        person_id = item.get('person_id', 'unknown')
        by_source[source] += 1
        by_person[person_id] += 1
    
    print(f"    Source breakdown: {dict(by_source)}")
    print(f"    Person breakdown: {dict(by_person)}")
    
    return all_face_data


def save_outputs(video_path, face_data_for_video, output_dir):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_csv_path = os.path.join(output_dir, f"{base_name}_fixed_face_recognition.csv")
    output_video_path = os.path.join(output_dir, f"{base_name}_fixed_face_recognition.mp4")

    # Save CSV with additional metadata
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'source', 'confidence'])
        for data in face_data_for_video:
            if data.get('person_id', 'unknown') != 'unknown':
                x1, y1, x2, y2 = data['bbox']
                source = data.get('source', 'unknown')
                confidence = data.get('confidence', 1.0)
                writer.writerow([data['frame_number'], data['person_id'], x1, y1, x2, y2, source, confidence])
    print(f"    - Saved annotations to: {output_csv_path}")

    # Create annotated video
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
                    source = data.get('source', '')
                    confidence = data.get('confidence', 1.0)
                    
                    # Color code by source: green for GT, blue for recognized
                    color = (0, 255, 0) if source == 'ground_truth' else (255, 0, 0)
                    
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    label = f"P{person_id}"
                    if source == 'recognized':
                        label += f" ({confidence:.2f})"
                    cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
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
    
    print(f"\nProcessing video file: {os.path.basename(args.video_path)}")

    # Load ground truth - this is REQUIRED for the new approach
    ground_truth_df = None
    if args.ground_truth_dir:
        video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
        ground_truth_path = os.path.join(args.ground_truth_dir, f"{video_basename}.csv")
        ground_truth_df = load_ground_truth(ground_truth_path)
        
        if ground_truth_df is None or ground_truth_df.empty:
            print("ERROR: Ground truth is required but not found or empty!")
            print("Cannot proceed without ground truth data.")
            return
    else:
        print("ERROR: Ground truth directory is required!")
        print("Please provide --ground_truth_dir argument.")
        return
    
    print("\n=== PHASE 1: Extract Ground Truth Embeddings ===")
    start_time = time.time()
    
    # Phase 1: Extract embeddings only from ground truth frames
    gt_embeddings_by_person = extract_gt_embeddings(args.video_path, app, ground_truth_df)
    
    if not gt_embeddings_by_person:
        print("ERROR: No ground truth embeddings found!")
        print("Please check that:")
        print("1. Ground truth file contains valid face annotations")
        print("2. Face detection is working properly")
        print("3. IoU matching threshold is appropriate")
        return
    
    # Create person prototypes from ground truth embeddings
    prototypes, person_ids = create_person_prototypes(gt_embeddings_by_person)
    
    print(f"\n=== PHASE 2: Process All Frames with GT-based Recognition ===")
    # Phase 2: Process all frames using GT labels where available and recognition elsewhere
    all_face_data = process_all_frames(
        args.video_path, app, ground_truth_df, prototypes, person_ids, 
        recognition_threshold=args.recognition_threshold, max_frames=args.max_frames
    )
    
    print(f"\n=== PHASE 3: Save Results ===")
    save_outputs(args.video_path, all_face_data, args.output_dir)
    
    end_time = time.time()
    print(f"\nCompleted processing '{os.path.basename(args.video_path)}' in {end_time - start_time:.2f} seconds.")
    print(f"Total faces labeled: {len(all_face_data)}")
    
    # Final validation
    unique_person_ids = set()
    for item in all_face_data:
        if 'person_id' in item:
            unique_person_ids.add(int(item['person_id']))
    
    expected_person_ids = set(person_ids)
    print(f"Expected person IDs from GT: {sorted(expected_person_ids)}")
    print(f"Actual person IDs in results: {sorted(unique_person_ids)}")
    
    if unique_person_ids != expected_person_ids:
        print("WARNING: Mismatch between expected and actual person IDs!")
    else:
        print("✓ Person ID validation passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fixed face recognition for a single video file.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the video file to process.")
    parser.add_argument('--output_dir', type=str, default="processed_videos", help="Directory to save output files.")
    parser.add_argument('--recognition_threshold', type=float, default=0.6, help="Cosine distance threshold for recognition (lower = more lenient).")
    parser.add_argument(
        '--execution_provider',
        type=str,
        default='auto',
        choices=['auto', 'CUDAExecutionProvider'],
        help="ONNX Runtime EP (CUDA required). 'auto' enforces CUDA."
    )
    parser.add_argument('--insightface_root', type=str, default=None, help="Optional cache dir for InsightFace models (defaults to $INSIGHTFACE_HOME).")
    parser.add_argument('--insightface_model', type=str, default='antelopev2', choices=['antelopev2','buffalo_l','antelope'], help="InsightFace model pack (default: antelopev2)")
    parser.add_argument('--ground_truth_dir', type=str, required=True, help="Directory containing ground truth CSV files (REQUIRED).")
    parser.add_argument('--max_frames', type=int, default=None, help="Maximum frames to process (default: None = process entire video).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    main(args)
