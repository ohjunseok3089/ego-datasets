"""
Processes video clips to track a red circle, calculates head movement,
and aggregates the analysis results, correctly handling overlapping frames
and adding 'next_movement' data to each frame.
"""

import argparse
import cv2
import json
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Assuming track_red.py with these functions exists in the same directory
from track_red import detect_red_circle, calculate_head_movement, remap_position_from_movement


def compute_content_roi(frame: np.ndarray, white_threshold: int = 240) -> Optional[Tuple[int, int, int, int]]:
    """
    Region of interest is found by finding the bounding box of the non-white pixels in the frame.
    returns the top left corner coordinates and the width and height of the region of interest.
    """
    if frame is None or frame.size == 0:
        return None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Non-white mask
    non_white_mask = (gray < white_threshold).astype(np.uint8) * 255

    # If almost everything is white, bail out
    non_white_ratio = float(np.count_nonzero(non_white_mask)) / non_white_mask.size
    if non_white_ratio < 0.01:
        return None

    # Find the bounding box of non-white pixels
    contours, _ = cv2.findContours(non_white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Use the largest contour area as the content region
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Sanity check: avoid tiny or degenerate boxes
    H, W = gray.shape[:2]
    if w < W * 0.2 or h < H * 0.2:
        return None

    return int(x), int(y), int(w), int(h)


def parse_video_filename(filename: str, mode: str = "advanced") -> Optional[Dict[str, Any]]:
    """
    Parses a video filename to extract metadata based on the specified mode.
    
    Args:
        filename: The video filename to parse
        mode: "advanced" for complex pattern, "default" for simple recording_X_Y.mp4 pattern
    """
    if mode == "advanced":
        # Original complex pattern
        pattern = re.compile(
            r"^(?P<base_name>.+?\((?P<start_frame>\d+)_(?P<end_frame>\d+)_(?P<category>[a-zA-Z_]+)\))_"
            r"(?P<processed_start>\d+)_(?P<processed_end>\d+)"
            r"\.MP4$",
            re.IGNORECASE
        )
        match = pattern.match(filename)
        if not match:
            return None
            
        data = match.groupdict()
        data['group_id'] = data.pop('base_name')
        data['social_category'] = data.pop('category')
        
        for key in ['start_frame', 'end_frame', 'processed_start', 'processed_end']:
            data[key] = int(data[key])
            
        return data
    
    elif mode == "default":
        # Simple pattern: recording_379_390.mp4
        pattern = re.compile(
            r"^(?P<base_name>.+?)_(?P<start_frame>\d+)_(?P<end_frame>\d+)"
            r"\.mp4$",
            re.IGNORECASE
        )
        match = pattern.match(filename)
        if not match:
            return None
            
        data = match.groupdict()
        data['group_id'] = data['base_name']
        data['social_category'] = 'default'  # Default category
        data['processed_start'] = int(data['start_frame'])
        data['processed_end'] = int(data['end_frame'])
        
        for key in ['start_frame', 'end_frame']:
            data[key] = int(data[key])
            
        return data
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'advanced' or 'default'.")

def analyze_video_clip(
    video_path: Path, 
    global_frame_offset: int, 
    social_category: str,
    is_first_clip_in_group: bool,
    last_n_frames: int = 0,
    output_video_path: Optional[Path] = None, 
    fps_override: Optional[float] = None, 
    show_video: bool = False,
    video_fov_degrees: float = 104.0,
    skip_frozen: bool = False,
    frozen_strategy: str = "circle",  # "circle" or "diff"
    frozen_warmup_frames: int = 15,
    position_epsilon_px: float = 0.5,
    diff_mean_threshold: float = 1.0,
    debug: bool = False,
) -> Tuple[List[Dict[str, Any]], List[float], Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Processes a single video clip, aware of overlapping frames between clips.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {video_path}")
        return [], []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_override if fps_override is not None else detected_fps
    
    # Calculate total frame count and start processing frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_processing_frame = 0
    if last_n_frames > 0 and total_frames > last_n_frames:
        start_processing_frame = total_frames - last_n_frames
        print(f"Processing last {last_n_frames} frames (from frame {start_processing_frame} to {total_frames-1})")
    else:
        print(f"Processing all {total_frames} frames")
    
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_data = []
    prediction_errors = []
    prev_red_pos = None
    frame_idx_in_video = 0
    output_frame_count = 0
    content_roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h) of content without white borders
    # State for frozen-frame skipping
    skipping_frozen = skip_frozen
    static_reference_pos = None  # in cropped coords
    prev_gray_for_diff = None
    static_counter = 0
    skipped_leading_frames = 0
    # State for cross-clip overlap handling
    awaiting_transition_for_prev = False
    transition_movement_for_prev: Optional[Dict[str, Any]] = None

    # For clip-level metadata
    clip_level_roi: Optional[Tuple[int, int, int, int]] = None
    clip_level_frame_size = {"width": width, "height": height}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames until we reach the start processing frame
        if frame_idx_in_video < start_processing_frame:
            frame_idx_in_video += 1
            continue

        # Establish content ROI at the first processed frame
        if content_roi is None:
            content_roi = compute_content_roi(frame)
            clip_level_roi = content_roi

        # Select the frame region used for all computations (exclude white borders if detected)
        if content_roi is not None:
            roi_x, roi_y, roi_w, roi_h = content_roi
            proc_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
        else:
            roi_x = roi_y = 0
            roi_w, roi_h = frame.shape[1], frame.shape[0]
            proc_frame = frame

        if not is_first_clip_in_group and frame_idx_in_video == start_processing_frame:
            red_circle = detect_red_circle(proc_frame)
            if red_circle:
                prev_red_pos = (float(red_circle[0]), float(red_circle[1]))  # coords in cropped space
                # We will use the next frame to compute movement and pass it back to update
                # the previous batch's last frame's next_movement
                awaiting_transition_for_prev = True
            frame_idx_in_video += 1
            continue

        vis_frame = frame.copy()
        red_circle = detect_red_circle(proc_frame)
        
        # Positions for computation are in cropped space; for reporting keep full-frame coords
        curr_red_pos_cropped = (float(red_circle[0]), float(red_circle[1])) if red_circle else None
        curr_red_pos = (curr_red_pos_cropped[0] + roi_x, curr_red_pos_cropped[1] + roi_y) if red_circle else None
        curr_radius = int(red_circle[2]) if red_circle else None

        # Skip initial frozen frames
        processed_idx = (frame_idx_in_video - start_processing_frame)
        if skipping_frozen and processed_idx < frozen_warmup_frames:
            if frozen_strategy == "circle":
                if curr_red_pos_cropped is not None:
                    if static_reference_pos is None:
                        static_reference_pos = curr_red_pos_cropped
                        static_counter = 1
                    else:
                        dx = curr_red_pos_cropped[0] - static_reference_pos[0]
                        dy = curr_red_pos_cropped[1] - static_reference_pos[1]
                        if (dx * dx + dy * dy) ** 0.5 <= position_epsilon_px:
                            static_counter += 1
                        else:
                            skipping_frozen = False
                            prev_red_pos = curr_red_pos_cropped
                else:
                    # If undetected, stop skipping to avoid dropping valid frames
                    skipping_frozen = False
            else:  # diff strategy
                gray = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY)
                if prev_gray_for_diff is None:
                    prev_gray_for_diff = gray
                    static_counter = 1
                else:
                    mean_abs = float(np.mean(cv2.absdiff(prev_gray_for_diff, gray)))
                    if mean_abs <= diff_mean_threshold:
                        static_counter += 1
                    else:
                        skipping_frozen = False
                        prev_red_pos = curr_red_pos_cropped if curr_red_pos_cropped is not None else prev_red_pos
                    prev_gray_for_diff = gray

            if skipping_frozen:
                skipped_leading_frames += 1
                frame_idx_in_video += 1
                continue
        
        head_movement, recalculated_pos, prediction_error = None, None, None

        if prev_red_pos and curr_red_pos_cropped:
            head_movement = calculate_head_movement(
                prev_red_pos,
                curr_red_pos_cropped,
                roi_w,
                roi_h,
                video_fov_degrees=video_fov_degrees,
            )
            
            if awaiting_transition_for_prev and transition_movement_for_prev is None:
                transition_movement_for_prev = head_movement
                awaiting_transition_for_prev = False

            if head_movement and frame_data:
                frame_data[-1]['next_movement'] = head_movement

            if head_movement and 'horizontal' in head_movement and 'radians' in head_movement['horizontal'] and not np.isnan(head_movement['horizontal']['radians']):
                recalculated_pos_cropped = remap_position_from_movement(prev_red_pos, head_movement, roi_w, roi_h)
                recalculated_pos = (
                    recalculated_pos_cropped[0] + roi_x,
                    recalculated_pos_cropped[1] + roi_y,
                ) if recalculated_pos_cropped else None
                # if recalculated_pos:
                #     # Compute error in the cropped coordinate space for numerical stability
                #     error_x = (recalculated_pos[0] - roi_x) - curr_red_pos_cropped[0]
                #     error_y = (recalculated_pos[1] - roi_y) - curr_red_pos_cropped[1]
                #     prediction_error = {
                #         "error_x": float(error_x), "error_y": float(error_y),
                #         "distance": float(np.sqrt(error_x**2 + error_y**2))
                #     }
                #     prediction_errors.append(prediction_error["distance"])

        # Draw debug overlays on the full original frame (not cropped)
        if debug:
            # Draw ROI rectangle if detected
            if content_roi is not None:
                cv2.rectangle(
                    vis_frame,
                    (int(roi_x), int(roi_y)),
                    (int(roi_x + roi_w), int(roi_y + roi_h)),
                    (0, 255, 255),
                    2,
                )
            # Draw detected red circle (in red)
            if curr_red_pos is not None:
                cv2.circle(
                    vis_frame,
                    (int(curr_red_pos[0]), int(curr_red_pos[1])),
                    int(curr_radius) if curr_radius else 6,
                    (0, 0, 255),
                    2,
                )
            # Draw recalculated position (in green)
            if recalculated_pos is not None:
                cv2.circle(
                    vis_frame,
                    (int(recalculated_pos[0]), int(recalculated_pos[1])),
                    int(curr_radius) if curr_radius else 6,
                    (0, 255, 0),
                    2,
                )
            # Draw previous detected position (small blue dot) if available
            if prev_red_pos is not None:
                prev_full_x = int(prev_red_pos[0] + roi_x)
                prev_full_y = int(prev_red_pos[1] + roi_y)
                cv2.circle(vis_frame, (prev_full_x, prev_full_y), 3, (255, 0, 0), -1)
            # Optional text info
            info_text = f"Frame {global_frame_offset + (frame_idx_in_video - start_processing_frame)}"
            cv2.putText(
                vis_frame,
                info_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        # Add frame info to output (we already skipped to start_processing_frame)
        # Use only source/global frame index for indexing and timestamps
        source_frame_index = global_frame_offset + (frame_idx_in_video - start_processing_frame)
        frame_info = {
            "frame_index": source_frame_index,
            "timestamp": source_frame_index / fps,
            "social_category": social_category,
            "red_circle": {
                "detected": curr_red_pos is not None,
                "position_full": curr_red_pos,
                "position_content": curr_red_pos_cropped if red_circle else None,
                "radius": curr_radius
            },
            "head_movement": head_movement,
            "next_movement": None,
            "prediction": {"recalculated_position": recalculated_pos}
            # "prediction": {"recalculated_position": recalculated_pos, "error": prediction_error}
        }
        frame_data.append(frame_info)
        
        output_frame_count += 1
        
        if show_video or video_writer:
            if video_writer:
                video_writer.write(vis_frame)
            if show_video:
                cv2.imshow('Frame Analysis', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if curr_red_pos_cropped:
            prev_red_pos = curr_red_pos_cropped
        frame_idx_in_video += 1
        
    cap.release()
    if video_writer:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
        
    clip_meta = {
        "clip_name": str(video_path.name),
        "roi": {"x": int(clip_level_roi[0]), "y": int(clip_level_roi[1]), "w": int(clip_level_roi[2]), "h": int(clip_level_roi[3])} if clip_level_roi else None,
        "frame_size_full": clip_level_frame_size,
    }
    return frame_data, prediction_errors, transition_movement_for_prev, clip_meta

def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def main():
    parser = argparse.ArgumentParser(description='Analyze head movement in video clips and aggregate results.')
    parser.add_argument('input_dir', type=str, help='Directory containing the video clips to process.')
    parser.add_argument('--output_dir', '-o', type=str, default='.', help='Directory to save output JSON and video files.')
    parser.add_argument('--save_video', '-sv', action='store_true', help='Save annotated output videos for each clip.')
    parser.add_argument('--mode', '-m', type=str, default='advanced', choices=['advanced', 'default'], 
                       help='Filename parsing mode: "advanced" for complex pattern, "default" for recording_X_Y.mp4 pattern.')
    parser.add_argument('--fps', '-f', type=float, help='Override video FPS.')
    parser.add_argument('--show', '-s', action='store_true', help='Show video processing in real-time.')
    parser.add_argument('--fov', '-v', type=float, default=104.0, help='Field of view of the video.')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable visual debug overlays and real-time display.')
    args = parser.parse_args()

    # Debug implies showing the video in real-time
    if args.debug:
        args.show = True

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.is_dir():
        print(f"Error: Input path '{input_path}' is not a valid directory.")
        return

    # Find both .MP4 and .mp4 files
    video_files = list(input_path.glob('*.MP4')) + list(input_path.glob('*.mp4'))
    grouped_videos = defaultdict(list)

    for f_path in video_files:
        parsed_info = parse_video_filename(f_path.name, mode=args.mode)
        if parsed_info:
            grouped_videos[parsed_info['group_id']].append((f_path, parsed_info))
        else:
            print(f"Warning: Skipping file with unrecognized name format: {f_path.name}")

    for group_id, files_with_info in grouped_videos.items():
        print(f"\nProcessing group: {group_id}")
        
        # Check if JSON file already exists and skip if it does
        output_json_path = output_path / f"{group_id}_analysis.json"
        if output_json_path.exists():
            print(f"  Skipping group {group_id}: JSON file already exists at {output_json_path}")
            continue
        
        files_with_info.sort(key=lambda x: x[1]['processed_start'])
        
        all_frames_data = []
        all_prediction_errors = []
        
        roi_by_clip: Dict[str, Optional[Dict[str, int]]] = {}
        frame_size_full_by_clip: Dict[str, Dict[str, int]] = {}

        for i, (video_path, info) in enumerate(files_with_info):
            print(f"  - Analyzing clip: {video_path.name}")
            
            is_first_clip = (i == 0)
            
            output_video_file = output_path / f"{video_path.stem}_analysis.mp4" if args.save_video else None
            
            global_offset = info['processed_start']

            clip_frames, clip_errors, transition_movement_for_prev, clip_meta = analyze_video_clip(
                video_path=video_path,
                global_frame_offset=global_offset,
                social_category=info['social_category'], 
                is_first_clip_in_group=is_first_clip,
                last_n_frames=0,
                output_video_path=output_video_file,
                fps_override=args.fps,
                show_video=args.show,
                video_fov_degrees=args.fov,
                debug=args.debug,
            )
            # collect clip-level ROI and frame size
            roi_by_clip[clip_meta["clip_name"]] = clip_meta.get("roi")
            frame_size_full_by_clip[clip_meta["clip_name"]] = clip_meta.get("frame_size_full", {})
            # If this clip starts at an overlapped frame, update the last frame
            # from the previous clip with the computed transition movement
            if transition_movement_for_prev is not None and len(all_frames_data) > 0:
                all_frames_data[-1]['next_movement'] = transition_movement_for_prev
            all_frames_data.extend(clip_frames)
            all_prediction_errors.extend(clip_errors)
            
        if not all_frames_data:
            print(f"Warning: No frames were processed for group {group_id}. Skipping JSON output.")
            continue
            
        analysis_summary = {
            "total_frames_processed": len(all_frames_data),
            "detected_frames": sum(1 for f in all_frames_data if f["red_circle"]["detected"]),
            # "valid_predictions": len(all_prediction_errors),
            "prediction_accuracy": {}
        }
        
        # if all_prediction_errors:
        #       analysis_summary["prediction_accuracy"] = {
        #         "mean_error_px": np.mean(all_prediction_errors),
        #         "median_error_px": np.median(all_prediction_errors),
        #         "std_error_px": np.std(all_prediction_errors),
        #         "min_error_px": np.min(all_prediction_errors),
        #         "max_error_px": np.max(all_prediction_errors)
        #     }
        
        # Consolidate ROI metadata
        unique_rois = {tuple((v["x"], v["y"], v["w"], v["h"])) for v in roi_by_clip.values() if v is not None}
        roi_metadata: Dict[str, Any] = {}
        if len(unique_rois) == 1:
            only_roi_tuple = next(iter(unique_rois)) if unique_rois else None
            roi_metadata["roi"] = {"x": only_roi_tuple[0], "y": only_roi_tuple[1], "w": only_roi_tuple[2], "h": only_roi_tuple[3]} if only_roi_tuple else None
        else:
            roi_metadata["roi_by_clip"] = roi_by_clip

        # Consolidate frame size metadata
        unique_sizes = {tuple((v.get("width"), v.get("height"))) for v in frame_size_full_by_clip.values() if v}
        size_metadata: Dict[str, Any] = {}
        if len(unique_sizes) == 1:
            only_size_tuple = next(iter(unique_sizes)) if unique_sizes else None
            size_metadata["frame_size_full"] = {"width": only_size_tuple[0], "height": only_size_tuple[1]} if only_size_tuple else None
        else:
            size_metadata["frame_size_full_by_clip"] = frame_size_full_by_clip

        output_json_path = output_path / f"{group_id}_analysis.json"
        final_output_data = {
            "metadata": {
                "group_id": group_id,
                "analysis_type": "past_frame_prediction",
                **roi_metadata,
                **size_metadata,
            },
            "analysis_summary": analysis_summary,
            "frames": all_frames_data
        }

        final_output_data = convert_to_json_serializable(final_output_data)

        print(f"Saving aggregated analysis to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(final_output_data, f, indent=2)

if __name__ == "__main__":
    main()