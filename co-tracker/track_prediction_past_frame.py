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
    video_fov_degrees: float = 104.0
) -> Tuple[List[Dict[str, Any]], List[float]]:
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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Skip frames until we reach the start processing frame
        if frame_idx_in_video < start_processing_frame:
            frame_idx_in_video += 1
            continue

        if not is_first_clip_in_group and frame_idx_in_video == start_processing_frame:
            red_circle = detect_red_circle(frame)
            if red_circle:
                prev_red_pos = (float(red_circle[0]), float(red_circle[1]))
            frame_idx_in_video += 1
            continue

        vis_frame = frame.copy()
        red_circle = detect_red_circle(frame)
        
        curr_red_pos = (float(red_circle[0]), float(red_circle[1])) if red_circle else None
        curr_radius = int(red_circle[2]) if red_circle else None
        
        head_movement, recalculated_pos, prediction_error = None, None, None

        if prev_red_pos and curr_red_pos:
            head_movement = calculate_head_movement(prev_red_pos, curr_red_pos, width, height, video_fov_degrees=video_fov_degrees)
            if head_movement and frame_data:
                frame_data[-1]['next_movement'] = head_movement

            if head_movement and 'horizontal' in head_movement and 'radians' in head_movement['horizontal'] and not np.isnan(head_movement['horizontal']['radians']):
                recalculated_pos = remap_position_from_movement(prev_red_pos, head_movement, width, height)
                if recalculated_pos:
                    error_x = recalculated_pos[0] - curr_red_pos[0]
                    error_y = recalculated_pos[1] - curr_red_pos[1]
                    prediction_error = {
                        "error_x": float(error_x), "error_y": float(error_y),
                        "distance": float(np.sqrt(error_x**2 + error_y**2))
                    }
                    prediction_errors.append(prediction_error["distance"])

        # Add frame info to output (we already skipped to start_processing_frame)
        frame_info = {
            "frame_index": global_frame_offset + (frame_idx_in_video - start_processing_frame),
            "timestamp": (global_frame_offset + (frame_idx_in_video - start_processing_frame)) / fps,
            "social_category": social_category,
            "red_circle": {"detected": curr_red_pos is not None, "position": curr_red_pos, "radius": curr_radius},
            "head_movement": head_movement,
            "next_movement": None,
            "prediction": {"recalculated_position": recalculated_pos, "error": prediction_error}
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
        
        if curr_red_pos:
            prev_red_pos = curr_red_pos
        frame_idx_in_video += 1
        
    cap.release()
    if video_writer:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
        
    return frame_data, prediction_errors

def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
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
    args = parser.parse_args()

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
        
        files_with_info.sort(key=lambda x: x[1]['processed_start'])
        
        all_frames_data = []
        all_prediction_errors = []
        
        for i, (video_path, info) in enumerate(files_with_info):
            print(f"  - Analyzing clip: {video_path.name}")
            
            is_first_clip = (i == 0)
            
            output_video_file = output_path / f"{video_path.stem}_analysis.mp4" if args.save_video else None
            
            # Calculate expected frames from filename (for last N frames processing)
            expected_frames = info['end_frame'] - info['start_frame']
            if expected_frames <= 0:
                expected_frames = 0  # Process all frames if calculation is invalid
            
            global_offset = info['processed_start']

            clip_frames, clip_errors = analyze_video_clip(
                video_path=video_path,
                global_frame_offset=global_offset,
                social_category=info['social_category'], 
                is_first_clip_in_group=is_first_clip,
                last_n_frames=expected_frames,
                output_video_path=output_video_file,
                fps_override=args.fps,
                show_video=args.show,
                video_fov_degrees=args.fov
            )
            all_frames_data.extend(clip_frames)
            all_prediction_errors.extend(clip_errors)
            
        if not all_frames_data:
            print(f"Warning: No frames were processed for group {group_id}. Skipping JSON output.")
            continue
            
        analysis_summary = {
            "total_frames_processed": len(all_frames_data),
            "detected_frames": sum(1 for f in all_frames_data if f["red_circle"]["detected"]),
            "valid_predictions": len(all_prediction_errors),
            "prediction_accuracy": {}
        }
        
        if all_prediction_errors:
            analysis_summary["prediction_accuracy"] = {
                "mean_error_px": np.mean(all_prediction_errors),
                "median_error_px": np.median(all_prediction_errors),
                "std_error_px": np.std(all_prediction_errors),
                "min_error_px": np.min(all_prediction_errors),
                "max_error_px": np.max(all_prediction_errors)
            }
        
        output_json_path = output_path / f"{group_id}_analysis.json"
        final_output_data = {
            "metadata": {"group_id": group_id, "analysis_type": "past_frame_prediction"},
            "analysis_summary": analysis_summary,
            "frames": all_frames_data
        }

        final_output_data = convert_to_json_serializable(final_output_data)

        print(f"Saving aggregated analysis to: {output_json_path}")
        with open(output_json_path, 'w') as f:
            json.dump(final_output_data, f, indent=2)

if __name__ == "__main__":
    main()