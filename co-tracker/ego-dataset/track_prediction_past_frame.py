#!/usr/bin/env python3
"""
Processes video clips to track a red circle, calculates head movement,
and aggregates the analysis results from multiple clips into a single JSON file.
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
# You need to have a `track_red.py` file with these functions defined.
# from track_red import detect_red_circle, calculate_head_movement, remap_position_from_movement

# --- Mock functions if track_red.py is not available ---
def detect_red_circle(frame):
    # This is a placeholder. Use your actual detection logic.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 70])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red, upper_red)
    mask = mask1 + mask2
    circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=15, minRadius=5, maxRadius=50)
    if circles is not None:
        return circles[0][0]
    return None

def calculate_head_movement(prev_pos, curr_pos, width, height):
    # This is a placeholder.
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    return {"horizontal": {"degrees": dx}, "vertical": {"degrees": dy}, "radians": {}}

def remap_position_from_movement(prev_pos, movement, width, height):
    # This is a placeholder.
    return (prev_pos[0] + movement['horizontal']['degrees'], prev_pos[1] + movement['vertical']['degrees'])
# --- End of mock functions ---


def parse_video_filename(filename: str) -> Optional[Dict[str, Any]]:
    """
    Parses a video filename to extract metadata based on a specific pattern.

    Args:
        filename: The name of the video file.

    Pattern Example: 'vid_003__day_1__con_1__person_1_part3(1140_1800_social_interaction)_0_15.MP4'
    
    Returns:
        A dictionary with parsed components or None if the pattern doesn't match.
    """
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
    data['social_category'] = data.pop('category') # Explicitly handle social_category
    
    for key in ['start_frame', 'end_frame', 'processed_start', 'processed_end']:
        data[key] = int(data[key])
        
    return data

def analyze_video_clip(
    video_path: Path, 
    global_frame_offset: int, 
    social_category: str, # <-- ADDED: Pass category to the function
    frozen_frame_skip: int = 0,
    output_video_path: Optional[Path] = None, 
    fps_override: Optional[float] = None, 
    show_video: bool = False
) -> Tuple[List[Dict[str, Any]], List[float]]:
    """
    Processes a single video clip to detect a red circle and analyze movement.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {video_path}")
        return [], []
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detected_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps_override if fps_override is not None else detected_fps
    
    video_writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter.fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
    
    frame_data = []
    prediction_errors = []
    prev_red_pos = None
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        vis_frame = frame.copy()
        red_circle = detect_red_circle(frame)
        
        curr_red_pos = (float(red_circle[0]), float(red_circle[1])) if red_circle else None
        curr_radius = int(red_circle[2]) if red_circle else None
        
        head_movement, recalculated_pos, prediction_error = None, None, None

        if prev_red_pos and curr_red_pos:
            head_movement = calculate_head_movement(prev_red_pos, curr_red_pos, width, height)
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

        if frame_idx >= frozen_frame_skip:
            frame_info = {
                "frame_index": global_frame_offset + frame_idx,
                "timestamp": (global_frame_offset + frame_idx) / fps,
                "social_category": social_category, # <-- ADDED: Include category in frame data
                "red_circle": {"detected": curr_red_pos is not None, "position": curr_red_pos, "radius": curr_radius},
                "head_movement": head_movement,
                "prediction": {"recalculated_position": recalculated_pos, "error": prediction_error}
            }
            frame_data.append(frame_info)
        
        if show_video or video_writer:
            if curr_red_pos:
                cv2.circle(vis_frame, (int(curr_red_pos[0]), int(curr_red_pos[1])), curr_radius or 5, (0, 0, 255), 2)
            if recalculated_pos:
                cv2.circle(vis_frame, (int(recalculated_pos[0]), int(recalculated_pos[1])), 5, (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Frame: {global_frame_offset + frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if video_writer:
                video_writer.write(vis_frame)
            if show_video:
                cv2.imshow('Frame Analysis', vis_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        if curr_red_pos:
            prev_red_pos = curr_red_pos
        frame_idx += 1
        
    cap.release()
    if video_writer:
        video_writer.release()
    if show_video:
        cv2.destroyAllWindows()
        
    return frame_data, prediction_errors

def convert_to_json_serializable(obj: Any) -> Any:
    """Recursively converts numpy types and NaN to JSON-friendly formats."""
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def main():
    """Main function to parse arguments and orchestrate the video processing."""
    parser = argparse.ArgumentParser(description='Analyze head movement in video clips and aggregate results.')
    parser.add_argument('input_dir', type=str, help='Directory containing the video clips to process.')
    parser.add_argument('--output_dir', '-o', type=str, default='.', help='Directory to save output JSON and video files.')
    parser.add_argument('--save_video', '-sv', action='store_true', help='Save annotated output videos for each clip.')
    parser.add_argument('--frozen_frame', action='store_true', help='Ignore the first 7 frames of each clip in the JSON output.')
    parser.add_argument('--fps', '-f', type=float, help='Override video FPS.')
    parser.add_argument('--show', '-s', action='store_true', help='Show video processing in real-time.')
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not input_path.is_dir():
        print(f"Error: Input path '{input_path}' is not a valid directory.")
        return

    video_files = list(input_path.glob('*.MP4'))
    grouped_videos = defaultdict(list)

    for f_path in video_files:
        parsed_info = parse_video_filename(f_path.name)
        if parsed_info:
            grouped_videos[parsed_info['group_id']].append((f_path, parsed_info))
        else:
            print(f"Warning: Skipping file with unrecognized name format: {f_path.name}")

    for group_id, files_with_info in grouped_videos.items():
        print(f"\nProcessing group: {group_id}")
        
        files_with_info.sort(key=lambda x: x[1]['processed_start'])
        
        all_frames_data = []
        all_prediction_errors = []
        
        for video_path, info in files_with_info:
            print(f"  - Analyzing clip: {video_path.name}")
            
            output_video_file = output_path / f"{video_path.stem}_analysis.mp4" if args.save_video else None
            frozen_skip_count = 7 if args.frozen_frame else 0
            global_offset = info['start_frame']
            
            # <-- MODIFIED: Pass the 'social_category' from parsed info
            clip_frames, clip_errors = analyze_video_clip(
                video_path=video_path,
                global_frame_offset=global_offset,
                social_category=info['social_category'], 
                frozen_frame_skip=frozen_skip_count,
                output_video_path=output_video_file,
                fps_override=args.fps,
                show_video=args.show
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
                "mean_error_px": float(np.mean(all_prediction_errors)),
                "median_error_px": float(np.median(all_prediction_errors)),
                "std_error_px": float(np.std(all_prediction_errors)),
                "min_error_px": float(np.min(all_prediction_errors)),
                "max_error_px": float(np.max(all_prediction_errors))
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