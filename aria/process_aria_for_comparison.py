import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, Any, Optional, List

def format_movement_dict(dx_rad: float, dy_rad: float, dz_rad: float) -> Optional[Dict[str, Any]]:
    if pd.isna(dx_rad):
        return None

    movement = {
        "horizontal_yaw": {"radians": float(dz_rad), "degrees": np.rad2deg(dz_rad)},
        "vertical_pitch": {"radians": float(dy_rad), "degrees": np.rad2deg(dy_rad)},
        "roll": {"radians": float(dx_rad), "degrees": np.rad2deg(dx_rad)}
    }
    return movement


def convert_to_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(v) for v in obj]
    if isinstance(obj, (np.integer, np.int_)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float_)):
        return float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj

def generate_final_comparison(input_path: str, output_path: str, fps: int, expected_frames: Optional[int] = None):
    try:
        cols_to_use = [
            'tracking_timestamp_us',
            'angular_velocity_x_device',
            'angular_velocity_y_device',
            'angular_velocity_z_device'
        ]
        df = pd.read_csv(input_path, usecols=cols_to_use)
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return

    # Ensure sorted by time
    df = df.sort_values('tracking_timestamp_us').reset_index(drop=True)

    # Compute incremental angles
    df['time_delta_s'] = df['tracking_timestamp_us'].diff() / 1_000_000.0
    # First row has NaN delta; replace with 0 so its increments contribute 0
    mean_interval = df['time_delta_s'].dropna().mean()
    df['time_delta_s'].iloc[0] = mean_interval

    df['inc_angle_x'] = df['angular_velocity_x_device'] * df['time_delta_s']
    df['inc_angle_y'] = df['angular_velocity_y_device'] * df['time_delta_s']
    df['inc_angle_z'] = df['angular_velocity_z_device'] * df['time_delta_s']

    frame_duration_us: float = 1_000_000.0 / float(fps)
    start_time_us: int = int(df['tracking_timestamp_us'].iloc[0])
    end_time_us: int = int(df['tracking_timestamp_us'].iloc[-1])

    # Determine number of frames
    if expected_frames is not None and expected_frames > 0:
        num_frames = int(expected_frames)
    else:
        # Include last partial frame by adding 1
        num_frames = int(np.floor((end_time_us - start_time_us) / frame_duration_us)) + 1

    # Map each sample to a frame bin (floor); clamp to [0, num_frames-1]
    df['frame_index'] = np.floor((df['tracking_timestamp_us'] - start_time_us) / frame_duration_us).astype(int)
    df['frame_index'] = df['frame_index'].clip(lower=0, upper=max(0, num_frames - 1))

    angle_sum_cols = ['inc_angle_x', 'inc_angle_y', 'inc_angle_z']
    grouped = df.groupby('frame_index', as_index=True)[angle_sum_cols].sum()
    # Reindex to full range [0, num_frames-1], fill missing with 0
    grouped = grouped.reindex(range(num_frames), fill_value=0.0)

    # For timestamps, define canonical per-frame timestamp: start_time + i * frame_duration
    frame_timestamps_s = (start_time_us + np.arange(num_frames) * frame_duration_us) / 1_000_000.0

    all_frames_data: List[Dict[str, Any]] = []
    for i in range(num_frames):
        row = grouped.iloc[i]
        dx_rad = float(row['inc_angle_x'])
        dy_rad = float(row['inc_angle_y'])
        dz_rad = float(row['inc_angle_z'])

        current_movement = format_movement_dict(dx_rad, dy_rad, dz_rad)

        frame_info = {
            "frame_index": int(i),
            "timestamp": float(frame_timestamps_s[i]),
            "source": "aria_ground_truth",
            "head_movement": current_movement,
            "next_movement": None,
        }
        all_frames_data.append(frame_info)

        if i > 0:
            all_frames_data[i - 1]['next_movement'] = current_movement
    
    final_output = {
        "metadata": {
            "source_file": input_path,
            "analysis_type": "ground_truth_integrated_angle",
            "fps": float(fps),
            "start_time_us": int(start_time_us),
            "end_time_us": int(end_time_us),
            "frame_duration_us": float(frame_duration_us),
            "total_frames": int(len(all_frames_data)),
        },
        "frames": all_frames_data,
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(convert_to_json_serializable(final_output), f, indent=2)
        print(f"Successfully generated comparison JSON at '{output_path}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison JSON with integrated delta angles from Aria data.")
    parser.add_argument("input_file", type=str, help="Input Aria CSV file")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second to analyze")
    parser.add_argument("--expected_frames", type=int, default=None, help="Optionally force total number of frames (e.g., video frame count)")
    args = parser.parse_args()

    generate_final_comparison(args.input_file, args.output_file, args.fps, args.expected_frames)