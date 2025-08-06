import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, Any, Optional

def format_movement_dict(dx_rad: float, dy_rad: float, dz_rad: float) -> Optional[Dict[str, Any]]:
    if pd.isna(dx_rad):
        return None

    movement = {
        "horizontal_yaw": {"radians": float(dz_rad), "degrees": np.rad2deg(dz_rad)},
        "vertical_pitch": {"radians": float(dy_rad), "degrees": np.rad2deg(dy_rad)},
        "roll": {"radians": float(dx_rad), "degrees": np.rad2deg(dx_rad)}
    }
    return movement

def generate_final_comparison(input_path: str, output_path: str, fps: int):
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

    df['time_delta_s'] = df['tracking_timestamp_us'].diff() / 1_000_000.0

    df['inc_angle_x'] = df['angular_velocity_x_device'] * df['time_delta_s']
    df['inc_angle_y'] = df['angular_velocity_y_device'] * df['time_delta_s']
    df['inc_angle_z'] = df['angular_velocity_z_device'] * df['time_delta_s']

    frame_duration_us = 1_000_000 / fps
    start_time_us = df['tracking_timestamp_us'].iloc[0]
    df['frame_index'] = ((df['tracking_timestamp_us'] - start_time_us) / frame_duration_us).astype(int)

    angle_sum_cols = ['inc_angle_x', 'inc_angle_y', 'inc_angle_z']
    total_delta_angle_per_frame = df.groupby('frame_index')[angle_sum_cols].sum()
    
    all_frames_data = []
    frame_timestamps = df.groupby('frame_index')['tracking_timestamp_us'].first() / 1_000_000.0
    
    for i in range(len(total_delta_angle_per_frame)):
        row = total_delta_angle_per_frame.iloc[i]
        dx_rad, dy_rad, dz_rad = row['inc_angle_x'], row['inc_angle_y'], row['inc_angle_z']
        
        current_movement = format_movement_dict(dx_rad, dy_rad, dz_rad)

        frame_info = {
            "frame_index": i,
            "timestamp": frame_timestamps.get(i, 0.0),
            "source": "aria_ground_truth",
            "head_movement": current_movement,
            "next_movement": None
        }
        all_frames_data.append(frame_info)
        
        if i > 0:
            all_frames_data[i-1]['next_movement'] = current_movement
    
    final_output = {
        "metadata": {"source_file": input_path, "analysis_type": "ground_truth_integrated_angle"},
        "frames": all_frames_data
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
        print(f"Successfully generated comparison JSON at '{output_path}'.")
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a comparison JSON with integrated delta angles from Aria data.")
    parser.add_argument("input_file", type=str, help="Input Aria CSV file")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second to analyze")
    args = parser.parse_args()
    
    generate_final_comparison(args.input_file, args.output_file, args.fps)