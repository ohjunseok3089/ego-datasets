import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, Any, Optional

def calculate_aria_movement(dx: float, dy: float, dz: float) -> Optional[Dict[str, Any]]:
    if pd.isna(dx) or (dx == 0 and dy == 0 and dz == 0):
        return None

    horizontal_rad = np.arctan2(dy, dx)
    horizontal_deg = np.rad2deg(horizontal_rad)

    xy_displacement = np.sqrt(dx**2 + dy**2)
    vertical_rad = np.arctan2(dz, xy_displacement)
    vertical_deg = np.rad2deg(vertical_rad)

    movement = {
        "horizontal": {"radians": float(horizontal_rad), "degrees": float(horizontal_deg)},
        "vertical": {"radians": float(vertical_rad), "degrees": float(vertical_deg)},
        "raw_delta_m": {"dx": float(dx), "dy": float(dy), "dz": float(dz)}
    }
    return movement

def process_aria_by_timestamp(input_path: str, output_path: str):
    try:
        cols_to_use = [
            'tracking_timestamp_us', 
            'tx_world_device', 
            'ty_world_device', 
            'tz_world_device'
        ]
        df = pd.read_csv(input_path, usecols=cols_to_use)
    except FileNotFoundError:
        print(f"Error: Input file '{input_path}' not found.")
        return
    except ValueError as e:
        print(f"Error: CSV file is missing required columns. Details: {e}")
        return

    time_unique_df = df.groupby('tracking_timestamp_us').last().reset_index()

    time_unique_df['delta_x'] = time_unique_df['tx_world_device'].diff()
    time_unique_df['delta_y'] = time_unique_df['ty_world_device'].diff()
    time_unique_df['delta_z'] = time_unique_df['tz_world_device'].diff()

    all_frames_data = []
    
    for i, row in time_unique_df.iterrows():
        dx, dy, dz = row['delta_x'], row['delta_y'], row['delta_z']
        current_movement = calculate_aria_movement(dx, dy, dz)

        frame_info = {
            "frame_index": i,
            "timestamp": row['tracking_timestamp_us'] / 1_000_000.0,
            "source": "aria_ground_truth",
            "head_movement": current_movement,
            "next_movement": None
        }
        all_frames_data.append(frame_info)
        
        if i > 0:
            all_frames_data[i-1]['next_movement'] = current_movement
    
    final_output = {
        "metadata": {
            "source_file": input_path,
            "analysis_type": "ground_truth_trajectory_by_timestamp"
        },
        "frames": all_frames_data
    }

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
        print(f"Success: Timestamp-based delta analysis results saved to '{output_path}' file.")
    except Exception as e:
        print(f"Error: JSON file saving failed. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze Aria trajectory CSV by timestamp to calculate position deltas."
    )
    parser.add_argument("input_file", type=str, help="Input CSV file path")
    parser.add_argument("output_file", type=str, help="Output JSON file path")
    args = parser.parse_args()
    process_aria_by_timestamp(args.input_file, args.output_file)