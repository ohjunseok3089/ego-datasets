import pandas as pd
import numpy as np
import json
import argparse
from typing import Dict, Any, Optional

def calculate_aria_movement(dx: float, dy: float, dz: float) -> Optional[Dict[str, Any]]:
    """
    Converts 3D position deltas (dx, dy, dz) into a 'head_movement'
    dictionary with horizontal (yaw) and vertical (pitch) angles.
    """
    # If there is no movement, return None
    if pd.isna(dx) or (dx == 0 and dy == 0 and dz == 0):
        return None

    # Horizontal movement (Yaw) in the XY plane
    horizontal_rad = np.arctan2(dy, dx)
    horizontal_deg = np.rad2deg(horizontal_rad)

    # Vertical movement (Pitch)
    xy_displacement = np.sqrt(dx**2 + dy**2)
    vertical_rad = np.arctan2(dz, xy_displacement)
    vertical_deg = np.rad2deg(vertical_rad)

    movement = {
        "horizontal": {"radians": float(horizontal_rad), "degrees": float(horizontal_deg)},
        "vertical": {"radians": float(vertical_rad), "degrees": float(vertical_deg)},
        "raw_delta_m": {"dx": float(dx), "dy": float(dy), "dz": float(dz)}
    }
    return movement

def process_aria_to_match_cv(input_path: str, output_path: str):
    """
    Reads Aria ground truth trajectory data and formats it into a JSON structure
    that matches the provided computer vision analysis script for easy comparison.
    """
    try:
        # We need timestamp for timing and tx/ty/tz for position
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

    # Calculate frame-to-frame difference for each coordinate
    df['delta_x'] = df['tx_world_device'].diff()
    df['delta_y'] = df['ty_world_device'].diff()
    df['delta_z'] = df['tz_world_device'].diff()

    all_frames_data = []

    # Iterate through the dataframe to build the frame-by-frame JSON structure
    for i in range(len(df)):
        # Movement that *led to* this frame (from i-1 to i)
        dx, dy, dz = df.at[i, 'delta_x'], df.at[i, 'delta_y'], df.at[i, 'delta_z']
        current_movement = calculate_aria_movement(dx, dy, dz)

        # Create the basic structure for the current frame
        frame_info = {
            "frame_index": i,
            "timestamp": df.at[i, 'tracking_timestamp_us'] / 1_000_000.0, # Convert us to s
            "source": "aria_ground_truth",
            "head_movement": current_movement,
            "next_movement": None # Will be populated by the next iteration
        }
        all_frames_data.append(frame_info)
        
        # This is the key part: Go back and populate the 'next_movement' of the *previous* frame
        # with the movement we just calculated for the current frame.
        if i > 0:
            all_frames_data[i-1]['next_movement'] = current_movement
    
    # Final JSON structure to match the CV script's output
    final_output = {
        "metadata": {
            "source_file": input_path,
            "analysis_type": "ground_truth_trajectory_reformatted"
        },
        "frames": all_frames_data
    }

    # Save to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, indent=2)
        print(f"Successfully converted Aria data to '{output_path}' for comparison.")
    except Exception as e:
        print(f"Error: Could not write to JSON file. Details: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
        Converts Aria ground truth trajectory data into a JSON format that matches the
        output of a computer vision analysis script for direct comparison.
        """
    )
    parser.add_argument(
        "input_file", 
        type=str, 
        help="Input Aria CSV file (e.g., closed_loop_trajectory.csv)"
    )
    parser.add_argument(
        "output_file", 
        type=str, 
        help="Output JSON file path (e.g., aria_ground_truth.json)"
    )

    args = parser.parse_args()
    process_aria_to_match_cv(args.input_file, args.output_file)