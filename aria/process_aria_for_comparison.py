import pandas as pd
import numpy as np
import argparse

def comprehensive_analysis(input_path: str, output_path: str, fps: int):
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
    
    df['abs_inc_angle_x'] = df['inc_angle_x'].abs()
    df['abs_inc_angle_y'] = df['inc_angle_y'].abs()
    df['abs_inc_angle_z'] = df['inc_angle_z'].abs()

    frame_duration_us = 1_000_000 / fps
    start_time_us = df['tracking_timestamp_us'].iloc[0]
    df['frame_index'] = ((df['tracking_timestamp_us'] - start_time_us) / frame_duration_us).astype(int)

    results = df.groupby('frame_index').agg(
        avg_vel_x=('angular_velocity_x_device', 'mean'),
        avg_vel_y=('angular_velocity_y_device', 'mean'),
        avg_vel_z=('angular_velocity_z_device', 'mean'),
        
        net_angle_x=('inc_angle_x', 'sum'),
        net_angle_y=('inc_angle_y', 'sum'),
        net_angle_z=('inc_angle_z', 'sum'),
        
        total_rotation_x=('abs_inc_angle_x', 'sum'),
        total_rotation_y=('abs_inc_angle_y', 'sum'),
        total_rotation_z=('abs_inc_angle_z', 'sum')
    )
    
    try:
        results.to_csv(output_path)
        print(f"Analysis complete. All results saved to '{output_path}'.")
        print("\n--- Comprehensive Analysis Results (first 15 frames) ---")
        print(results.head(15).to_markdown())
    except Exception as e:
        print(f"Error saving file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a comprehensive analysis on Aria angular velocity data.")
    parser.add_argument("input_file", type=str, help="Input Aria CSV file")
    parser.add_argument("output_file", type=str, help="Output CSV file path for all metrics")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second to analyze")
    args = parser.parse_args()
    
    comprehensive_analysis(args.input_file, args.output_file, args.fps)