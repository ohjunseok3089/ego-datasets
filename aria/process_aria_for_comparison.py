import pandas as pd
import numpy as np
import argparse

def comprehensive_analysis(input_path: str, output_path: str, fps: int):
    """
    Calculates three different metrics from angular velocity:
    1. Average angular velocity.
    2. Net delta angle (via integration).
    3. Total rotation path (via integration of absolute velocity).
    """
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

    # --- Step 1: Pre-calculation for all metrics ---
    # Calculate time delta (dt) in SECONDS between each high-frequency reading
    df['time_delta_s'] = df['tracking_timestamp_us'].diff() / 1_000_000.0

    # Create columns for incremental angles (for Net Delta Angle)
    df['inc_angle_x'] = df['angular_velocity_x_device'] * df['time_delta_s']
    df['inc_angle_y'] = df['angular_velocity_y_device'] * df['time_delta_s']
    df['inc_angle_z'] = df['angular_velocity_z_device'] * df['time_delta_s']
    
    # Create columns for absolute incremental angles (for Total Rotation Path)
    df['abs_inc_angle_x'] = df['inc_angle_x'].abs()
    df['abs_inc_angle_y'] = df['inc_angle_y'].abs()
    df['abs_inc_angle_z'] = df['inc_angle_z'].abs()

    # --- Step 2: Bin data and aggregate all metrics at once ---
    frame_duration_us = 1_000_000 / fps
    start_time_us = df['tracking_timestamp_us'].iloc[0]
    df['frame_index'] = ((df['tracking_timestamp_us'] - start_time_us) / frame_duration_us).astype(int)

    # Use .agg() to calculate multiple aggregations simultaneously
    results = df.groupby('frame_index').agg(
        # Metric 1: Average Velocity
        avg_vel_x=('angular_velocity_x_device', 'mean'),
        avg_vel_y=('angular_velocity_y_device', 'mean'),
        avg_vel_z=('angular_velocity_z_device', 'mean'),
        
        # Metric 2: Net Delta Angle
        net_angle_x=('inc_angle_x', 'sum'),
        net_angle_y=('inc_angle_y', 'sum'),
        net_angle_z=('inc_angle_z', 'sum'),
        
        # Metric 3: Total Rotation Path
        total_rotation_x=('abs_inc_angle_x', 'sum'),
        total_rotation_y=('abs_inc_angle_y', 'sum'),
        total_rotation_z=('abs_inc_angle_z', 'sum')
    )
    
    try:
        results.to_csv(output_path)
        print(f"✅ 분석 완료. 모든 결과가 '{output_path}'에 저장되었습니다.")
        print("\n--- 종합 분석 결과 (처음 15개 프레임) ---")
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