import pandas as pd
import argparse

def calculate_delta_angle(input_path: str, output_path: str, fps: int):
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
    except ValueError as e:
        print(f"Error: CSV file is missing required columns. Details: {e}")
        return

    if df.empty:
        print("Error: The input file is empty.")
        return

    frame_duration_us = 1_000_000 / fps
    start_time_us = df['tracking_timestamp_us'].iloc[0]

    df['frame_index'] = ((df['tracking_timestamp_us'] - start_time_us) / frame_duration_us).astype(int)

    velocity_cols = [
        'angular_velocity_x_device',
        'angular_velocity_y_device',
        'angular_velocity_z_device'
    ]

    avg_rad_per_second = df.groupby('frame_index')[velocity_cols].mean()

    frame_duration_s = 1.0 / fps

    delta_angle_per_frame = avg_rad_per_second * frame_duration_s

    delta_angle_per_frame.columns = [
        'delta_angle_x_rad',
        'delta_angle_y_rad',
        'delta_angle_z_rad'
    ]
    
    try:
        delta_angle_per_frame.to_csv(output_path)
        print(f"Successfully saved delta angles to '{output_path}'")
        print("\n--- Delta Radian Angle per 20 FPS Frame ---")
        print(delta_angle_per_frame.head(15).to_markdown())
    except Exception as e:
        print(f"Error saving file: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate delta angle per frame from Aria data for a specific FPS."
    )
    parser.add_argument("input_file", type=str, help="Input Aria CSV file (e.g., closed_loop_trajectory.csv)")
    parser.add_argument("output_file", type=str, help="Output CSV file path (e.g., delta_angles.csv)")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second to analyze (default: 20)")
    args = parser.parse_args()
    
    calculate_delta_angle(args.input_file, args.output_file, args.fps)