import json
import pandas as pd

# Load the first JSON file: the ground truth given by ARIA
with open('/Volumes/T7 Shield Portable/Github/ego-datasets/aria/closed_loop_trajectory_20fps.json', 'r') as f:
    data_ground_truth = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_ground_truth = pd.json_normalize(data_ground_truth['frames'])
df_ground_truth = df_ground_truth[['frame_index', 'head_movement.horizontal_yaw.degrees', 'head_movement.vertical_pitch.degrees']]
df_ground_truth.rename(columns={
    'head_movement.horizontal_yaw.degrees': 'ground_truth_horizontal_yaw',
    'head_movement.vertical_pitch.degrees': 'ground_truth_vertical_pitch'
}, inplace=True)


# Load the second JSON file: the co-tracker analysis
with open('/Volumes/T7 Shield Portable/Github/ego-datasets/aria/recording/recording_analysis.json', 'r') as f:
    data_recording_analysis = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_recording_analysis = pd.json_normalize(data_recording_analysis['frames'])
df_recording_analysis = df_recording_analysis[['frame_index', 'head_movement.horizontal.degrees', 'head_movement.vertical.degrees']]
df_recording_analysis.rename(columns={
    'head_movement.horizontal.degrees': 'analysis_horizontal',
    'head_movement.vertical.degrees': 'analysis_vertical'
}, inplace=True)

# Merge the two DataFrames on 'frame_index'
df_merged = pd.merge(df_ground_truth, df_recording_analysis, on='frame_index')

# Calculate the difference (error) between the ground truth and recording analysis
df_merged['horizontal_error'] = df_merged['ground_truth_horizontal_yaw'] - df_merged['analysis_horizontal']
df_merged['vertical_error'] = df_merged['ground_truth_vertical_pitch'] - df_merged['analysis_vertical']

# Display summary statistics of the errors
print("Ground Truth vs. Recording Analysis Accuracy (in degrees)")
print("\nHorizontal Movement (Yaw) Error:")
print(df_merged['horizontal_error'].describe())

print("\nVertical Movement (Pitch) Error:")
print(df_merged['vertical_error'].describe())

# Save the merged dataframe to a CSV file for inspection if needed.
df_merged.to_csv('ground_truth_vs_analysis_comparison.csv', index=False)

# Plotting the data for visual comparison
import matplotlib.pyplot as plt

# Plot horizontal movement comparison
plt.figure(figsize=(12, 6))
plt.plot(df_merged['frame_index'], df_merged['ground_truth_horizontal_yaw'], label='Ground Truth (Yaw)')
plt.plot(df_merged['frame_index'], df_merged['analysis_horizontal'], label='Recording Analysis (Horizontal)', linestyle='--')
plt.xlabel('Frame Index')
plt.ylabel('Degrees')
plt.title('Horizontal Movement: Ground Truth vs. Recording Analysis')
plt.legend()
plt.grid(True)
plt.savefig('horizontal_movement_comparison.png')
plt.close()

# Plot vertical movement comparison
plt.figure(figsize=(12, 6))
plt.plot(df_merged['frame_index'], df_merged['ground_truth_vertical_pitch'], label='Ground Truth (Pitch)')
plt.plot(df_merged['frame_index'], df_merged['analysis_vertical'], label='Recording Analysis (Vertical)', linestyle='--')
plt.xlabel('Frame Index')
plt.ylabel('Degrees')
plt.title('Vertical Movement: Ground Truth vs. Recording Analysis')
plt.legend()
plt.grid(True)
plt.savefig('vertical_movement_comparison.png')
plt.close()

# Scatter plot for horizontal movement
plt.figure(figsize=(8, 8))
plt.scatter(df_merged['ground_truth_horizontal_yaw'], df_merged['analysis_horizontal'], alpha=0.5)
plt.xlabel('Ground Truth Horizontal Yaw (degrees)')
plt.ylabel('Recording Analysis Horizontal (degrees)')
plt.title('Correlation: Horizontal Movement')
plt.grid(True)
plt.plot([-30, 30], [-30, 30], 'r--')  # Add a y=x line for reference
plt.axis('equal')
plt.savefig('horizontal_correlation.png')
plt.close()


# Scatter plot for vertical movement
plt.figure(figsize=(8, 8))
plt.scatter(df_merged['ground_truth_vertical_pitch'], df_merged['analysis_vertical'], alpha=0.5)
plt.xlabel('Ground Truth Vertical Pitch (degrees)')
plt.ylabel('Recording Analysis Vertical (degrees)')
plt.title('Correlation: Vertical Movement')
plt.grid(True)
plt.plot([-20, 20], [-20, 20], 'r--')  # Add a y=x line for reference
plt.axis('equal')
plt.savefig('vertical_correlation.png')
plt.close()

# Plotting error distributions
plt.figure(figsize=(12, 6))
plt.hist(df_merged['horizontal_error'], bins=50, alpha=0.7, label='Horizontal Error')
plt.hist(df_merged['vertical_error'], bins=50, alpha=0.7, label='Vertical Error')
plt.xlabel('Error (degrees)')
plt.ylabel('Frequency')
plt.title('Distribution of Errors (Ground Truth - Analysis)')
plt.legend()
plt.grid(True)
plt.savefig('error_distribution.png')
plt.close()