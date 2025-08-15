import json
import pandas as pd

# Load the first JSON file: the ground truth given by ARIA
with open('./aria/closed_loop_trajectory_20fps.json', 'r') as f:
    data_ground_truth = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_ground_truth = pd.json_normalize(data_ground_truth['frames'])
df_ground_truth = df_ground_truth[['frame_index', 'head_movement.horizontal_yaw.degrees', 'head_movement.vertical_pitch.degrees']]
df_ground_truth.rename(columns={
    'head_movement.horizontal_yaw.degrees': 'ground_truth_horizontal_yaw',
    'head_movement.vertical_pitch.degrees': 'ground_truth_vertical_pitch'
}, inplace=True)


# Load the second JSON file: the co-tracker analysis
with open('./aria/recording_analysis.json', 'r') as f:
    data_recording_analysis = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_recording_analysis = pd.json_normalize(data_recording_analysis['frames'])
df_recording_analysis = df_recording_analysis[['frame_index', 'head_movement.horizontal.degrees', 'head_movement.vertical.degrees']]
df_recording_analysis.rename(columns={
    'head_movement.horizontal.degrees': 'analysis_horizontal',
    'head_movement.vertical.degrees': 'analysis_vertical'
}, inplace=True)

# Handle temporal alignment by removing first N frames from analysis data
print("=== DATA ALIGNMENT ANALYSIS ===")
print(f"Ground Truth: {len(df_ground_truth)} frames (range: {df_ground_truth['frame_index'].min()}-{df_ground_truth['frame_index'].max()})")
print(f"Analysis: {len(df_recording_analysis)} frames (range: {df_recording_analysis['frame_index'].min()}-{df_recording_analysis['frame_index'].max()})")

# Dynamically calculate the frame offset based on the difference in dataset sizes
gt_frame_count = len(df_ground_truth)
analysis_frame_count = len(df_recording_analysis)
frames_to_remove = analysis_frame_count - gt_frame_count

print(f"Frame count difference: {frames_to_remove} (Analysis has {frames_to_remove} more frames than Ground Truth)")
print(f"Removing first {frames_to_remove} frames from analysis data to align temporal windows...")

# Drop the first N frames by frame_index
df_recording_analysis_shifted = df_recording_analysis[df_recording_analysis['frame_index'] >= frames_to_remove].copy()

# Shift the frame indices down by N to align with ground truth
df_recording_analysis_shifted['frame_index'] = df_recording_analysis_shifted['frame_index'] - frames_to_remove

print(f"Analysis (after removing first {frames_to_remove} frames): {len(df_recording_analysis_shifted)} frames")
print(f"Analysis (shifted frame range): {df_recording_analysis_shifted['frame_index'].min()}-{df_recording_analysis_shifted['frame_index'].max()}")

# Remove any NaN values 
df_recording_analysis_clean = df_recording_analysis_shifted.dropna(subset=['analysis_horizontal', 'analysis_vertical'])
print(f"Analysis (after removing NaN): {len(df_recording_analysis_clean)} frames")

# Now merge the aligned datasets
df_merged = pd.merge(df_ground_truth, df_recording_analysis_clean, on='frame_index', how='inner')
print(f"Final merged dataset: {len(df_merged)} frames")

# Calculate the difference (error) between the ground truth and recording analysis
df_merged['horizontal_error'] = df_merged['ground_truth_horizontal_yaw'] - df_merged['analysis_horizontal']
df_merged['vertical_error'] = df_merged['ground_truth_vertical_pitch'] - df_merged['analysis_vertical']

# Calculate absolute errors for outlier removal
df_merged['horizontal_error_abs'] = df_merged['horizontal_error'].abs()
df_merged['vertical_error_abs'] = df_merged['vertical_error'].abs()

# Remove outliers: 2.5% cut on both sides (keep middle 95%)
horizontal_lower = df_merged['horizontal_error_abs'].quantile(0.025)
horizontal_upper = df_merged['horizontal_error_abs'].quantile(0.975)
vertical_lower = df_merged['vertical_error_abs'].quantile(0.025)
vertical_upper = df_merged['vertical_error_abs'].quantile(0.975)

print("=== OUTLIER REMOVAL ===")
print(f"Horizontal error thresholds: {horizontal_lower:.3f}° to {horizontal_upper:.3f}°")
print(f"Vertical error thresholds: {vertical_lower:.3f}° to {vertical_upper:.3f}°")

# Create filtered dataset (remove outliers from both horizontal and vertical)
df_filtered = df_merged[
    (df_merged['horizontal_error_abs'] >= horizontal_lower) & 
    (df_merged['horizontal_error_abs'] <= horizontal_upper) &
    (df_merged['vertical_error_abs'] >= vertical_lower) & 
    (df_merged['vertical_error_abs'] <= vertical_upper)
].copy()

print(f"Original dataset: {len(df_merged)} frames")
print(f"After outlier removal: {len(df_filtered)} frames ({len(df_filtered)/len(df_merged)*100:.1f}%)")
print(f"Outliers removed: {len(df_merged) - len(df_filtered)} frames")

# Display summary statistics of the errors
print("\n" + "="*50)
print("ORIGINAL DATA (with outliers)")
print("="*50)
print("Horizontal Movement (Yaw) Error:")
print(df_merged['horizontal_error'].describe())
print("\nVertical Movement (Pitch) Error:")
print(df_merged['vertical_error'].describe())

print("\n" + "="*50)
print("FILTERED DATA (outliers removed)")
print("="*50)
print("Horizontal Movement (Yaw) Error:")
print(df_filtered['horizontal_error'].describe())
print("\nVertical Movement (Pitch) Error:")
print(df_filtered['vertical_error'].describe())

# Use filtered data for further analysis and plotting
df_merged = df_filtered

# Save the merged dataframe to a CSV file for inspection if needed.
df_merged.to_csv('ground_truth_vs_analysis_comparison.csv', index=False)

# Plotting the data for visual comparison
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
# Use symmetric log scale for both axes and format ticks as plain integers
plt.xscale('symlog', linthresh=1)
plt.yscale('symlog', linthresh=1)

int_tick_formatter = FuncFormatter(lambda x, pos: f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:g}")
ax = plt.gca()
ax.xaxis.set_major_formatter(int_tick_formatter)
ax.yaxis.set_major_formatter(int_tick_formatter)
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
plt.xscale('symlog', linthresh=1)
plt.yscale('symlog', linthresh=1)

int_tick_formatter = FuncFormatter(lambda x, pos: f"{int(round(x))}" if abs(x - round(x)) < 1e-6 else f"{x:g}")
ax = plt.gca()
ax.xaxis.set_major_formatter(int_tick_formatter)
ax.yaxis.set_major_formatter(int_tick_formatter)
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