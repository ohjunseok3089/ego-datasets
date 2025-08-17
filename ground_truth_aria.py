import json
import pandas as pd
import numpy as np

# ground truth given by ARIA
with open('./aria/closed_loop_trajectory_20fps.json', 'r') as f:
    data_ground_truth = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_ground_truth = pd.json_normalize(data_ground_truth['frames'])
df_ground_truth = df_ground_truth[['frame_index', 'head_movement.horizontal_yaw.degrees', 'head_movement.vertical_pitch.degrees']]
df_ground_truth.rename(columns={
    'head_movement.horizontal_yaw.degrees': 'ground_truth_horizontal_yaw',
    'head_movement.vertical_pitch.degrees': 'ground_truth_vertical_pitch'
}, inplace=True)


# the co-tracker analysis
with open('../recording_analysis.json', 'r') as f:
    data_recording_analysis = json.load(f)

# Flatten the nested JSON structure into a pandas DataFrame
df_recording_analysis = pd.json_normalize(data_recording_analysis['frames'])
df_recording_analysis = df_recording_analysis[['frame_index', 'head_movement.horizontal.degrees', 'head_movement.vertical.degrees']]
df_recording_analysis.rename(columns={
    'head_movement.horizontal.degrees': 'analysis_horizontal',
    'head_movement.vertical.degrees': 'analysis_vertical'
}, inplace=True)

# temporal alignment
print("=== DATA ALIGNMENT ANALYSIS ===")
print(f"Ground Truth: {len(df_ground_truth)} frames (range: {df_ground_truth['frame_index'].min()}-{df_ground_truth['frame_index'].max()})")
print(f"Analysis: {len(df_recording_analysis)} frames (range: {df_recording_analysis['frame_index'].min()}-{df_recording_analysis['frame_index'].max()})")

# frame offset
gt_frame_count = len(df_ground_truth)
analysis_frame_count = len(df_recording_analysis)
frames_to_remove = analysis_frame_count - gt_frame_count

print(f"Frame count difference: {frames_to_remove} (Analysis has {frames_to_remove} more frames than Ground Truth)")
print(f"Removing first {frames_to_remove} frames from analysis data to align temporal windows...")

# remove first N frames by frame_index
df_recording_analysis_shifted = df_recording_analysis[df_recording_analysis['frame_index'] >= frames_to_remove].copy()

# shift the frame indices down by N to align with ground truth
df_recording_analysis_shifted['frame_index'] = df_recording_analysis_shifted['frame_index'] - frames_to_remove

print(f"Analysis (after removing first {frames_to_remove} frames): {len(df_recording_analysis_shifted)} frames")
print(f"Analysis (shifted frame range): {df_recording_analysis_shifted['frame_index'].min()}-{df_recording_analysis_shifted['frame_index'].max()}")

# remove any NaN values 
df_recording_analysis_clean = df_recording_analysis_shifted.dropna(subset=['analysis_horizontal', 'analysis_vertical'])
print(f"Analysis (after removing NaN): {len(df_recording_analysis_clean)} frames")

# merge the aligned datasets
df_merged = pd.merge(df_ground_truth, df_recording_analysis_clean, on='frame_index', how='inner')
print(f"Final merged dataset: {len(df_merged)} frames")

# calculate the difference (error) between the ground truth and recording analysis
df_merged['horizontal_error'] = df_merged['ground_truth_horizontal_yaw'] - df_merged['analysis_horizontal']
df_merged['vertical_error'] = df_merged['ground_truth_vertical_pitch'] - df_merged['analysis_vertical']

# calculate absolute errors for outlier removal
df_merged['horizontal_error_abs'] = df_merged['horizontal_error'].abs()
df_merged['vertical_error_abs'] = df_merged['vertical_error'].abs()

# remove outliers: 2.5% cut on both sides (keep middle 99%)
horizontal_lower = df_merged['horizontal_error_abs'].quantile(0.005)
horizontal_upper = df_merged['horizontal_error_abs'].quantile(0.995)
vertical_lower = df_merged['vertical_error_abs'].quantile(0.005)
vertical_upper = df_merged['vertical_error_abs'].quantile(0.995)

print("=== OUTLIER REMOVAL ===")
print(f"Horizontal error thresholds: {horizontal_lower:.3f}° to {horizontal_upper:.3f}°")
print(f"Vertical error thresholds: {vertical_lower:.3f}° to {vertical_upper:.3f}°")

# create filtered dataset (remove outliers from both horizontal and vertical)
df_filtered = df_merged[
    (df_merged['horizontal_error_abs'] >= horizontal_lower) & 
    (df_merged['horizontal_error_abs'] <= horizontal_upper) &
    (df_merged['vertical_error_abs'] >= vertical_lower) & 
    (df_merged['vertical_error_abs'] <= vertical_upper)
].copy()

print(f"Original dataset: {len(df_merged)} frames")
print(f"After outlier removal: {len(df_filtered)} frames ({len(df_filtered)/len(df_merged)*100:.1f}%)")
print(f"Outliers removed: {len(df_merged) - len(df_filtered)} frames")

# display summary statistics of the errors
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

# use filtered data for further analysis and plotting
df_merged = df_filtered

# ----------------------------------------------------
# GROUND TRUTH ANGULAR MOVEMENT ANALYSIS
# ----------------------------------------------------
print("\n" + "="*50)
print("GROUND TRUTH ANGULAR MOVEMENT STATISTICS")
print("="*50)

# Calculate absolute angular movements for ground truth
gt_horizontal_abs = df_merged['ground_truth_horizontal_yaw'].abs()
gt_vertical_abs = df_merged['ground_truth_vertical_pitch'].abs()

# Ground truth movement statistics
gt_mae_h = float(gt_horizontal_abs.mean()) if len(gt_horizontal_abs) else float('nan')
gt_mae_v = float(gt_vertical_abs.mean()) if len(gt_vertical_abs) else float('nan')
gt_rmse_h = float(np.sqrt((df_merged['ground_truth_horizontal_yaw'] ** 2).mean())) if len(df_merged) else float('nan')
gt_rmse_v = float(np.sqrt((df_merged['ground_truth_vertical_pitch'] ** 2).mean())) if len(df_merged) else float('nan')
gt_std_h = float(df_merged['ground_truth_horizontal_yaw'].std()) if len(df_merged) else float('nan')
gt_std_v = float(df_merged['ground_truth_vertical_pitch'].std()) if len(df_merged) else float('nan')

print("Ground Truth Horizontal Movement (Yaw):")
print(f"  MAE:  {gt_mae_h:.3f}°")
print(f"  RMSE: {gt_rmse_h:.3f}°")
print(f"  STD:  {gt_std_h:.3f}°")
print(f"  Range: {df_merged['ground_truth_horizontal_yaw'].min():.3f}° to {df_merged['ground_truth_horizontal_yaw'].max():.3f}°")

print("\nGround Truth Vertical Movement (Pitch):")
print(f"  MAE:  {gt_mae_v:.3f}°")
print(f"  RMSE: {gt_rmse_v:.3f}°")
print(f"  STD:  {gt_std_v:.3f}°")
print(f"  Range: {df_merged['ground_truth_vertical_pitch'].min():.3f}° to {df_merged['ground_truth_vertical_pitch'].max():.3f}°")

# Calculate frame-to-frame angular velocity (movement changes)
if len(df_merged) > 1:
    df_sorted = df_merged.sort_values('frame_index')
    gt_h_velocity = df_sorted['ground_truth_horizontal_yaw'].diff().abs()
    gt_v_velocity = df_sorted['ground_truth_vertical_pitch'].diff().abs()
    
    # Remove NaN from first frame
    gt_h_velocity = gt_h_velocity.dropna()
    gt_v_velocity = gt_v_velocity.dropna()
    
    gt_vel_mae_h = float(gt_h_velocity.mean()) if len(gt_h_velocity) else float('nan')
    gt_vel_mae_v = float(gt_v_velocity.mean()) if len(gt_v_velocity) else float('nan')
    gt_vel_max_h = float(gt_h_velocity.max()) if len(gt_h_velocity) else float('nan')
    gt_vel_max_v = float(gt_v_velocity.max()) if len(gt_v_velocity) else float('nan')
    
    print("\nGround Truth Angular Velocity (frame-to-frame change):")
    print(f"Horizontal - Mean: {gt_vel_mae_h:.3f}°/frame | Max: {gt_vel_max_h:.3f}°/frame")
    print(f"Vertical   - Mean: {gt_vel_mae_v:.3f}°/frame | Max: {gt_vel_max_v:.3f}°/frame")

# Combined 2D movement magnitude for ground truth
gt_combined_2d = np.sqrt(df_merged['ground_truth_horizontal_yaw']**2 + df_merged['ground_truth_vertical_pitch']**2)
gt_combined_mae = float(gt_combined_2d.mean()) if len(gt_combined_2d) else float('nan')
print(f"\nGround Truth Combined 2D Movement MAE: {gt_combined_mae:.3f}°")

# ----------------------------------------------------
# ANALYSIS ANGULAR MOVEMENT STATISTICS
# ----------------------------------------------------
print("\n" + "="*50)
print("ANALYSIS ANGULAR MOVEMENT STATISTICS")
print("="*50)

# Calculate absolute angular movements for analysis
analysis_horizontal_abs = df_merged['analysis_horizontal'].abs()
analysis_vertical_abs = df_merged['analysis_vertical'].abs()

# Analysis movement statistics
analysis_mae_h = float(analysis_horizontal_abs.mean()) if len(analysis_horizontal_abs) else float('nan')
analysis_mae_v = float(analysis_vertical_abs.mean()) if len(analysis_vertical_abs) else float('nan')
analysis_rmse_h = float(np.sqrt((df_merged['analysis_horizontal'] ** 2).mean())) if len(df_merged) else float('nan')
analysis_rmse_v = float(np.sqrt((df_merged['analysis_vertical'] ** 2).mean())) if len(df_merged) else float('nan')
analysis_std_h = float(df_merged['analysis_horizontal'].std()) if len(df_merged) else float('nan')
analysis_std_v = float(df_merged['analysis_vertical'].std()) if len(df_merged) else float('nan')

print("Analysis Horizontal Movement:")
print(f"  MAE:  {analysis_mae_h:.3f}°")
print(f"  RMSE: {analysis_rmse_h:.3f}°")
print(f"  STD:  {analysis_std_h:.3f}°")
print(f"  Range: {df_merged['analysis_horizontal'].min():.3f}° to {df_merged['analysis_horizontal'].max():.3f}°")

print("\nAnalysis Vertical Movement:")
print(f"  MAE:  {analysis_mae_v:.3f}°")
print(f"  RMSE: {analysis_rmse_v:.3f}°")
print(f"  STD:  {analysis_std_v:.3f}°")
print(f"  Range: {df_merged['analysis_vertical'].min():.3f}° to {df_merged['analysis_vertical'].max():.3f}°")

# Calculate frame-to-frame angular velocity for analysis (movement changes)
if len(df_merged) > 1:
    df_sorted = df_merged.sort_values('frame_index')
    analysis_h_velocity = df_sorted['analysis_horizontal'].diff().abs()
    analysis_v_velocity = df_sorted['analysis_vertical'].diff().abs()
    
    # Remove NaN from first frame
    analysis_h_velocity = analysis_h_velocity.dropna()
    analysis_v_velocity = analysis_v_velocity.dropna()
    
    analysis_vel_mae_h = float(analysis_h_velocity.mean()) if len(analysis_h_velocity) else float('nan')
    analysis_vel_mae_v = float(analysis_v_velocity.mean()) if len(analysis_v_velocity) else float('nan')
    analysis_vel_max_h = float(analysis_h_velocity.max()) if len(analysis_h_velocity) else float('nan')
    analysis_vel_max_v = float(analysis_v_velocity.max()) if len(analysis_v_velocity) else float('nan')
    
    print("\nAnalysis Angular Velocity (frame-to-frame change):")
    print(f"Horizontal - Mean: {analysis_vel_mae_h:.3f}°/frame | Max: {analysis_vel_max_h:.3f}°/frame")
    print(f"Vertical   - Mean: {analysis_vel_mae_v:.3f}°/frame | Max: {analysis_vel_max_v:.3f}°/frame")

# Combined 2D movement magnitude for analysis
analysis_combined_2d = np.sqrt(df_merged['analysis_horizontal']**2 + df_merged['analysis_vertical']**2)
analysis_combined_mae = float(analysis_combined_2d.mean()) if len(analysis_combined_2d) else float('nan')
print(f"\nAnalysis Combined 2D Movement MAE: {analysis_combined_mae:.3f}°")

# ----------------------------------------------------
# COMPARISON BETWEEN GROUND TRUTH AND ANALYSIS
# ----------------------------------------------------
print("\n" + "="*50)
print("COMPARISON: GROUND TRUTH vs ANALYSIS")
print("="*50)

print("Horizontal Movement Comparison:")
print(f"  Ground Truth MAE: {gt_mae_h:.3f}° | Analysis MAE: {analysis_mae_h:.3f}° | Difference: {abs(gt_mae_h - analysis_mae_h):.3f}°")
print(f"  Ground Truth RMSE: {gt_rmse_h:.3f}° | Analysis RMSE: {analysis_rmse_h:.3f}° | Difference: {abs(gt_rmse_h - analysis_rmse_h):.3f}°")
print(f"  Ground Truth STD: {gt_std_h:.3f}° | Analysis STD: {analysis_std_h:.3f}° | Difference: {abs(gt_std_h - analysis_std_h):.3f}°")

print("\nVertical Movement Comparison:")
print(f"  Ground Truth MAE: {gt_mae_v:.3f}° | Analysis MAE: {analysis_mae_v:.3f}° | Difference: {abs(gt_mae_v - analysis_mae_v):.3f}°")
print(f"  Ground Truth RMSE: {gt_rmse_v:.3f}° | Analysis RMSE: {analysis_rmse_v:.3f}° | Difference: {abs(gt_rmse_v - analysis_rmse_v):.3f}°")
print(f"  Ground Truth STD: {gt_std_v:.3f}° | Analysis STD: {analysis_std_v:.3f}° | Difference: {abs(gt_std_v - analysis_std_v):.3f}°")

print("\nCombined 2D Movement Comparison:")
print(f"  Ground Truth MAE: {gt_combined_mae:.3f}° | Analysis MAE: {analysis_combined_mae:.3f}° | Difference: {abs(gt_combined_mae - analysis_combined_mae):.3f}°")

if len(df_merged) > 1:
    print("\nAngular Velocity Comparison:")
    print(f"  Ground Truth Horizontal - Mean: {gt_vel_mae_h:.3f}°/frame | Analysis - Mean: {analysis_vel_mae_h:.3f}°/frame | Difference: {abs(gt_vel_mae_h - analysis_vel_mae_h):.3f}°/frame")
    print(f"  Ground Truth Vertical   - Mean: {gt_vel_mae_v:.3f}°/frame | Analysis - Mean: {analysis_vel_mae_v:.3f}°/frame | Difference: {abs(gt_vel_mae_v - analysis_vel_mae_v):.3f}°/frame")

# Save the merged dataframe to a CSV file for inspection if needed.
df_merged.to_csv('ground_truth_vs_analysis_comparison.csv', index=False)

# ----------------------------------------------------
# GROUND TRUTH MOVEMENT VISUALIZATIONS
# ----------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Plot ground truth movement patterns over time
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df_merged['frame_index'], df_merged['ground_truth_horizontal_yaw'], 
         color='blue', alpha=0.8, linewidth=1)
plt.ylabel('Horizontal Yaw (degrees)')
plt.title('Ground Truth Angular Movement Over Time')
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
plt.plot(df_merged['frame_index'], df_merged['ground_truth_vertical_pitch'], 
         color='red', alpha=0.8, linewidth=1)
plt.xlabel('Frame Index')
plt.ylabel('Vertical Pitch (degrees)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ground_truth_movement_over_time.png', dpi=150)
plt.close()

# Distribution of ground truth movements
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(df_merged['ground_truth_horizontal_yaw'], bins=50, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Horizontal Yaw (degrees)')
plt.ylabel('Frequency')
plt.title('Ground Truth Horizontal Movement Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(df_merged['ground_truth_vertical_pitch'], bins=50, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('Vertical Pitch (degrees)')
plt.ylabel('Frequency')
plt.title('Ground Truth Vertical Movement Distribution')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ground_truth_movement_distribution.png', dpi=150)
plt.close()

# Angular velocity visualization (if calculated)
if len(df_merged) > 1:
    df_sorted = df_merged.sort_values('frame_index')
    gt_h_velocity = df_sorted['ground_truth_horizontal_yaw'].diff().abs()
    gt_v_velocity = df_sorted['ground_truth_vertical_pitch'].diff().abs()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(df_sorted['frame_index'][1:], gt_h_velocity.dropna(), alpha=0.8, color='blue')
    plt.xlabel('Frame Index')
    plt.ylabel('Angular Velocity (degrees/frame)')
    plt.title('Ground Truth Horizontal Angular Velocity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(df_sorted['frame_index'][1:], gt_v_velocity.dropna(), alpha=0.8, color='red')
    plt.xlabel('Frame Index')
    plt.ylabel('Angular Velocity (degrees/frame)')
    plt.title('Ground Truth Vertical Angular Velocity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ground_truth_angular_velocity.png', dpi=150)
    plt.close()

# Plotting the data for visual comparison

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

# ----------------------
# Error rate visualizations
# ----------------------
# Compute metrics on filtered data
h_abs = df_merged['horizontal_error'].abs()
v_abs = df_merged['vertical_error'].abs()
mae_h = float(h_abs.mean()) if len(h_abs) else float('nan')
mae_v = float(v_abs.mean()) if len(v_abs) else float('nan')
rmse_h = float(np.sqrt((df_merged['horizontal_error'] ** 2).mean())) if len(df_merged) else float('nan')
rmse_v = float(np.sqrt((df_merged['vertical_error'] ** 2).mean())) if len(df_merged) else float('nan')
print("\n" + "="*50)
print("ERROR METRICS (filtered data)")
print("="*50)
print(f"Horizontal MAE: {mae_h:.3f}° | RMSE: {rmse_h:.3f}°")
print(f"Vertical   MAE: {mae_v:.3f}° | RMSE: {rmse_v:.3f}°")

non_zero_h = df_merged['ground_truth_horizontal_yaw'] != 0
non_zero_v = df_merged['ground_truth_vertical_pitch'] != 0

mpe_h = np.mean(df_merged['horizontal_error'][non_zero_h] / df_merged['ground_truth_horizontal_yaw'][non_zero_h]) * 100
mpe_v = np.mean(df_merged['vertical_error'][non_zero_v] / df_merged['ground_truth_vertical_pitch'][non_zero_v]) * 100

print(f"Horizontal MPE: {mpe_h:.3f}%")
print(f"Vertical   MPE: {mpe_v:.3f}%")
# Percentage within thresholds
thresholds = [1, 2, 5, 10]
rates_h = [(h_abs <= t).mean() * 100.0 if len(h_abs) else 0.0 for t in thresholds]
rates_v = [(v_abs <= t).mean() * 100.0 if len(v_abs) else 0.0 for t in thresholds]
print("\nWithin-threshold accuracy (% of frames):")
for t, rh, rv in zip(thresholds, rates_h, rates_v):
    print(f"  ≤{t:>2}°  | Horizontal: {rh:6.2f}%  | Vertical: {rv:6.2f}%")
# Bar chart of error rates
import matplotlib.pyplot as plt
indices = np.arange(len(thresholds))
bar_width = 0.35
plt.figure(figsize=(8, 5))
plt.bar(indices - bar_width/2, rates_h, width=bar_width, label='Horizontal')
plt.bar(indices + bar_width/2, rates_v, width=bar_width, label='Vertical')
plt.xticks(indices, [f"≤{t}°" for t in thresholds])
plt.ylim(0, 100)
plt.ylabel('Percent within threshold (%)')
plt.title('Error Rate by Threshold (Filtered)')
plt.grid(axis='y', linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('error_rate_thresholds.png')
plt.close()
# Absolute error over time
df_plot = df_merged.sort_values('frame_index')
plt.figure(figsize=(12, 5))
plt.plot(df_plot['frame_index'], df_plot['horizontal_error'].abs(), label='|Horizontal error|', alpha=0.8)
plt.plot(df_plot['frame_index'], df_plot['vertical_error'].abs(), label='|Vertical error|', alpha=0.8)
plt.xlabel('Frame Index')
plt.ylabel('Absolute Error (degrees)')
plt.title('Absolute Error over Time (Filtered)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.savefig('absolute_error_over_time.png')
plt.close()

df_merged['combined_error_2d'] = np.sqrt(
    df_merged['horizontal_error']**2 + df_merged['vertical_error']**2
)
print("\n" + "="*50)
print("COMBINED 2D ERROR (Euclidean Distance)")
print("="*50)
print(df_merged['combined_error_2d'].describe())
mae_combined = df_merged['combined_error_2d'].mean()
print(f"\nCombined 2D Error MAE: {mae_combined:.3f}°")