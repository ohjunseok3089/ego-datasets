#!/usr/bin/env python3

import cv2
import pandas as pd
import numpy as np

def analyze_coordinate_systems(video_path, gt_file, output_file):
    """Analyze coordinate system differences"""
    
    # Load files
    gt_df = pd.read_csv(gt_file, header=None, 
                       names=['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    
    # Convert coordinate columns to numeric
    for col in ['frame_number', 'x1', 'y1', 'x2', 'y2']:
        gt_df[col] = pd.to_numeric(gt_df[col], errors='coerce')
    
    output_df = pd.read_csv(output_file)
    
    # Get video properties
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"Video resolution: {width} x {height}")
    print(f"Ground truth coordinate ranges:")
    print(f"  X: {gt_df['x1'].min():.1f} - {gt_df['x2'].max():.1f}")
    print(f"  Y: {gt_df['y1'].min():.1f} - {gt_df['y2'].max():.1f}")
    
    print(f"Detected coordinate ranges:")
    print(f"  X: {output_df['x1'].min()} - {output_df['x2'].max()}")
    print(f"  Y: {output_df['y1'].min()} - {output_df['y2'].max()}")
    
    # Find frames with both ground truth and detected faces
    common_frames = set(gt_df['frame_number']) & set(output_df['frame_number'])
    print(f"\nFrames with both GT and detected faces: {len(common_frames)}")
    
    # Analyze a few specific frames
    test_frames = [497, 498, 499, 500]
    for frame_num in test_frames:
        if frame_num in common_frames:
            print(f"\n=== FRAME {frame_num} ===")
            
            gt_frame = gt_df[gt_df['frame_number'] == frame_num]
            det_frame = output_df[output_df['frame_number'] == frame_num]
            
            print("Ground Truth:")
            for _, gt_row in gt_frame.iterrows():
                w_gt = gt_row['x2'] - gt_row['x1']
                h_gt = gt_row['y2'] - gt_row['y1']
                print(f"  Person {gt_row['person_id']}: [{gt_row['x1']:.1f}, {gt_row['y1']:.1f}, {gt_row['x2']:.1f}, {gt_row['y2']:.1f}] size: {w_gt:.1f}x{h_gt:.1f}")
            
            print("Detected:")
            for _, det_row in det_frame.iterrows():
                w_det = det_row['x2'] - det_row['x1']
                h_det = det_row['y2'] - det_row['y1']
                print(f"  {det_row['person_id']}: [{det_row['x1']}, {det_row['y1']}, {det_row['x2']}, {det_row['y2']}] size: {w_det}x{h_det}")
            
            # Try to find potential transformations
            if len(gt_frame) == 1 and len(det_frame) == 1:
                gt_row = gt_frame.iloc[0]
                det_row = det_frame.iloc[0]
                
                print("\nPotential coordinate transformations:")
                
                # Scale factors
                scale_x = width / gt_df['x2'].max() if gt_df['x2'].max() > 0 else 1
                scale_y = height / gt_df['y2'].max() if gt_df['y2'].max() > 0 else 1
                print(f"  If GT coordinates are normalized: scale_x={scale_x:.2f}, scale_y={scale_y:.2f}")
                
                # Scaled GT coordinates
                gt_scaled_x1 = gt_row['x1'] * scale_x
                gt_scaled_y1 = gt_row['y1'] * scale_y
                gt_scaled_x2 = gt_row['x2'] * scale_x
                gt_scaled_y2 = gt_row['y2'] * scale_y
                print(f"  Scaled GT: [{gt_scaled_x1:.1f}, {gt_scaled_y1:.1f}, {gt_scaled_x2:.1f}, {gt_scaled_y2:.1f}]")
                print(f"  Detected:  [{det_row['x1']}, {det_row['y1']}, {det_row['x2']}, {det_row['y2']}]")
                
                # Check if coordinates might be swapped
                print(f"  GT (x,y) swapped: [{gt_row['y1']:.1f}, {gt_row['x1']:.1f}, {gt_row['y2']:.1f}, {gt_row['x2']:.1f}]")

if __name__ == "__main__":
    video_path = "/mas/robots/prg-ego4d/raw/v2/full_scale.gaze/9c5b7322-d1cc-4b56-ae9d-85831f28fac1.mp4"
    gt_file = "/mas/robots/prg-ego4d/face_detection/9c5b7322-d1cc-4b56-ae9d-85831f28fac1.csv"
    output_file = "/mas/robots/prg-ego4d/processed_face_recognition_videos/9c5b7322-d1cc-4b56-ae9d-85831f28fac1_global_gallery.csv"
    
    analyze_coordinate_systems(video_path, gt_file, output_file)
