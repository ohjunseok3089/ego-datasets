#!/usr/bin/env python3

import pandas as pd
import numpy as np
import sys

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def debug_frame_matching(frame_num, gt_file, output_file):
    """Debug matching for a specific frame"""
    
    # Load ground truth
    gt_df = pd.read_csv(gt_file, header=None, 
                       names=['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    
    # Load output
    output_df = pd.read_csv(output_file)
    
    print(f"\n=== FRAME {frame_num} DEBUG ===")
    
    # Get ground truth for this frame
    gt_frame = gt_df[gt_df['frame_number'] == frame_num]
    output_frame = output_df[output_df['frame_number'] == frame_num]
    
    print(f"Ground Truth faces: {len(gt_frame)}")
    for _, gt_row in gt_frame.iterrows():
        print(f"  GT Person {gt_row['person_id']}: [{gt_row['x1']:.1f}, {gt_row['y1']:.1f}, {gt_row['x2']:.1f}, {gt_row['y2']:.1f}]")
    
    print(f"Detected faces: {len(output_frame)}")
    for _, out_row in output_frame.iterrows():
        print(f"  Detected {out_row['person_id']}: [{out_row['x1']}, {out_row['y1']}, {out_row['x2']}, {out_row['y2']}]")
    
    # Calculate IoUs
    print("\nIoU Matrix:")
    if len(gt_frame) > 0 and len(output_frame) > 0:
        for _, gt_row in gt_frame.iterrows():
            gt_bbox = [gt_row['x1'], gt_row['y1'], gt_row['x2'], gt_row['y2']]
            for _, out_row in output_frame.iterrows():
                out_bbox = [out_row['x1'], out_row['y1'], out_row['x2'], out_row['y2']]
                iou = calculate_iou(gt_bbox, out_bbox)
                print(f"  GT Person {gt_row['person_id']} vs Detected {out_row['person_id']}: IoU = {iou:.3f}")
    else:
        print("  No faces to compare")

def analyze_coordinate_differences(gt_file, output_file, frames_to_check=[0, 1, 2, 498, 499, 500]):
    """Analyze coordinate system differences"""
    
    # Load files
    gt_df = pd.read_csv(gt_file, header=None, 
                       names=['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
    output_df = pd.read_csv(output_file)
    
    print("=== COORDINATE SYSTEM ANALYSIS ===")
    print(f"Ground truth unique person IDs: {sorted(gt_df['person_id'].unique())}")
    print(f"Output unique person IDs: {sorted(output_df['person_id'].unique())}")
    
    # Check each frame
    for frame_num in frames_to_check:
        debug_frame_matching(frame_num, gt_file, output_file)

if __name__ == "__main__":
    gt_file = "/mas/robots/prg-ego4d/face_detection/9c5b7322-d1cc-4b56-ae9d-85831f28fac1.csv"
    output_file = "/mas/robots/prg-ego4d/processed_face_recognition_videos/9c5b7322-d1cc-4b56-ae9d-85831f28fac1_global_gallery.csv"
    
    analyze_coordinate_differences(gt_file, output_file)
