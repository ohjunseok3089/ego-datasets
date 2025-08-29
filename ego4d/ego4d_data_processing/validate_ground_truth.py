#!/usr/bin/env python3
"""
Validation script to check ground truth injection before running face recognition.
This script verifies that:
1. Ground truth CSV files exist and are readable
2. Ground truth contains valid person IDs and bounding boxes
3. Ground truth frames exist in the video
4. Face detection works on ground truth frames
"""

import cv2
import numpy as np
import pandas as pd
import os
import argparse
import sys
from typing import Optional


def load_and_validate_ground_truth(ground_truth_path: str) -> Optional[pd.DataFrame]:
    """Load and validate ground truth CSV file"""
    print(f"🔍 Checking ground truth file: {ground_truth_path}")
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth file not found: {ground_truth_path}")
        return None
    
    try:
        df = pd.read_csv(ground_truth_path, header=0)
        print(f"✅ Successfully loaded CSV with {len(df)} rows")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return None
    
    # Check required columns
    required_cols = ['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return None
    
    print(f"✅ All required columns present: {required_cols}")
    
    # Validate person_id column
    df = df[pd.to_numeric(df['person_id'], errors='coerce').notnull()]
    df['person_id'] = df['person_id'].astype(int)
    
    if len(df) == 0:
        print("❌ No valid person_id entries found")
        return None
    
    unique_person_ids = sorted(df['person_id'].unique())
    print(f"✅ Found {len(unique_person_ids)} unique person IDs: {unique_person_ids}")
    
    # Check frame numbers
    unique_frames = sorted(df['frame_number'].unique())
    print(f"✅ Ground truth covers {len(unique_frames)} frames")
    print(f"   Frame range: {min(unique_frames)} to {max(unique_frames)}")
    
    # Validate bounding boxes
    invalid_boxes = 0
    for idx, row in df.iterrows():
        x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            invalid_boxes += 1
    
    if invalid_boxes > 0:
        print(f"⚠️  Found {invalid_boxes} invalid bounding boxes")
    else:
        print("✅ All bounding boxes are valid")
    
    return df


def validate_video_access(video_path: str, ground_truth_df: pd.DataFrame) -> bool:
    """Validate that we can access video frames referenced in ground truth"""
    print(f"\n🔍 Validating video access: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video file: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video opened successfully")
    print(f"   Total frames: {total_frames}")
    print(f"   FPS: {fps}")
    print(f"   Resolution: {width}x{height}")
    
    # Check if ground truth frames are within video bounds
    max_gt_frame = ground_truth_df['frame_number'].max()
    if max_gt_frame >= total_frames:
        print(f"❌ Ground truth references frame {max_gt_frame}, but video only has {total_frames} frames")
        cap.release()
        return False
    
    print(f"✅ Ground truth frames are within video bounds")
    
    # Test accessing a few ground truth frames
    gt_frames = sorted(ground_truth_df['frame_number'].unique())
    test_frames = gt_frames[:min(5, len(gt_frames))]  # Test first 5 GT frames
    
    accessible_frames = 0
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret and frame is not None:
            accessible_frames += 1
        else:
            print(f"⚠️  Cannot read frame {frame_num}")
    
    cap.release()
    
    if accessible_frames == len(test_frames):
        print(f"✅ Successfully accessed all tested GT frames ({accessible_frames}/{len(test_frames)})")
        return True
    else:
        print(f"❌ Could only access {accessible_frames}/{len(test_frames)} tested GT frames")
        return False


def test_face_detection_sample(video_path: str, ground_truth_df: pd.DataFrame) -> bool:
    """Test face detection on a sample of ground truth frames"""
    print(f"\n🔍 Testing face detection on sample GT frames...")
    
    try:
        # Try to import InsightFace
        import insightface
        from insightface.app import FaceAnalysis
        print("✅ InsightFace import successful")
    except ImportError as e:
        print(f"❌ Cannot import InsightFace: {e}")
        return False
    
    try:
        # Try to initialize face analysis (CPU mode for testing)
        app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode
        print("✅ InsightFace model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize InsightFace: {e}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot reopen video file")
        return False
    
    # Test a few GT frames
    gt_frames = sorted(ground_truth_df['frame_number'].unique())
    test_frames = gt_frames[:min(3, len(gt_frames))]  # Test first 3 GT frames
    
    successful_detections = 0
    total_expected_faces = 0
    total_detected_faces = 0
    
    for frame_num in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Get ground truth for this frame
        frame_gt = ground_truth_df[ground_truth_df['frame_number'] == frame_num]
        expected_faces = len(frame_gt)
        total_expected_faces += expected_faces
        
        # Run face detection
        try:
            faces = app.get(frame)
            detected_faces = len(faces)
            total_detected_faces += detected_faces
            
            if detected_faces > 0:
                successful_detections += 1
                print(f"   Frame {frame_num}: detected {detected_faces} faces (expected {expected_faces})")
            else:
                print(f"   Frame {frame_num}: detected 0 faces (expected {expected_faces}) ⚠️")
                
        except Exception as e:
            print(f"   Frame {frame_num}: face detection failed: {e}")
    
    cap.release()
    
    print(f"\n📊 Face Detection Test Results:")
    print(f"   Frames tested: {len(test_frames)}")
    print(f"   Frames with detections: {successful_detections}")
    print(f"   Total expected faces: {total_expected_faces}")
    print(f"   Total detected faces: {total_detected_faces}")
    
    if successful_detections > 0:
        print("✅ Face detection is working")
        return True
    else:
        print("❌ Face detection failed on all test frames")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate ground truth before face recognition")
    parser.add_argument('--video_path', type=str, required=True, help="Path to video file")
    parser.add_argument('--ground_truth_dir', type=str, required=True, help="Directory containing ground truth CSV files")
    parser.add_argument('--skip_face_detection_test', action='store_true', help="Skip face detection test (faster)")
    
    args = parser.parse_args()
    
    # Determine ground truth file path
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]
    ground_truth_path = os.path.join(args.ground_truth_dir, f"{video_basename}.csv")
    
    print("="*60)
    print("🧪 GROUND TRUTH VALIDATION")
    print("="*60)
    print(f"Video: {args.video_path}")
    print(f"Ground Truth: {ground_truth_path}")
    print()
    
    # Step 1: Validate ground truth file
    ground_truth_df = load_and_validate_ground_truth(ground_truth_path)
    if ground_truth_df is None:
        print("\n❌ VALIDATION FAILED: Ground truth file issues")
        sys.exit(1)
    
    # Step 2: Validate video access
    if not validate_video_access(args.video_path, ground_truth_df):
        print("\n❌ VALIDATION FAILED: Video access issues")
        sys.exit(1)
    
    # Step 3: Test face detection (optional)
    if not args.skip_face_detection_test:
        if not test_face_detection_sample(args.video_path, ground_truth_df):
            print("\n⚠️  WARNING: Face detection test failed")
            print("   This might cause issues during processing")
            print("   Consider checking InsightFace installation or video quality")
        else:
            print("\n✅ Face detection test passed")
    else:
        print("\n⏭️  Skipped face detection test")
    
    print("\n" + "="*60)
    print("✅ VALIDATION PASSED")
    print("="*60)
    print("Ground truth is properly injected and ready for processing!")
    print(f"✅ {len(ground_truth_df)} ground truth annotations loaded")
    print(f"✅ {len(ground_truth_df['person_id'].unique())} unique person IDs: {sorted(ground_truth_df['person_id'].unique())}")
    print(f"✅ {len(ground_truth_df['frame_number'].unique())} ground truth frames available")
    print("\nYou can now safely run the face recognition script.")


if __name__ == "__main__":
    main()
