#!/usr/bin/env python3
"""
Test script for the fixed face recognition implementation.
This demonstrates the proper workflow and validates the approach.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_validation(video_path: str, ground_truth_dir: str, skip_face_test: bool = False) -> bool:
    """Run ground truth validation"""
    print("🧪 Step 1: Validating Ground Truth Injection...")
    print("="*60)
    
    cmd = [
        sys.executable, "validate_ground_truth.py",
        "--video_path", video_path,
        "--ground_truth_dir", ground_truth_dir
    ]
    
    if skip_face_test:
        cmd.append("--skip_face_detection_test")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Ground truth validation failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def run_fixed_face_recognition(video_path: str, ground_truth_dir: str, output_dir: str, 
                              recognition_threshold: float = 0.6) -> bool:
    """Run the fixed face recognition implementation"""
    print("\n🔧 Step 2: Running Fixed Face Recognition...")
    print("="*60)
    
    cmd = [
        sys.executable, "face_recognition_fixed.py",
        "--video_path", video_path,
        "--ground_truth_dir", ground_truth_dir,
        "--output_dir", output_dir,
        "--recognition_threshold", str(recognition_threshold),
        "--execution_provider", "auto"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Fixed face recognition failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False


def analyze_results(output_dir: str, video_path: str) -> None:
    """Analyze the results and compare with expectations"""
    print("\n📊 Step 3: Analyzing Results...")
    print("="*60)
    
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_csv = os.path.join(output_dir, f"{video_basename}_fixed_face_recognition.csv")
    output_video = os.path.join(output_dir, f"{video_basename}_fixed_face_recognition.mp4")
    
    if not os.path.exists(output_csv):
        print(f"❌ Output CSV not found: {output_csv}")
        return
    
    if not os.path.exists(output_video):
        print(f"❌ Output video not found: {output_video}")
        return
    
    # Read and analyze the CSV
    try:
        import pandas as pd
        df = pd.read_csv(output_csv)
        
        print(f"✅ Results CSV found: {output_csv}")
        print(f"   Total detections: {len(df)}")
        
        if 'person_id' in df.columns:
            unique_persons = sorted(df['person_id'].unique())
            print(f"   Unique person IDs: {unique_persons}")
            
            # Check for problematic high person IDs
            problematic_ids = [pid for pid in unique_persons if pid and str(pid).isdigit() and int(pid) > 10]
            if problematic_ids:
                print(f"⚠️  WARNING: Found potentially problematic person IDs: {problematic_ids}")
                print("   This suggests the issue may still persist.")
            else:
                print("✅ No problematic high person IDs found!")
        
        if 'source' in df.columns:
            source_counts = df['source'].value_counts()
            print(f"   Source breakdown: {dict(source_counts)}")
        
        if 'confidence' in df.columns:
            recognized_df = df[df['source'] == 'recognized'] if 'source' in df.columns else df
            if len(recognized_df) > 0:
                avg_confidence = recognized_df['confidence'].mean()
                min_confidence = recognized_df['confidence'].min()
                max_confidence = recognized_df['confidence'].max()
                print(f"   Recognition confidence - avg: {avg_confidence:.3f}, range: [{min_confidence:.3f}, {max_confidence:.3f}]")
        
        print(f"✅ Results video found: {output_video}")
        
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Test the fixed face recognition implementation")
    parser.add_argument('--video_path', type=str, required=True, help="Path to test video file")
    parser.add_argument('--ground_truth_dir', type=str, required=True, help="Directory containing ground truth CSV files")
    parser.add_argument('--output_dir', type=str, default="test_fixed_face_recognition_output", help="Output directory for results")
    parser.add_argument('--recognition_threshold', type=float, default=0.6, help="Recognition threshold (lower = more lenient)")
    parser.add_argument('--skip_validation', action='store_true', help="Skip ground truth validation step")
    parser.add_argument('--skip_face_test', action='store_true', help="Skip face detection test in validation")
    
    args = parser.parse_args()
    
    # Ensure we're in the right directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    print("🚀 TESTING FIXED FACE RECOGNITION IMPLEMENTATION")
    print("="*80)
    print(f"Video: {args.video_path}")
    print(f"Ground Truth Dir: {args.ground_truth_dir}")
    print(f"Output Dir: {args.output_dir}")
    print(f"Recognition Threshold: {args.recognition_threshold}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    success = True
    
    # Step 1: Validate ground truth injection
    if not args.skip_validation:
        if not run_validation(args.video_path, args.ground_truth_dir, args.skip_face_test):
            print("❌ Validation failed - aborting test")
            sys.exit(1)
    else:
        print("⏭️  Skipping validation step")
    
    # Step 2: Run fixed face recognition
    if not run_fixed_face_recognition(args.video_path, args.ground_truth_dir, args.output_dir, args.recognition_threshold):
        print("❌ Fixed face recognition failed")
        success = False
    
    # Step 3: Analyze results
    analyze_results(args.output_dir, args.video_path)
    
    print("\n" + "="*80)
    if success:
        print("✅ TEST COMPLETED SUCCESSFULLY!")
        print("✅ The fixed implementation should now properly constrain person IDs to ground truth values.")
        print("✅ No more spurious person IDs like 11, 12, etc. should appear.")
    else:
        print("❌ TEST FAILED!")
        print("❌ Please check the error messages above and fix the issues.")
    print("="*80)


if __name__ == "__main__":
    main()
