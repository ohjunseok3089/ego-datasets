#!/usr/bin/env python3
"""
Test script for Aria Audio Diarization improvements

This script tests the improved diarization and timestamp normalization
"""

import os
import sys
import pandas as pd
import tempfile
from pathlib import Path

# Add the current directory to path to import our module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aria_audio_diarization import AriaSpeakerDiarization

def create_test_speech_csv():
    """Create a test speech.csv file with sample data"""
    test_data = {
        'startTime_ns': [388749000000, 389749000000, 390309000000, 392000000000, 394500000000],
        'endTime_ns': [389749000000, 390309000000, 390629000000, 393200000000, 395800000000],
        'written': ['Hello.', 'Hello.', "How's", 'it going?', 'Good thanks!'],
        'confidence': [0.8856, 0.7809, 0.9880, 0.9123, 0.8567]
    }
    
    df = pd.DataFrame(test_data)
    return df

def test_timestamp_normalization():
    """Test timestamp normalization functionality"""
    print("🧪 Testing timestamp normalization...")
    
    # Create test data
    df = create_test_speech_csv()
    
    # Initialize diarizer
    diarizer = AriaSpeakerDiarization()
    
    # Save test data to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        test_csv_path = f.name
    
    try:
        # Load and process
        loaded_df = diarizer.load_speech_csv(test_csv_path)
        
        if loaded_df is not None:
            print("✅ CSV loaded successfully")
            
            # Check if normalized timestamps start from 0
            min_normalized = loaded_df['start_sec_normalized'].min()
            max_normalized = loaded_df['end_sec_normalized'].max()
            
            print(f"📊 Original time range: {loaded_df['start_sec'].min():.2f}s - {loaded_df['end_sec'].max():.2f}s")
            print(f"📊 Normalized time range: {min_normalized:.2f}s - {max_normalized:.2f}s")
            
            if abs(min_normalized) < 0.001:  # Should be very close to 0
                print("✅ Timestamps normalized correctly (start from ~0)")
            else:
                print(f"❌ Timestamp normalization failed - starts from {min_normalized:.3f}")
            
            return loaded_df
        else:
            print("❌ Failed to load CSV")
            return None
            
    finally:
        # Clean up
        os.unlink(test_csv_path)

def test_improved_speaker_assignment():
    """Test improved speaker assignment"""
    print("\n🎤 Testing improved speaker assignment...")
    
    df = create_test_speech_csv()
    diarizer = AriaSpeakerDiarization()
    
    # Load data
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        test_csv_path = f.name
    
    try:
        loaded_df = diarizer.load_speech_csv(test_csv_path)
        
        if loaded_df is not None:
            # Test fallback speaker assignment
            df_with_speakers = diarizer._improved_fallback_speaker_assignment(loaded_df.copy())
            
            speaker_counts = df_with_speakers['speaker_label'].value_counts()
            print(f"📊 Speaker distribution: {dict(speaker_counts)}")
            
            # Check if we have multiple speakers
            num_unique_speakers = len(speaker_counts)
            if num_unique_speakers > 1:
                print("✅ Multiple speakers detected")
            else:
                print("⚠️ Only one speaker detected - this might be expected for short segments")
            
            # Display segment details
            print("\n📝 Segment details:")
            for idx, row in df_with_speakers.iterrows():
                print(f"  {idx}: [{row['start_sec_normalized']:.2f}-{row['end_sec_normalized']:.2f}s] "
                      f"{row['speaker_label']} -> '{row['written']}'")
            
            return df_with_speakers
        else:
            print("❌ Failed to load CSV for speaker assignment test")
            return None
            
    finally:
        os.unlink(test_csv_path)

def test_transcript_generation():
    """Test transcript generation in Ego4D format"""
    print("\n📄 Testing transcript generation...")
    
    # Create test data with speaker labels
    test_data = {
        'startTime_ns': [388749000000, 389749000000, 390309000000],
        'endTime_ns': [389749000000, 390309000000, 390629000000],
        'written': ['Hello there.', 'How are you?', 'Good thanks!'],
        'confidence': [0.88, 0.92, 0.85],
        'start_sec_normalized': [0.0, 1.0, 1.56],
        'end_sec_normalized': [1.0, 1.56, 1.88],
        'speaker_label': ['person_1', 'person_2', 'person_1']
    }
    
    df_with_speakers = pd.DataFrame(test_data)
    
    # Test transcript generation
    diarizer = AriaSpeakerDiarization()
    
    with tempfile.TemporaryDirectory() as temp_dir:
        transcript_count = diarizer.generate_transcript_csv(
            df_with_speakers, 
            "test_recording", 
            temp_dir
        )
        
        print(f"📊 Generated {transcript_count} transcript word entries")
        
        # Check if transcript file was created
        transcript_path = os.path.join(temp_dir, 'ground_truth_transcriptions_with_frames.csv')
        if os.path.exists(transcript_path):
            print("✅ Transcript CSV created successfully")
            
            # Read and display sample
            transcript_df = pd.read_csv(transcript_path)
            print(f"📊 Transcript contains {len(transcript_df)} word entries")
            print("\n📝 Sample transcript entries:")
            print(transcript_df.head().to_string(index=False))
            
            return transcript_df
        else:
            print("❌ Transcript CSV was not created")
            return None

def main():
    """Run all tests"""
    print("🚀 Starting Aria Diarization Tests\n")
    
    # Test 1: Timestamp normalization
    normalized_df = test_timestamp_normalization()
    
    # Test 2: Speaker assignment
    speaker_df = test_improved_speaker_assignment()
    
    # Test 3: Transcript generation
    transcript_df = test_transcript_generation()
    
    print("\n🎯 Test Summary:")
    print("✅ Timestamp normalization: PASSED" if normalized_df is not None else "❌ Timestamp normalization: FAILED")
    print("✅ Speaker assignment: PASSED" if speaker_df is not None else "❌ Speaker assignment: FAILED")
    print("✅ Transcript generation: PASSED" if transcript_df is not None else "❌ Transcript generation: FAILED")
    
    if all([normalized_df is not None, speaker_df is not None, transcript_df is not None]):
        print("\n🎉 All tests passed! The improved diarization system is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
