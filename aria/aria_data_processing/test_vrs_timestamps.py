#!/usr/bin/env python3
"""
Test script to read VRS file timestamps and understand the structure
"""

import pyvrs as vrs
import sys
import os

def analyze_vrs_file(vrs_path):
    """Analyze VRS file to understand timestamps and structure"""
    if not os.path.exists(vrs_path):
        print(f"VRS file not found: {vrs_path}")
        return
    
    try:
        # Open VRS file
        reader = vrs.SyncVRSReader(vrs_path)
        
        print(f"=== VRS File Analysis: {vrs_path} ===")
        print(f"File size: {os.path.getsize(vrs_path):,} bytes")
        
        # Get record count
        record_count = reader.n_records
        print(f"Total records: {record_count}")
        
        # Get available stream IDs
        stream_ids = reader.stream_ids
        print(f"Available streams: {stream_ids}")
        
        # Get time range
        min_timestamp = reader.min_timestamp
        max_timestamp = reader.max_timestamp
        
        print(f"\n=== Overall Recording Info ===")
        print(f"Recording start timestamp: {min_timestamp} ns")
        print(f"Recording end timestamp: {max_timestamp} ns")
        print(f"Total duration: {(max_timestamp - min_timestamp) / 1e9:.2f} seconds")
        
        # Convert to datetime
        import datetime
        start_dt = datetime.datetime.fromtimestamp(min_timestamp / 1e9)
        end_dt = datetime.datetime.fromtimestamp(max_timestamp / 1e9)
        print(f"Recording start time: {start_dt}")
        print(f"Recording end time: {end_dt}")
        
        # Analyze each stream
        for stream_id in stream_ids:
            stream_info = reader.get_stream_info(stream_id)
            print(f"\nStream {stream_id}:")
            print(f"  - Info: {stream_info}")
            
            # Get stream tags
            stream_tags = reader.stream_tags.get(stream_id, {})
            print(f"  - Tags: {stream_tags}")
            
            # Count records for this stream
            stream_records = 0
            stream_timestamps = []
            
            # Read through records to count this stream
            reader.read_next_record()  # Reset to beginning
            while True:
                try:
                    record = reader.read_next_record()
                    if record.stream_id == stream_id:
                        stream_records += 1
                        stream_timestamps.append(record.timestamp)
                except:
                    break
            
            if stream_timestamps:
                print(f"  - Record count: {stream_records}")
                print(f"  - First timestamp: {min(stream_timestamps)} ns")
                print(f"  - Last timestamp: {max(stream_timestamps)} ns")
                print(f"  - Duration: {(max(stream_timestamps) - min(stream_timestamps)) / 1e9:.2f} seconds")
        
        return min_timestamp
        
    except Exception as e:
        print(f"Error reading VRS file: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_speech_csv_conversion(speech_csv_path, vrs_start_timestamp):
    """Test converting speech.csv timestamps to relative seconds"""
    if not os.path.exists(speech_csv_path):
        print(f"Speech CSV not found: {speech_csv_path}")
        return
    
    print(f"\n=== Speech CSV Analysis ===")
    
    import pandas as pd
    df = pd.read_csv(speech_csv_path)
    
    print(f"Speech segments: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    if len(df) > 0:
        print(f"First segment start: {df.iloc[0]['startTime_ns']} ns")
        print(f"Last segment end: {df.iloc[-1]['endTime_ns']} ns")
        
        # Convert to relative seconds
        if vrs_start_timestamp:
            df['start_relative_sec'] = (df['startTime_ns'] - vrs_start_timestamp) / 1e9
            df['end_relative_sec'] = (df['endTime_ns'] - vrs_start_timestamp) / 1e9
            
            print(f"\n=== Converted to Relative Seconds ===")
            print(f"First segment: {df.iloc[0]['start_relative_sec']:.3f}s - {df.iloc[0]['end_relative_sec']:.3f}s")
            print(f"Last segment: {df.iloc[-1]['start_relative_sec']:.3f}s - {df.iloc[-1]['end_relative_sec']:.3f}s")
            print(f"Total speech duration: {df.iloc[-1]['end_relative_sec']:.2f}s")
            
            # Show first few examples
            print(f"\n=== First 5 segments ===")
            for i in range(min(5, len(df))):
                row = df.iloc[i]
                print(f"{i+1}: {row['start_relative_sec']:.3f}s-{row['end_relative_sec']:.3f}s: '{row['written']}'")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_vrs_timestamps.py <recording_dir>")
        sys.exit(1)
    
    recording_dir = sys.argv[1]
    vrs_path = os.path.join(recording_dir, "recording.vrs")
    speech_csv_path = os.path.join(recording_dir, "speech.csv")
    
    # Analyze VRS file
    vrs_start_timestamp = analyze_vrs_file(vrs_path)
    
    # Test speech CSV conversion
    test_speech_csv_conversion(speech_csv_path, vrs_start_timestamp)
