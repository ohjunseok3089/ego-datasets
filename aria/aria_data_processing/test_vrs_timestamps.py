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
        
        # Get time range from stream info
        # Find the actual data timestamps (not configuration/state records)
        actual_start_timestamp = None
        actual_end_timestamp = None
        
        for stream_id in stream_ids:
            stream_info = reader.get_stream_info(stream_id)
            if stream_info['data_records_count'] > 0:
                data_start = stream_info['first_data_record_timestamp']
                data_end = stream_info['last_data_record_timestamp']
                
                if actual_start_timestamp is None or data_start < actual_start_timestamp:
                    actual_start_timestamp = data_start
                if actual_end_timestamp is None or data_end > actual_end_timestamp:
                    actual_end_timestamp = data_end
        
        print(f"\n=== Overall Recording Info ===")
        print(f"Recording start timestamp: {actual_start_timestamp} seconds")
        print(f"Recording end timestamp: {actual_end_timestamp} seconds")
        print(f"Total duration: {actual_end_timestamp - actual_start_timestamp:.2f} seconds")
        
        # Convert to datetime (assuming Unix timestamp)
        import datetime
        start_dt = datetime.datetime.fromtimestamp(actual_start_timestamp)
        end_dt = datetime.datetime.fromtimestamp(actual_end_timestamp)
        print(f"Recording start time: {start_dt}")
        print(f"Recording end time: {end_dt}")
        
        # Convert to nanoseconds for speech.csv comparison
        vrs_start_ns = int(actual_start_timestamp * 1e9)
        print(f"VRS start in nanoseconds: {vrs_start_ns}")
        
        return vrs_start_ns
        
        # Analyze each stream
        for stream_id in stream_ids:
            stream_info = reader.get_stream_info(stream_id)
            print(f"\nStream {stream_id}:")
            print(f"  - Device: {stream_info.get('device_name', 'Unknown')}")
            print(f"  - Data records: {stream_info.get('data_records_count', 0)}")
            
            if stream_info.get('data_records_count', 0) > 0:
                print(f"  - Data start: {stream_info['first_data_record_timestamp']:.6f} seconds")
                print(f"  - Data end: {stream_info['last_data_record_timestamp']:.6f} seconds")
                print(f"  - Duration: {stream_info['last_data_record_timestamp'] - stream_info['first_data_record_timestamp']:.2f} seconds")
        
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
