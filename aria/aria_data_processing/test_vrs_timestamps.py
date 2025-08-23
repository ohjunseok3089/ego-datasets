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
        reader = vrs.RecordFileReader(vrs_path)
        
        print(f"=== VRS File Analysis: {vrs_path} ===")
        print(f"File size: {os.path.getsize(vrs_path):,} bytes")
        
        # Get record count
        record_count = reader.get_record_count()
        print(f"Total records: {record_count}")
        
        # Get available stream IDs
        stream_ids = reader.get_stream_ids()
        print(f"Available streams: {stream_ids}")
        
        # Analyze each stream
        for stream_id in stream_ids:
            stream_info = reader.get_stream_info(stream_id)
            print(f"\nStream {stream_id}:")
            print(f"  - Type: {stream_info}")
            
            # Get records for this stream
            stream_records = reader.get_records_by_stream(stream_id)
            if stream_records:
                print(f"  - Record count: {len(stream_records)}")
                
                # Check first and last timestamps
                first_record = stream_records[0]
                last_record = stream_records[-1]
                
                first_timestamp = first_record.timestamp
                last_timestamp = last_record.timestamp
                
                print(f"  - First timestamp: {first_timestamp} ns")
                print(f"  - Last timestamp: {last_timestamp} ns")
                print(f"  - Duration: {(last_timestamp - first_timestamp) / 1e9:.2f} seconds")
                
                # Convert to human readable
                import datetime
                first_dt = datetime.datetime.fromtimestamp(first_timestamp / 1e9)
                last_dt = datetime.datetime.fromtimestamp(last_timestamp / 1e9)
                print(f"  - First time: {first_dt}")
                print(f"  - Last time: {last_dt}")
        
        # Find minimum timestamp across all streams (recording start time)
        all_timestamps = []
        for stream_id in stream_ids:
            stream_records = reader.get_records_by_stream(stream_id)
            if stream_records:
                all_timestamps.extend([r.timestamp for r in stream_records])
        
        if all_timestamps:
            min_timestamp = min(all_timestamps)
            max_timestamp = max(all_timestamps)
            
            print(f"\n=== Overall Recording Info ===")
            print(f"Recording start timestamp: {min_timestamp} ns")
            print(f"Recording end timestamp: {max_timestamp} ns")
            print(f"Total duration: {(max_timestamp - min_timestamp) / 1e9:.2f} seconds")
            
            # Convert to datetime
            start_dt = datetime.datetime.fromtimestamp(min_timestamp / 1e9)
            end_dt = datetime.datetime.fromtimestamp(max_timestamp / 1e9)
            print(f"Recording start time: {start_dt}")
            print(f"Recording end time: {end_dt}")
            
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
