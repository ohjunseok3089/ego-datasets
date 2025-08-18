#!/usr/bin/env python3
"""
Post-processing script for Aria face tracking results.
Since Aria recordings typically contain only one person, this script
normalizes all person_x labels to person_1.
"""

import os
import csv
import glob
import argparse
from pathlib import Path


def process_csv_file(csv_path, output_dir=None):
    """
    Process a single CSV file and normalize all person_x to person_1
    
    Args:
        csv_path (str): Path to the input CSV file
        output_dir (str): Output directory. If None, overwrites the original file
    """
    print(f"Processing: {os.path.basename(csv_path)}")
    
    # Read the CSV file
    rows = []
    with open(csv_path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header
        rows.append(header)
        
        for row in reader:
            if len(row) >= 2:  # Ensure we have at least frame_number and person_id
                # Normalize person_id - change any person_x to person_1
                if row[1].startswith('person_') and row[1] != 'unknown':
                    row[1] = 'person_1'
            rows.append(row)
    
    # Determine output path
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, os.path.basename(csv_path))
    else:
        output_path = csv_path
    
    # Write the processed CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    
    print(f"  -> Saved to: {output_path}")


def process_directory(input_dir, output_dir=None, pattern="*_global_gallery.csv"):
    """
    Process all CSV files in a directory
    
    Args:
        input_dir (str): Input directory containing CSV files
        output_dir (str): Output directory. If None, overwrites original files
        pattern (str): File pattern to match
    """
    search_pattern = os.path.join(input_dir, pattern)
    csv_files = glob.glob(search_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching pattern: {search_pattern}")
        return
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for csv_file in sorted(csv_files):
        print(f"  - {os.path.basename(csv_file)}")
    
    print("\nProcessing files...")
    for csv_file in sorted(csv_files):
        process_csv_file(csv_file, output_dir)
    
    print(f"\nCompleted processing {len(csv_files)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process Aria face tracking CSV files to normalize person IDs to person_1"
    )
    parser.add_argument(
        '--input_dir', 
        type=str, 
        required=True,
        help="Directory containing face tracking CSV files"
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        default=None,
        help="Output directory. If not specified, overwrites original files"
    )
    parser.add_argument(
        '--pattern', 
        type=str, 
        default="*_global_gallery.csv",
        help="File pattern to match (default: *_global_gallery.csv)"
    )
    parser.add_argument(
        '--single_file', 
        type=str, 
        default=None,
        help="Process a single CSV file instead of a directory"
    )
    
    args = parser.parse_args()
    
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"Error: File '{args.single_file}' does not exist!")
            return
        
        print(f"Processing single file: {args.single_file}")
        process_csv_file(args.single_file, args.output_dir)
    else:
        if not os.path.exists(args.input_dir):
            print(f"Error: Directory '{args.input_dir}' does not exist!")
            return
        
        print(f"Processing directory: {args.input_dir}")
        process_directory(args.input_dir, args.output_dir, args.pattern)


if __name__ == "__main__":
    main()
