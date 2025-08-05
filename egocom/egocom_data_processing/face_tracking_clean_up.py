import os
import pandas as pd
import numpy as np
import sys

def calculate_center(row):
    return np.array([(row['x1'] + row['x2']) / 2, (row['y1'] + row['y2']) / 2])

def clean_csv_file(file_path, output_dir):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return

    if df.empty or not all(col in df.columns for col in ['frame_number', 'person_id', 'x1', 'y1', 'x2', 'y2']):
        print(f"File '{file_path}' is missing required columns or is empty, skipping.")
        return

    df = df.sort_values('frame_number').reset_index(drop=True)
    
    indices_to_drop = []
    
    last_positions = {}

    for frame_number, frame_group in df.groupby('frame_number'):
        person_counts = frame_group['person_id'].value_counts()
        duplicate_persons = person_counts[person_counts > 1].index
        
        unique_persons = person_counts[person_counts == 1].index
        unique_rows = frame_group[frame_group['person_id'].isin(unique_persons)]
        for _, row in unique_rows.iterrows():
            last_positions[row['person_id']] = calculate_center(row)

        for person_id in duplicate_persons:
            person_group = frame_group[frame_group['person_id'] == person_id]
            
            if person_id in last_positions:
                last_pos = last_positions[person_id]
                
                distances = person_group.apply(
                    lambda row: np.linalg.norm(calculate_center(row) - last_pos),
                    axis=1
                )
                
                closest_index = distances.idxmin()
                indices_to_drop.extend(person_group.index.drop(closest_index))
                
                last_positions[person_id] = calculate_center(df.loc[closest_index])

            else:
                indices_to_drop.extend(person_group.index[1:])
                last_positions[person_id] = calculate_center(person_group.iloc[0])

    if indices_to_drop:
        dropped_df = df.loc[indices_to_drop]
        for _, row in dropped_df.iterrows():
            print(f"Deletion log: {os.path.basename(file_path)} file - Frame: {row['frame_number']}, Person: {row['person_id']}, Row: {row.to_dict()}")
    
    cleaned_df = df.drop(indices_to_drop).reset_index(drop=True)
    
    output_filename = os.path.basename(file_path)
    if not output_dir:
        output_dir = os.path.dirname(file_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    cleaned_output_path = os.path.join(output_dir, f"{os.path.splitext(output_filename)[0]}_cleaned.csv")
    cleaned_df.to_csv(cleaned_output_path, index=False)
    print(f"\nCleaned file saved to: {cleaned_output_path}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_duplicates.py <input_path_or_file> [output_directory]")
        sys.exit(1)
        
    input_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "cleaned_csvs"

    if os.path.isfile(input_path):
        clean_csv_file(input_path, output_dir)
    elif os.path.isdir(input_path):
        for root, _, files in os.walk(input_path):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing: {file_path}...")
                    clean_csv_file(file_path, output_dir)
    else:
        print(f"Error: '{input_path}' is not a valid file or directory.")
        sys.exit(1)

if __name__ == "__main__":
    main()