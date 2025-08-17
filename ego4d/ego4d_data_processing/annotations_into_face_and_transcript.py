import json
import csv
import os
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def compute_frame_range(start_time_s: float, end_time_s: float = None, fps: float = 30.0):
    """Compute an inclusive list of frame indices covering [start, end).
    
    If `end_time_s` is None or not finite, returns a single-frame list at the
    start frame.
    """
    if not np.isfinite(start_time_s):
        return []

    start_frame: int = math.floor(start_time_s * fps)

    if end_time_s is None or not np.isfinite(end_time_s):
        return [start_frame]

    # Inclusive end frame covers up to but not including end_time_s
    end_frame_inclusive: int = max(start_frame, math.ceil(end_time_s * fps) - 1)

    # Guard against pathological spans
    if end_frame_inclusive < start_frame:
        end_frame_inclusive = start_frame

    # Generate consecutive frames [start_frame, ..., end_frame_inclusive]
    return list(range(start_frame, end_frame_inclusive + 1))

def load_json_annotations(train_path, val_path):
    """Load and combine train and validation JSON annotations"""
    videos_data = {}
    
    # Load train data
    with open(train_path, 'r') as f:
        train_data = json.load(f)
        for video in train_data['videos']:
            video_uid = video['video_uid']
            videos_data[video_uid] = video
    
    # Load validation data
    with open(val_path, 'r') as f:
        val_data = json.load(f)
        for video in val_data['videos']:
            video_uid = video['video_uid']
            videos_data[video_uid] = video
    
    return videos_data

def process_face_detection(video_uid, video_data, output_dir):
    """Process face detection data from tracking paths"""
    face_detection_path = os.path.join(output_dir, 'face_detection', f'{video_uid}.csv')
    os.makedirs(os.path.dirname(face_detection_path), exist_ok=True)
    
    face_data = []
    
    # Process each clip in the video
    for clip in video_data.get('clips', []):
        # Process persons (excluding camera wearer)
        for person in clip.get('persons', []):
            person_id = person.get('person_id', '')
            
            # Skip if person_id is '0' or empty
            if not person_id or person_id == '0':
                continue
                
            # Process tracking paths for this person
            for track_data in person.get('tracking_paths', []):
                for track_item in track_data.get('track', []):
                    frame_number = track_item.get('video_frame', track_item.get('frame', 0))
                    x1 = track_item.get('x', 0)
                    y1 = track_item.get('y', 0)
                    width = track_item.get('width', 0)
                    height = track_item.get('height', 0)
                    
                    # x2 and y2 are width and height as per requirement
                    face_data.append({
                        'frame_number': frame_number,
                        'person_id': person_id,
                        'x1': x1,
                        'x2': width,  # x2 is width
                        'y1': y1,
                        'y2': height,  # y2 is height
                        'speaker_id': person_id  # speaker_id is same as person_id
                    })
    
    # Sort by frame number for better readability
    face_data.sort(key=lambda x: (x['frame_number'], x['person_id']))
    
    # Write to CSV
    if face_data:
        with open(face_detection_path, 'w', newline='') as f:
            fieldnames = ['frame_number', 'person_id', 'x1', 'x2', 'y1', 'y2', 'speaker_id']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(face_data)
        print(f"Created face detection file: {face_detection_path}")
    
    return len(face_data)

def process_transcriptions(video_uid, video_data, output_dir):
    """Process transcription data"""
    transcript_path = os.path.join(output_dir, 'transcript', 'ground_truth_transcriptions_with_frames.csv')
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    
    transcription_data = []
    
    # Process each clip in the video
    for clip in video_data.get('clips', []):
        # Process transcriptions
        for trans in clip.get('transcriptions', []):
            person_id = trans.get('person_id', '')
            
            # Skip if person_id is '0' or empty
            if not person_id or person_id == '0':
                continue
            
            # Split transcription into words
            transcription_text = trans.get('transcription', '')
            start_time = trans.get('video_start_time', trans.get('start_time_sec', 0))
            end_time = trans.get('video_end_time', trans.get('end_time_sec', 0))
            
            # Simple word splitting - distribute time evenly across words
            words = transcription_text.split()
            if words:
                time_per_word = (end_time - start_time) / len(words)
                
                for i, word in enumerate(words):
                    word_start = start_time + (i * time_per_word)
                    word_end = start_time + ((i + 1) * time_per_word)
                    
                    # Compute frame range for this word
                    frame_list = compute_frame_range(word_start, word_end, fps=30.0)
                    
                    transcription_data.append({
                        'conversation_id': video_uid,  # Using video_uid as conversation_id
                        'endTime': round(word_end, 2),
                        'speaker_id': person_id,
                        'startTime': round(word_start, 2),
                        'word': word,
                        'frame': json.dumps(frame_list, separators=(",", ":"))  # Store as compact JSON array
                    })
    
    # Sort transcription data by frame (use the first frame in the frame list for sorting)
    if transcription_data:
        def get_first_frame(item):
            frame_list = json.loads(item['frame'])
            return frame_list[0] if frame_list else 0
        
        transcription_data.sort(key=get_first_frame)
    
    # Write or append to CSV
    file_exists = os.path.exists(transcript_path)
    
    if transcription_data:
        with open(transcript_path, 'a' if file_exists else 'w', newline='') as f:
            fieldnames = ['conversation_id', 'endTime', 'speaker_id', 'startTime', 'word', 'frame']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerows(transcription_data)
        
        print(f"{'Appended to' if file_exists else 'Created'} transcript file: {transcript_path}")
    
    return len(transcription_data)

def main():
    # Paths
    train_json_path = '/mas/robots/prg-ego4d/raw/v2/annotations/av_train.json'
    val_json_path = '/mas/robots/prg-ego4d/raw/v2/annotations/av_val.json'
    output_base_dir = '/mas/robots/prg-ego4d'
    
    # Target video UIDs
    target_videos = {
        '30294c41-c90d-438a-af19-c1c74787d06b',  # train
        '566ad4e5-1ce4-4679-9d19-ef63072c848c',  # val
        '9c5b7322-d1cc-4b56-ae9d-85831f28fac1',  # val
        '9ca2dc18-2c57-44cb-8c91-4b8b5c7ca223',  # val
        'a223fcb2-8ffa-4826-bd0c-91027cf1c11e',  # val
        'b3937482-c973-4263-957d-1d5366329dad'   # train
    }
    
    print("Loading JSON annotations...")
    videos_data = load_json_annotations(train_json_path, val_json_path)
    print(f"Loaded data for {len(videos_data)} videos")
    
    # Process each target video
    for video_uid in target_videos:
        if video_uid in videos_data:
            print(f"\nProcessing video: {video_uid}")
            
            # Process face detection data
            face_count = process_face_detection(video_uid, videos_data[video_uid], output_base_dir)
            print(f"  - Processed {face_count} face detection entries")
            
            # Process transcription data
            trans_count = process_transcriptions(video_uid, videos_data[video_uid], output_base_dir)
            print(f"  - Processed {trans_count} transcription word entries")
        else:
            print(f"\nWarning: Video {video_uid} not found in the loaded data")
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()