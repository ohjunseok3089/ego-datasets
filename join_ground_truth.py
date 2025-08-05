import sys
import os
import argparse
import re
import json
import pandas as pd
import math
import shutil
import cv2

FPS = 30.0
VIDEO_FOV_DEGREES = 104.0

def remap_position_from_movement(start_pos, head_movement, image_width, image_height, video_fov_degrees):
    start_x, start_y = start_pos
    # Head movement is of the camera. To find where a static point appears to move,
    # we subtract the camera's movement angle.
    move_h_rad = head_movement['horizontal']['radians']
    move_v_rad = head_movement['vertical']['radians']

    center_x, center_y = image_width / 2, image_height / 2

    # Convert FOV to radians and calculate vertical FOV based on aspect ratio
    fov_h_rad = math.radians(video_fov_degrees)
    fov_v_rad = fov_h_rad * (image_height / image_width)

    # Convert start pixel to angle
    angle_x = math.atan(((start_x - center_x) / image_width) * 2 * math.tan(fov_h_rad / 2))
    angle_y = math.atan(((start_y - center_y) / image_height) * 2 * math.tan(fov_v_rad / 2))

    # Add camera movement to find the new angle of the point
    new_angle_x = angle_x - move_h_rad
    new_angle_y = angle_y - move_v_rad

    # Convert new angle back to pixel coordinates
    new_x = center_x + (image_width / (2 * math.tan(fov_h_rad / 2))) * math.tan(new_angle_x)
    new_y = center_y + (image_height / (2 * math.tan(fov_v_rad / 2))) * math.tan(new_angle_y)

    return [new_x, new_y]

def parse_video_path(video_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Input video path not found or is not a file: {video_path}")

    # --- Extract Path Components ---
    video_filename_w_ext = os.path.basename(video_path)
    video_filename_no_ext = os.path.splitext(video_filename_w_ext)[0]
    parts_dir = os.path.dirname(video_path)
    duration_dir = os.path.dirname(parts_dir)
    p720_dir = os.path.dirname(duration_dir)

    # --- Construct File Paths ---
    paths = {}

    paths['angular_json_path'] = os.path.join(duration_dir, 'co-tracker', video_filename_w_ext, f"{video_filename_no_ext}_analysis.json")

    paths['transcriptions_csv_path'] = os.path.join(p720_dir, 'ground_truth_transcriptions.csv')

    gallery_base_match = re.match(r'(.*?)\(', video_filename_no_ext)
    if not gallery_base_match:
        raise ValueError(f"Could not determine gallery filename from video name '{video_filename_no_ext}' (e.g., missing '(...)').")
    gallery_filename = f"{gallery_base_match.group(1)}_global_gallery.csv"
    paths['head_tracking_csv_path'] = os.path.join(duration_dir, 'processed_face_recognition_videos', gallery_filename)

    conv_id_match = re.search(r'(day_\d+__con_\d+)', video_filename_no_ext)
    if not conv_id_match:
        raise ValueError(f"Could not extract conversation ID (e.g., 'day_X__con_Y') from video name '{video_filename_no_ext}'.")
    paths['conversation_id_substr'] = conv_id_match.group(1)

    print("--- Derived Paths and Identifiers ---")
    for key, val in paths.items():
        print(f"{key:>25}: {val}")
    print("------------------------------------")
    return paths

def preprocess_transcriptions(transcriptions_df, conversation_id_substr):
    """
    Filters transcription data and maps spoken words to their starting frame number.
    """
    conv_df = transcriptions_df[transcriptions_df['conversation_id'].str.contains(conversation_id_substr, na=False)].copy()
    conv_df.dropna(subset=['startTime', 'endTime', 'speaker_id', 'word'], inplace=True)
    conv_df['speaker_id'] = conv_df['speaker_id'].astype(int)
    conv_df['start_frame'] = (conv_df['startTime'] * FPS).apply(math.floor)

    speaker_events_by_frame = {}
    for _, row in conv_df.iterrows():
        frame_idx = row['start_frame']
        event = {'id': row['speaker_id'], 'word': str(row['word'])}
        speaker_events_by_frame.setdefault(frame_idx, []).append(event)
        
    return speaker_events_by_frame

def process_video(video_path):
    """
    Main function to orchestrate the data loading, processing, and saving.
    """
    try:
        paths_info = parse_video_path(video_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print("\nReading video properties...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"FATAL: Could not open video file to read properties: {video_path}", file=sys.stderr)
        sys.exit(1)
    img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if img_width == 0 or img_height == 0:
        print(f"FATAL: Video file has invalid dimensions (0x0): {video_path}", file=sys.stderr)
        sys.exit(1)
    print(f"Detected video dimensions: {img_width}x{img_height}")

    print("\nLoading data sources...")
    try:
        with open(paths_info['angular_json_path'], 'r') as f:
            angular_data = json.load(f)
        transcriptions_df = pd.read_csv(paths_info['transcriptions_csv_path'])
        head_tracking_df = pd.read_csv(paths_info['head_tracking_csv_path'])
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}", file=sys.stderr)
        sys.exit(1)

    print("Preprocessing data for efficient lookup...")
    speaker_events_by_frame = preprocess_transcriptions(transcriptions_df, paths_info['conversation_id_substr'])
    head_tracking_df.set_index(['frame_number', 'person_id'], inplace=True)
    angular_frames_by_index = {frame['frame_index']: frame for frame in angular_data['frames']}

    print("Augmenting frame data...")
    total_frames = len(angular_data['frames'])
    for i, frame_obj in enumerate(angular_data['frames']):
        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"  Processing frame {i+1}/{total_frames}...")
            
        current_frame_index = frame_obj['frame_index']
        frame_obj.update({
            'speaker_id': None, 'speaker_words': [],
            'speaker_location': {}, 'predicted_speaker_location': {}
        })

        if current_frame_index in speaker_events_by_frame:
            speaker_events = speaker_events_by_frame[current_frame_index]
            if speaker_events:
                speaker_id = speaker_events[0]['id']
                person_id_to_lookup = speaker_id
                
                frame_obj['speaker_id'] = speaker_id
                frame_obj['speaker_words'] = [evt['word'] for evt in speaker_events if evt['id'] == speaker_id]

                try:
                    location_data = head_tracking_df.loc[(current_frame_index, person_id_to_lookup)]
                    frame_obj['speaker_location'] = location_data[['x1', 'y1', 'x2', 'y2']].to_dict()
                except KeyError:
                    speaker_df = head_tracking_df.xs(person_id_to_lookup, level='person_id', drop_level=False)
                    past_frames = speaker_df[speaker_df.index.get_level_values('frame_number') < current_frame_index]

                    if not past_frames.empty:
                        last_known_frame_row = past_frames.iloc[-1]
                        last_known_frame_index = last_known_frame_row.name[0]

                        x1, y1, x2, y2 = last_known_frame_row[['x1', 'y1', 'x2', 'y2']]
                        current_pos = [(x1 + x2) / 2, (y1 + y2) / 2]

                        for frame_idx in range(last_known_frame_index, current_frame_index):
                            if frame_idx in angular_frames_by_index:
                                frame_to_remap_from = angular_frames_by_index[frame_idx]
                                if frame_to_remap_from.get('head_movement'):
                                    current_pos = remap_position_from_movement(
                                        start_pos=current_pos,
                                        head_movement=frame_to_remap_from['head_movement'],
                                        image_width=img_width,
                                        image_height=img_height,
                                        video_fov_degrees=VIDEO_FOV_DEGREES
                                    )
                        frame_obj['predicted_speaker_location'] = {'x': current_pos[0], 'y': current_pos[1]}

    output_path = paths_info['angular_json_path']
    backup_path = output_path + '.bak'
    print(f"\nProcessing complete. Saving augmented data to {output_path}")
    try:
        shutil.copyfile(output_path, backup_path)
        print(f"Original file backed up to {backup_path}")
        with open(output_path, 'w') as f:
            json.dump(angular_data, f, indent=2)
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)

    print("Successfully finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Joins ground truth transcriptions and head tracking data into an angular movement analysis JSON file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "video_path", type=str,
        help="Full path to the input video file. \nExample: '/mas/robots/prg-egocom/EGOCOM/720p/20min/parts/vid_117__...MP4'"
    )
    
    try:
        import pandas as pd
    except ImportError:
        print("Error: 'pandas' is required. Please install with 'pip install pandas'", file=sys.stderr)
        sys.exit(1)
    try:
        import cv2
    except ImportError:
        print("Error: 'opencv-python' is required. Please install with 'pip install opencv-python'", file=sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    process_video(args.video_path)