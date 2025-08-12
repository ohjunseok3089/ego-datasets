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

    video_filename_w_ext = os.path.basename(video_path)
    video_filename_no_ext = os.path.splitext(video_filename_w_ext)[0]
    parts_dir = os.path.dirname(video_path)
    duration_dir = os.path.dirname(parts_dir)
    p720_dir = os.path.dirname(duration_dir)
    egocom_root_dir = os.path.dirname(p720_dir)

    paths = {}

    paths['angular_json_path'] = os.path.join(duration_dir, 'co-tracker', video_filename_w_ext, f"{video_filename_no_ext}_analysis.json")

    paths['transcriptions_csv_path'] = os.path.join(p720_dir, 'ground_truth_transcriptions.csv')

    gallery_base_match = re.match(r'(.*?)\(', video_filename_no_ext)
    if not gallery_base_match:
        raise ValueError(f"Could not determine gallery filename from video name '{video_filename_no_ext}' (e.g., missing '(...)').")
    gallery_filename = f"{gallery_base_match.group(1)}_global_gallery.csv"
    paths['head_tracking_csv_path'] = os.path.join(duration_dir, 'processed_face_recognition_videos', gallery_filename)

    # Optional body detection CSV
    paths['body_detection_csv_path'] = os.path.join(egocom_root_dir, 'body_detection', f"{video_filename_no_ext}_detections.csv")

    # Output joined ground truth path (under EGOCOM/joined_ground_truth)
    output_dir = os.path.join(egocom_root_dir, 'joined_ground_truth')
    os.makedirs(output_dir, exist_ok=True)
    paths['output_joined_json_path'] = os.path.join(output_dir, f"{video_filename_no_ext}.json")

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

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize likely variant column names to expected ones."""
    rename_map = {}
    if 'frame' in df.columns and 'frame_number' not in df.columns:
        rename_map['frame'] = 'frame_number'
    if 'person' in df.columns and 'person_id' not in df.columns:
        rename_map['person'] = 'person_id'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _group_face_detections_by_frame(head_tracking_df: pd.DataFrame):
    """Return dict: frame_number -> list of face detection dicts."""
    faces_by_frame = {}
    required = {'frame_number', 'x1', 'y1', 'x2', 'y2'}
    missing = required - set(head_tracking_df.columns)
    if missing:
        return faces_by_frame

    for frame_number, group in head_tracking_df.groupby('frame_number'):
        detections = []
        for _, row in group.iterrows():
            det = {
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
            }
            if 'person_id' in row:
                # pandas Series supports 'in' on index; guard with get
                det['person_id'] = int(row.get('person_id')) if pd.notna(row.get('person_id')) else None
            if 'speaker_id' in row:
                det['speaker_id'] = int(row.get('speaker_id')) if pd.notna(row.get('speaker_id')) else None
            detections.append(det)
        faces_by_frame[int(frame_number)] = detections
    return faces_by_frame


def _group_body_detections_by_frame(body_df: pd.DataFrame):
    """Return dict: frame_number -> list of body detection dicts."""
    bodies_by_frame = {}
    if body_df is None or body_df.empty:
        return bodies_by_frame
    body_df = _standardize_columns(body_df)

    # Expected columns from notes: frame, class_name, confidence, x1, y1, x2, y2, speaker_id
    for frame_number, group in body_df.groupby('frame_number'):
        detections = []
        for _, row in group.iterrows():
            det = {
                'x1': float(row['x1']),
                'y1': float(row['y1']),
                'x2': float(row['x2']),
                'y2': float(row['y2']),
            }
            if 'class_name' in row:
                det['class_name'] = str(row['class_name'])
            if 'confidence' in row:
                try:
                    det['confidence'] = float(row['confidence'])
                except Exception:
                    det['confidence'] = None
            if 'speaker_id' in row:
                det['speaker_id'] = int(row['speaker_id']) if pd.notna(row['speaker_id']) else None
            detections.append(det)
        bodies_by_frame[int(frame_number)] = detections
    return bodies_by_frame


def _find_speaker_bbox_in_frame(head_tracking_df: pd.DataFrame, frame_index: int, speaker_id: int):
    """Try to locate the bounding box for the speaker in a given frame.
    Returns dict with x1..y2 or None.
    """
    if head_tracking_df is None or head_tracking_df.empty or speaker_id is None:
        return None

    # Try by person_id in MultiIndex if available
    if isinstance(head_tracking_df.index, pd.MultiIndex) and set(head_tracking_df.index.names) >= {'frame_number', 'person_id'}:
        try:
            row = head_tracking_df.loc[(frame_index, speaker_id)]
            return row[['x1', 'y1', 'x2', 'y2']].to_dict()
        except KeyError:
            pass

    # Try by direct filtering on columns 'frame_number' + 'speaker_id'
    candidate_cols = set(head_tracking_df.columns)
    if {'frame_number', 'speaker_id'} <= candidate_cols:
        match = head_tracking_df[(head_tracking_df['frame_number'] == frame_index) & (head_tracking_df['speaker_id'] == speaker_id)]
        if not match.empty:
            row = match.iloc[0]
            return {k: float(row[k]) for k in ['x1', 'y1', 'x2', 'y2']}

    # Try by 'person_id' column (non-index)
    if {'frame_number', 'person_id'} <= candidate_cols:
        match = head_tracking_df[(head_tracking_df['frame_number'] == frame_index) & (head_tracking_df['person_id'] == speaker_id)]
        if not match.empty:
            row = match.iloc[0]
            return {k: float(row[k]) for k in ['x1', 'y1', 'x2', 'y2']}

    return None


def _predict_speaker_position(head_tracking_df: pd.DataFrame, angular_frames_by_index: dict, img_width: int, img_height: int, current_frame_index: int, speaker_id: int):
    """Predict speaker position when not detected in the current frame using past frame and head-movement remapping."""
    if head_tracking_df is None or head_tracking_df.empty or speaker_id is None:
        return None

    # Build a candidate series of past frames for this speaker
    df = head_tracking_df.copy()
    df = _standardize_columns(df)

    # Prefer explicit speaker_id column; else fall back to person_id
    id_col = 'speaker_id' if 'speaker_id' in df.columns else ('person_id' if 'person_id' in df.columns else None)
    if id_col is None or 'frame_number' not in df.columns:
        return None

    speaker_df = df[df[id_col] == speaker_id]
    past_frames = speaker_df[speaker_df['frame_number'] < current_frame_index]
    if past_frames.empty:
        return None

    last_row = past_frames.sort_values('frame_number').iloc[-1]
    last_known_frame_index = int(last_row['frame_number'])
    x1, y1, x2, y2 = [float(last_row[k]) for k in ['x1', 'y1', 'x2', 'y2']]
    current_pos = [(x1 + x2) / 2.0, (y1 + y2) / 2.0]

    for frame_idx in range(last_known_frame_index, current_frame_index):
        if frame_idx in angular_frames_by_index:
            frame_to_remap_from = angular_frames_by_index[frame_idx]
            if frame_to_remap_from.get('head_movement'):
                current_pos = remap_position_from_movement(
                    start_pos=current_pos,
                    head_movement=frame_to_remap_from['head_movement'],
                    image_width=img_width,
                    image_height=img_height,
                    video_fov_degrees=VIDEO_FOV_DEGREES,
                )
    return {'x': current_pos[0], 'y': current_pos[1]}


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
    # Try to fetch fps and frame count from video (fallback to constants)
    fps_read = cap.get(cv2.CAP_PROP_FPS) or FPS
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
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
        head_tracking_df = _standardize_columns(head_tracking_df)
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}", file=sys.stderr)
        sys.exit(1)

    # Load optional body detections
    body_df = None
    if os.path.isfile(paths_info['body_detection_csv_path']):
        try:
            body_df = pd.read_csv(paths_info['body_detection_csv_path'])
        except Exception as e:
            print(f"Warning: Failed to read body detections: {e}")
            body_df = None
    else:
        print(f"Warning: Body detection CSV not found at {paths_info['body_detection_csv_path']}. Continuing without it.")

    print("Preprocessing data for efficient lookup...")
    speaker_events_by_frame = preprocess_transcriptions(transcriptions_df, paths_info['conversation_id_substr'])
    # Prepare indices and per-frame groupings for efficient lookup
    if set(['frame_number', 'person_id']).issubset(head_tracking_df.columns):
        head_tracking_multiindex = head_tracking_df.set_index(['frame_number', 'person_id'])
    else:
        head_tracking_multiindex = head_tracking_df  # fallback without MultiIndex

    faces_by_frame = _group_face_detections_by_frame(head_tracking_df)
    bodies_by_frame = _group_body_detections_by_frame(body_df) if body_df is not None else {}
    angular_frames_by_index = {frame['frame_index']: frame for frame in angular_data['frames']}

    print("Augmenting frame data...")
    total_frames = len(angular_data['frames'])
    joined_frames = []
    for i, frame_obj in enumerate(angular_data['frames']):
        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"  Processing frame {i+1}/{total_frames}...")
            
        current_frame_index = frame_obj['frame_index']
        # Base per-frame output (clone of relevant angular fields)
        out_frame = {
            'frame_index': frame_obj.get('frame_index'),
            'timestamp': frame_obj.get('timestamp'),
            'social_category': frame_obj.get('social_category'),
            'red_circle': frame_obj.get('red_circle'),
            'head_movement': frame_obj.get('head_movement'),
            'next_movement': frame_obj.get('next_movement'),
        }

        # Defaults
        out_frame['speaker_id'] = None
        out_frame['speaker_words'] = []
        out_frame['speaker_location'] = {}
        out_frame['predicted_speaker_location'] = {}

        if current_frame_index in speaker_events_by_frame:
            speaker_events = speaker_events_by_frame[current_frame_index]
            if speaker_events:
                speaker_id = speaker_events[0]['id']
                out_frame['speaker_id'] = speaker_id
                out_frame['speaker_words'] = [evt['word'] for evt in speaker_events if evt['id'] == speaker_id]

                loc = _find_speaker_bbox_in_frame(head_tracking_multiindex if isinstance(head_tracking_multiindex, pd.DataFrame) else head_tracking_df, current_frame_index, speaker_id)
                if loc:
                    out_frame['speaker_location'] = loc
                else:
                    pred = _predict_speaker_position(head_tracking_df, angular_frames_by_index, img_width, img_height, current_frame_index, speaker_id)
                    if pred:
                        out_frame['predicted_speaker_location'] = pred

        # Attach detections for the frame
        out_frame['face_detection'] = faces_by_frame.get(current_frame_index, [])
        out_frame['body_detection'] = bodies_by_frame.get(current_frame_index, [])

        joined_frames.append(out_frame)

    # Build joined output structure
    metadata = angular_data.get('metadata', {}).copy()
    analysis_summary = angular_data.get('analysis_summary', {}).copy()

    output_joined = {
        'metadata': {
            'video_name': os.path.splitext(os.path.basename(video_path))[0],
            'video_path': video_path,
            'video_duration': int(round(frame_count / (fps_read or FPS))) if frame_count > 0 else None,
            'video_fps': fps_read or FPS,
            'roi': metadata.get('roi'),
            'frame_size_full': metadata.get('frame_size_full'),
        },
        'analysis_summary': {
            k: analysis_summary.get(k)
            for k in ['total_frames_processed', 'detected_frames']
            if k in analysis_summary
        },
        'frames': joined_frames,
    }

    output_path = paths_info['output_joined_json_path']
    print(f"\nProcessing complete. Saving joined ground truth to {output_path}")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_joined, f, indent=2)
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)

    print("Successfully finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join transcripts, face detections, body detections, and angular analysis into a single JSON per video.",
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