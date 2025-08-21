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


def parse_conversation_id_from_name(video_basename: str) -> str:
    match = re.search(r'(day_\d+__con_\d+)', video_basename)
    if not match:
        return ''
    return match.group(1)


def parse_frame_range_from_name(video_basename: str):
    # e.g., vid_...part1(1980_2370_social_interaction)
    paren = re.search(r'\(([^)]*)\)', video_basename)
    if not paren:
        return None
    inner = paren.group(1)
    parts = inner.split('_')
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return int(parts[0]), int(parts[1])
    return None


def derive_paths_from_args(base_video: str, face_csv: str, body_csv: str, co_tracker_json: str, transcript_csv: str, output_json: str = None):
    """Build paths_info dict from explicit arguments; does not validate existence here."""
    video_basename = os.path.basename(base_video.rstrip('/'))
    conversation_id_substr = parse_conversation_id_from_name(video_basename)
    frame_range = parse_frame_range_from_name(video_basename)

    if output_json is None:
        # Derive EGOCOM root from co-tracker or face path
        candidate_root = None
        for p in [co_tracker_json, face_csv, body_csv, transcript_csv, base_video]:
            if p:
                try:
                    # Find '/EGOCOM/' in the path and take up to it
                    idx = p.find('/EGOCOM/')
                    if idx != -1:
                        candidate_root = p[: idx + len('/EGOCOM/') - 1]
                        break
                except Exception:
                    pass
        if candidate_root is None:
            candidate_root = os.path.dirname(os.path.dirname(co_tracker_json)) if co_tracker_json else os.getcwd()
        output_dir = os.path.join(candidate_root, 'joined_ground_truth')
        os.makedirs(output_dir, exist_ok=True)
        # Remove file extension if present
        clean_basename = os.path.splitext(video_basename)[0]
        output_json = os.path.join(output_dir, f"{clean_basename}.json")

    return {
        'angular_json_path': co_tracker_json,
        'transcriptions_csv_path': transcript_csv,
        'head_tracking_csv_path': face_csv,
        'body_detection_csv_path': body_csv,
        'output_joined_json_path': output_json,
        'conversation_id_substr': conversation_id_substr,
        'video_basename': video_basename,
        'frame_range': frame_range,
        'base_video_path': base_video,
    }

def preprocess_transcriptions(transcriptions_df, conversation_id_substr, frame_range=None, fps: float = FPS):
    """
    Build a mapping: frame_index -> list of {id, word} events from transcript.

    Supports two formats:
    - CSV with 'frame' column containing JSON arrays (e.g., "[2,3,4]")
    - CSV with times: compute start_frame from startTime
    """
    conv_df = transcriptions_df[transcriptions_df['conversation_id'].astype(str).str.contains(conversation_id_substr, na=False)].copy()
    speaker_events_by_frame = {}

    def add_event(frame_idx: int, speaker_id_val, word_val):
        if frame_idx is None or pd.isna(frame_idx):
            return
        try:
            frame_i = int(frame_idx)
        except Exception:
            return
        try:
            spk = int(speaker_id_val)
        except Exception:
            return
        event = {'id': spk, 'word': str(word_val)}
        speaker_events_by_frame.setdefault(frame_i, []).append(event)

    # Case A: has 'frame' column with JSON arrays or numeric
    if 'frame' in conv_df.columns:
        for _, row in conv_df.iterrows():
            word_val = row.get('word')
            speaker_id_val = row.get('speaker_id')
            frames_field = row.get('frame')
            frames_list = []
            if isinstance(frames_field, (list, tuple)):
                frames_list = list(frames_field)
            elif isinstance(frames_field, (int, float)) and not pd.isna(frames_field):
                frames_list = [int(frames_field)]
            elif isinstance(frames_field, str):
                # Expect a JSON array string; handle single int strings too
                try:
                    parsed = json.loads(frames_field)
                    if isinstance(parsed, list):
                        frames_list = parsed
                    elif isinstance(parsed, (int, float)):
                        frames_list = [int(parsed)]
                except Exception:
                    # Fallback: try to coerce to int
                    try:
                        frames_list = [int(frames_field)]
                    except Exception:
                        frames_list = []

            if frame_range is not None and frames_list:
                start_f, end_f = frame_range
                frames_list = [f for f in frames_list if isinstance(f, (int, float)) and start_f <= int(f) <= end_f]

            for fi in frames_list:
                add_event(fi, speaker_id_val, word_val)
        return speaker_events_by_frame

    # Case B: has explicit frame_number or start_frame columns
    conv_df = _standardize_columns(conv_df)
    if 'frame_number' in conv_df.columns:
        frame_col = 'frame_number'
    elif 'start_frame' in conv_df.columns:
        frame_col = 'start_frame'
    else:
        # Compute from times
        conv_df.dropna(subset=['startTime', 'speaker_id', 'word'], inplace=True)
        conv_df['start_frame'] = (pd.to_numeric(conv_df['startTime'], errors='coerce') * fps).apply(math.floor)
        frame_col = 'start_frame'

    # Ensure numeric type for comparison
    conv_df[frame_col] = pd.to_numeric(conv_df[frame_col], errors='coerce')
    if frame_range is not None:
        start_f, end_f = frame_range
        conv_df = conv_df[(conv_df[frame_col] >= start_f) & (conv_df[frame_col] <= end_f)]

    for _, row in conv_df.iterrows():
        add_event(row.get(frame_col), row.get('speaker_id'), row.get('word'))
        
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


def _parse_int_like(value):
    if value is None or (isinstance(value, float) and (pd.isna(value))):
        return None
    if isinstance(value, (int,)):
        return int(value)
    if isinstance(value, float):
        try:
            return int(value)
        except Exception:
            return None
    if isinstance(value, str):
        m = re.search(r'-?\d+', value)
        if m:
            try:
                return int(m.group(0))
            except Exception:
                return None
        return None
    return None


def _normalize_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ['person_id', 'speaker_id']:
        if col in df.columns:
            df[f'{col}_num'] = df[col].apply(_parse_int_like)
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
                det['person_id'] = _parse_int_like(row.get('person_id'))
            if 'person_id_num' in row:
                det['person_id'] = _parse_int_like(row.get('person_id_num'))
            if 'speaker_id' in row:
                det['speaker_id'] = _parse_int_like(row.get('speaker_id'))
            if 'speaker_id_num' in row:
                det['speaker_id'] = _parse_int_like(row.get('speaker_id_num'))
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
                det['speaker_id'] = _parse_int_like(row.get('speaker_id'))
            if 'speaker_id_num' in row:
                det['speaker_id'] = _parse_int_like(row.get('speaker_id_num'))
            detections.append(det)
        bodies_by_frame[int(frame_number)] = detections
    return bodies_by_frame


def _find_speaker_bbox_in_frame(head_tracking_df: pd.DataFrame, frame_index: int, speaker_id: int):
    """Try to locate the bounding box for the speaker in a given frame.
    Returns dict with x1..y2 or None.
    """
    if head_tracking_df is None or head_tracking_df.empty or speaker_id is None:
        return None

    # Try by person_id_num in MultiIndex if available
    if isinstance(head_tracking_df.index, pd.MultiIndex) and set(head_tracking_df.index.names) >= {'frame_number', 'person_id_num'}:
        try:
            row = head_tracking_df.loc[(frame_index, speaker_id)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            return {k: float(row[k]) for k in ['x1', 'y1', 'x2', 'y2'] if k in row}
        except KeyError:
            pass

    # Try by direct filtering on columns 'frame_number' + 'speaker_id'
    candidate_cols = set(head_tracking_df.columns)
    if {'frame_number', 'speaker_id_num'} <= candidate_cols:
        match = head_tracking_df[(head_tracking_df['frame_number'] == frame_index) & (head_tracking_df['speaker_id_num'] == speaker_id)]
        if not match.empty:
            row = match.iloc[0]
            return {k: float(row[k]) for k in ['x1', 'y1', 'x2', 'y2'] if k in match.columns}

    # Try by 'person_id_num' column (non-index)
    if {'frame_number', 'person_id_num'} <= candidate_cols:
        match = head_tracking_df[(head_tracking_df['frame_number'] == frame_index) & (head_tracking_df['person_id_num'] == speaker_id)]
        if not match.empty:
            row = match.iloc[0]
            return {k: float(row[k]) for k in ['x1', 'y1', 'x2', 'y2'] if k in match.columns}

    return None


def _predict_speaker_position(*args, **kwargs):
    # Deprecated per updated requirements; keep stub for compatibility.
    return None


def process_video(video_path=None, base_video=None, face_csv=None, body_csv=None, co_tracker_json=None, transcript_csv=None, output_json=None, fps: float = FPS, debug: bool = False):
    """
    Main function to orchestrate the data loading, processing, and saving.
    """
    # Determine paths
    if co_tracker_json or face_csv or transcript_csv or body_csv or base_video:
        # Use explicit arguments
        if not (base_video and co_tracker_json and transcript_csv and face_csv):
            print("Error: When using explicit arguments, please provide --base_video, --co_tracker_json, --transcript_csv, and --face_csv. --body_csv is optional.", file=sys.stderr)
            sys.exit(1)
        paths_info = derive_paths_from_args(base_video, face_csv, body_csv, co_tracker_json, transcript_csv, output_json)
    else:
        if not video_path:
            print("Error: Provide either a single video_path or explicit arguments.", file=sys.stderr)
            sys.exit(1)
        try:
            paths_info = parse_video_path(video_path)
        except (FileNotFoundError, ValueError) as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Attempt to read video properties only if a concrete video file is provided and exists
    img_width = img_height = None
    fps_read = fps or FPS
    frame_count = 0
    if video_path and os.path.isfile(video_path):
        print("\nReading video properties...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Could not open video file to read properties: {video_path}")
        else:
            img_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            img_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_read = cap.get(cv2.CAP_PROP_FPS) or FPS
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            cap.release()
            if img_width and img_height:
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
    if paths_info.get('body_detection_csv_path') and os.path.isfile(paths_info['body_detection_csv_path']):
        try:
            body_df = pd.read_csv(paths_info['body_detection_csv_path'])
        except Exception as e:
            print(f"Warning: Failed to read body detections: {e}")
            body_df = None
    else:
        if paths_info.get('body_detection_csv_path'):
            print(f"Warning: Body detection CSV not found at {paths_info['body_detection_csv_path']}. Continuing without it.")

    print("Preprocessing data for efficient lookup...")
    frame_range = paths_info.get('frame_range')
    speaker_events_by_frame = preprocess_transcriptions(transcriptions_df, paths_info.get('conversation_id_substr', ''), frame_range, fps=fps_read)
    # Prepare indices and per-frame groupings for efficient lookup
    # Apply frame range filtering to detections if present
    if paths_info.get('frame_range') is not None:
        start_f, end_f = paths_info['frame_range']
        if 'frame_number' in head_tracking_df.columns:
            head_tracking_df = head_tracking_df[(head_tracking_df['frame_number'] >= start_f) & (head_tracking_df['frame_number'] <= end_f)]
        if body_df is not None and 'frame_number' in body_df.columns:
            body_df = body_df[(body_df['frame_number'] >= start_f) & (body_df['frame_number'] <= end_f)]

    # Normalize id columns to numeric forms
    head_tracking_df = _normalize_id_columns(head_tracking_df)
    if body_df is not None:
        body_df = _normalize_id_columns(body_df)

    if set(['frame_number', 'person_id_num']).issubset(head_tracking_df.columns):
        head_tracking_multiindex = head_tracking_df.set_index(['frame_number', 'person_id_num']).sort_index()
    else:
        head_tracking_multiindex = head_tracking_df  # fallback without MultiIndex

    faces_by_frame = _group_face_detections_by_frame(head_tracking_df)
    bodies_by_frame = _group_body_detections_by_frame(body_df) if body_df is not None else {}
    angular_frames_by_index = {frame['frame_index']: frame for frame in angular_data['frames']}

    if debug:
        print(
            "\n[DEBUG] Dataset summaries:\n"
            f"  - Transcript events frames: {len(speaker_events_by_frame)} entries (min={min(speaker_events_by_frame.keys()) if speaker_events_by_frame else None}, max={max(speaker_events_by_frame.keys()) if speaker_events_by_frame else None})\n"
            f"  - Face detections frames: {len(faces_by_frame)} entries (min={min(faces_by_frame.keys()) if faces_by_frame else None}, max={max(faces_by_frame.keys()) if faces_by_frame else None})\n"
            f"  - Body detections frames: {len(bodies_by_frame)} entries (min={min(bodies_by_frame.keys()) if bodies_by_frame else None}, max={max(bodies_by_frame.keys()) if bodies_by_frame else None})"
        )

    print("Augmenting frame data...")
    total_frames = len(angular_data['frames'])
    joined_frames = []
    missing_debug_reported = 0
    processed_frame_count = 0
    for i, frame_obj in enumerate(angular_data['frames']):
        if (i + 1) % 100 == 0 or i == total_frames - 1:
            print(f"  Processing frame {i+1}/{total_frames}...")
            
        current_frame_index = frame_obj['frame_index']
        # Map local frame index to global frame number if a slice is used
        if frame_range is not None:
            start_f, _end_f = frame_range
            global_frame_index = int(start_f) + int(current_frame_index)
        else:
            global_frame_index = int(current_frame_index)
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
        out_frame['speaker_location'] = {}

        if global_frame_index in speaker_events_by_frame:
            speaker_events = speaker_events_by_frame[global_frame_index]
            if speaker_events:
                speaker_id = speaker_events[0]['id']
                out_frame['speaker_id'] = speaker_id
                # Try to find speaker bbox by scanning face detections for this global frame
                face_list = faces_by_frame.get(global_frame_index, [])
                for det in face_list:
                    det_speaker = det.get('speaker_id') if det.get('speaker_id') is not None else det.get('speaker_id_num')
                    if det_speaker == speaker_id:
                        try:
                            out_frame['speaker_location'] = {
                                'x1': float(det['x1']),
                                'y1': float(det['y1']),
                                'x2': float(det['x2']),
                                'y2': float(det['y2']),
                            }
                        except Exception:
                            out_frame['speaker_location'] = {}
                        break
                # Fallback: index-based lookup if not found in face list
                if not out_frame['speaker_location']:
                    loc = _find_speaker_bbox_in_frame(
                        head_tracking_multiindex if isinstance(head_tracking_multiindex, pd.DataFrame) else head_tracking_df,
                        global_frame_index,
                        speaker_id,
                    )
                    if loc:
                        out_frame['speaker_location'] = loc

        # Attach detections for the frame (global index)
        out_frame['face_detection'] = faces_by_frame.get(global_frame_index, [])
        out_frame['body_detection'] = bodies_by_frame.get(global_frame_index, [])

        if debug and (out_frame['speaker_id'] is None or not out_frame['face_detection']):
            if missing_debug_reported < 20:
                print(f"[DEBUG] Frame local={current_frame_index} global={global_frame_index} | speaker_in_transcript={global_frame_index in speaker_events_by_frame} | faces={len(out_frame['face_detection'])} | bodies={len(out_frame['body_detection'])}")
                if global_frame_index in speaker_events_by_frame:
                    print(f"        speaker_events={speaker_events_by_frame[global_frame_index]}")
                missing_debug_reported += 1

        joined_frames.append(out_frame)
        processed_frame_count += 1

    # Build joined output structure
    metadata = angular_data.get('metadata', {}).copy()
    analysis_summary = angular_data.get('analysis_summary', {}).copy()

    output_joined = {
        'metadata': {
            'video_name': paths_info.get('video_basename') or (os.path.splitext(os.path.basename(video_path))[0] if video_path else None),
            'video_path': video_path or paths_info.get('base_video_path'),
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
    print(f"Total frames processed: {processed_frame_count}")
    try:
        with open(output_path, 'w') as f:
            json.dump(output_joined, f, indent=2)
    except Exception as e:
        print(f"Error saving file: {e}", file=sys.stderr)
        sys.exit(1)

    print("Successfully finished.")
    return processed_frame_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Join transcripts, face detections, body detections, and angular analysis into a single JSON per video.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Mode 1: Legacy single argument (derive all other paths)
    parser.add_argument(
        "video_path", nargs='?', type=str,
        help="Full path to the input video file for deriving other paths. Optional if explicit paths are provided."
    )
    # Mode 2: Explicit paths
    parser.add_argument("--base_video", type=str, help="Base video name or path used to parse frame range and name.")
    parser.add_argument("--face_csv", type=str, help="Path to face detection CSV (global gallery with speaker).")
    parser.add_argument("--body_csv", type=str, default=None, help="Optional path to body detection CSV.")
    parser.add_argument("--co_tracker_json", type=str, help="Path to co-tracker ground-truth JSON.")
    parser.add_argument("--transcript_csv", type=str, help="Path to transcript CSV (with frames or startTime).")
    parser.add_argument("--output_json", type=str, default=None, help="Optional output path for joined JSON.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second for timestamp/frame computations (default 30).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging for missing speaker/face detections.")
    
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
    process_video(
        video_path=args.video_path,
        base_video=args.base_video,
        face_csv=args.face_csv,
        body_csv=args.body_csv,
        co_tracker_json=args.co_tracker_json,
        transcript_csv=args.transcript_csv,
        output_json=args.output_json,
        fps=float(args.fps),
        debug=bool(args.debug),
    )