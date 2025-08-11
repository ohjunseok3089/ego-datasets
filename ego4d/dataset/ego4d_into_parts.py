import os
import re
from moviepy.editor import VideoFileClip

def parse_processing_data(raw_data_string):
    """
    Parses a multiline string of video processing instructions into a dictionary.

    Args:
        raw_data_string (str): The raw text data containing filenames and time ranges.

    Returns:
        dict: A dictionary where keys are filenames and values are lists of
              time range strings with categories.
    """
    data = {}
    current_file = None
    lines = raw_data_string.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if the line is a filename
        if line.lower().endswith('.mp4'):
            current_file = os.path.basename(line)
            data[current_file] = []
        elif current_file:
            # This line is a time range for the current file
            data[current_file].append(line)
    return data

def time_to_seconds(time_str):
    """Converts a time string in MM:SS or H:MM:SS format to seconds."""
    parts = time_str.split(':')
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    elif len(parts) == 3:
        hrs, mins, secs = map(int, parts)
        return hrs * 3600 + mins * 60 + secs
    else:
        # Handle single number for seconds (e.g., '0:29' becomes '29')
        try:
            return int(time_str)
        except ValueError:
            raise ValueError(f"Invalid time format for: {time_str}")
def split_videos(video_base_path, processing_data, output_folder):
    """
    Splits video files based on the provided data, including categories and frame info in filenames.
    Skips files that already exist in the output folder.
    """
    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)

    print("Scanning for existing files to skip...")
    try:
        # Use a set for fast lookups
        existing_files = set(os.listdir(output_folder))
        print(f"Found {len(existing_files)} existing files in '{output_folder}'.")
    except FileNotFoundError:
        existing_files = set()

    for video_file, time_ranges_with_cat in processing_data.items():
        source_video_path = os.path.join(video_base_path, video_file)

        if not os.path.exists(source_video_path):
            print(f"--- WARNING: Source file not found, skipping: {source_video_path} ---")
            continue

        print(f"\nProcessing video: {source_video_path}")

        try:
            with VideoFileClip(source_video_path) as clip:
                duration = clip.duration
                fps = clip.fps
                if not fps:
                    print(f"  - WARNING: Could not determine FPS for {video_file}. Skipping.")
                    continue
                
                base_name, extension = os.path.splitext(os.path.basename(video_file))

                for time_range_str in time_ranges_with_cat:
                    try:
                        match = re.match(r'(.+?)\s*\((.+)\)', time_range_str)
                        if not match:
                            print(f"    - WARNING: Invalid format for time range string, skipping: '{time_range_str}'")
                            continue

                        time_part, category = match.groups()
                        category = category.strip().replace('manipulcation', 'manipulation')

                        time_parts = re.split(r'\s*[~-]\s*|\s+', time_part.strip())
                        if len(time_parts) != 2:
                            if len(time_parts) > 2 and ':' in time_parts[1]:
                                start_time_str = time_parts[0]
                                end_time_str = time_parts[1]
                            else:
                                print(f"    - WARNING: Could not parse start/end time from '{time_part}', skipping.")
                                continue
                        else:
                            start_time_str, end_time_str = [t.strip() for t in time_parts]

                        start_time_sec = 0 if start_time_str.lower() == 'start' else time_to_seconds(start_time_str)
                        end_time_sec = duration if end_time_str.lower() == 'end' else time_to_seconds(end_time_str)

                        if start_time_sec >= duration:
                            print(f"  - WARNING: Start time ({start_time_sec:.2f}s) is beyond video duration ({duration:.2f}s). Skipping range '{time_range_str}'.")
                            continue
                        
                        end_time_sec = min(end_time_sec, duration)

                        if start_time_sec >= end_time_sec:
                            print(f"  - WARNING: Start time ({start_time_sec:.2f}s) is not before end time ({end_time_sec:.2f}s). Skipping range '{time_range_str}'.")
                            continue

                        start_frame = int(start_time_sec * fps)
                        end_frame = int(end_time_sec * fps)

                        output_filename = f"{base_name}({start_frame}_{end_frame}_{category}){extension}"
                        
                        if output_filename in existing_files:
                            print(f"  - SKIPPING (already exists): {output_filename}")
                            continue  # Move to the next time range

                        output_path = os.path.join(output_folder, output_filename)

                        print(f"  - CREATING clip: {output_filename} from {start_time_sec:.2f}s to {end_time_sec:.2f}s")

                        subclip = clip.subclip(start_time_sec, end_time_sec)
                        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None, 
                                              temp_audiofile='temp-audio.m4a', 
                                              remove_temp=True, threads=16)

                    except Exception as e:
                        print(f"    - ERROR processing time range '{time_range_str}' for {video_file}: {e}")
                        continue

        except Exception as e:
            print(f"--- ERROR: Failed to process video file {video_file}: {e} ---")
            continue

    print("\nVideo processing complete.")
if __name__ == "__main__":
    # Define the base path for source videos and the output directory path
    video_path = "/mas/robots/prg-ego4d/raw/v2/full_scale/"
    output_path = "/mas/robots/prg-ego4d/parts/"

    # Raw data provided by the user
    raw_processing_data = """
    30294c41-c90d-438a-af19-c1c74787d06b.mp4
    1:05 ~ 1:01:52 (collaborative_task)
    566ad4e5-1ce4-4679-9d19-ef63072c848c.mp4
    0:00 ~ 42:20 (collaborative_task)
    42:20 ~ End (collaborative_task)
    9c5b7322-d1cc-4b56-ae9d-85831f28fac1.mp4
    1:00 ~ 1:00:17 (collaborative_task)
    9ca2dc18-2c57-44cb-8c91-4b8b5c7ca223.mp4
    2:40 ~ 47:35 (collaborative_task)
    47:35 ~ End (collaborative_task)
    a223fcb2-8ffa-4826-bd0c-91027cf1c11e.mp4
    0:23 ~ 5:13 (collaborative_task)
    5:13 ~ 19:50 (collaborative_task)
    19:50 ~ 50:12 (collaborative_task)
    50:12 ~ 1:00:12 (collaborative_task)
    b3937482-c973-4263-957d-1d5366329dad.mp4
    2:19 ~ 5:55 (collaborative_task)
    5:55 ~ 15:53 (collaborative_task)
    15:53 ~ 35:53 (collaborative_task)
    35:53 ~ 40:53 (collaborative_task)
    40:53 ~ 45:50 (collaborative_task)
    45:50 ~ 50:50 (collaborative_task)
    50:50 ~ 1:01:46 (collaborative_task)
    """

    # Parse the raw data into the required dictionary format
    video_processing_data = parse_processing_data(raw_processing_data)

    # Run the splitting process
    split_videos(video_path, video_processing_data, output_path)