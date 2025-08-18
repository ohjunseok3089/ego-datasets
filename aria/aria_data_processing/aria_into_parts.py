import os
import re
from moviepy.editor import VideoFileClip

def parse_processing_data(raw_data_string):
    """
    Parses a multiline string of video processing instructions into a dictionary.
    Updated for ARIA Dataset format.

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
            
        # Check if the line is a filename (starts with loc and doesn't contain parentheses for time ranges)
        if line.startswith('loc') and ('(' not in line or line.endswith('.mp4')):
            # Handle cases where filename might have additional info in parentheses
            if '(' in line and not line.endswith('.mp4'):
                # Extract just the filename part
                filename_part = line.split('(')[0].strip()
                current_file = filename_part + '.mp4'
                # If there's time range info on the same line, process it
                remaining_part = line[line.find('('):]
                if remaining_part and current_file:
                    data.setdefault(current_file, []).append(remaining_part)
            else:
                current_file = line if line.endswith('.mp4') else line + '.mp4'
                data[current_file] = []
        elif current_file and ('(' in line or line.startswith(('Start', '0:', '1:', '2:', '3:', '4:'))):
            # This line is a time range for the current file
            data[current_file].append(line)
    
    return data

def time_to_seconds(time_str):
    """Converts a time string in MM:SS or H:MM:SS format to seconds."""
    # Clean up the time string
    time_str = time_str.strip()
    
    # Handle special cases
    if time_str.lower() == 'start':
        return 0
    if time_str.lower() == 'end':
        return float('inf')  # Will be handled later
    
    # Remove any parentheses and extra text
    time_str = re.sub(r'\([^)]*\)', '', time_str).strip()
    
    parts = time_str.split(':')
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    elif len(parts) == 3:
        hrs, mins, secs = map(int, parts)
        return hrs * 3600 + mins * 60 + secs
    else:
        # Handle single number for seconds
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
                        # Handle various formats in ARIA dataset
                        # Format: "time ~ time (category)" or just "(category)" at end of filename
                        
                        # Clean up common typos
                        time_range_str = time_range_str.replace('collborative_task', 'collaborative_task')
                        
                        # Extract category
                        category_match = re.search(r'\(([^)]+)\)(?:\s*\*{0,2})?\.?$', time_range_str)
                        if not category_match:
                            print(f"    - WARNING: No category found in '{time_range_str}', skipping.")
                            continue
                        
                        category = category_match.group(1).strip()
                        
                        # Extract time part
                        time_part = time_range_str[:category_match.start()].strip()
                        
                        # Parse time range
                        if '~' in time_part:
                            time_parts = [t.strip() for t in time_part.split('~')]
                        elif ' - ' in time_part:
                            time_parts = [t.strip() for t in time_part.split(' - ')]
                        else:
                            print(f"    - WARNING: Could not parse time range from '{time_part}', skipping.")
                            continue
                        
                        if len(time_parts) != 2:
                            print(f"    - WARNING: Invalid time range format '{time_part}', skipping.")
                            continue
                        
                        start_time_str, end_time_str = time_parts
                        
                        # Handle "End (duration)" format
                        end_match = re.search(r'End\s*\(([^)]+)\)', end_time_str)
                        if end_match:
                            end_time_str = end_match.group(1)
                        elif end_time_str.lower() == 'end':
                            end_time_str = str(int(duration))

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

                        # Clean category name for filename
                        clean_category = re.sub(r'[^\w\s-]', '', category).strip().replace(' ', '_')
                        
                        output_filename = f"{base_name}({start_frame}_{end_frame}_{clean_category}){extension}"
                        
                        if output_filename in existing_files:
                            print(f"  - SKIPPING (already exists): {output_filename}")
                            continue

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
    video_path = "/mas/robots/prg-aria/dataset/"
    output_path = "/mas/robots/prg-aria/parts/"

    # Raw data for ARIA Dataset
    raw_processing_data = """
    loc1_script2_seq1_rec1
    0:47 ~ 1:35 (social_interaction)
    01:38 ~ 1:51 (social_interaction)
    loc1_script2_seq1_rec2
    0:37 ~ 0:42 (object_manipulation)
    0:47 ~ 1:35 (social_interaction)
    01:38 ~ 1:51 (social_interaction)
    loc1_script2_seq3_rec1
    0:32 ~ 3:16 (object_manipulation)
    loc1_script2_seq3_rec2
    0:32 ~ 3:15 (object_manipulation)
    loc1_script2_seq4_rec1
    1:01 ~ 1:09 (object_manipulation)
    1:36 ~ 1:44 (object_manipulation)
    1:51 ~ 3:34 (object_manipulation)
    loc1_script2_seq4_rec2
    1:05 ~ 3:29 (object_manipulation)
    loc1_script2_seq6_rec1
    Start ~ 0:36 (object_manipulation)
    loc1_script2_seq6_rec2 (collaborative_task)
    Start ~ 0:10 (object_manipulation)
    loc1_script2_seq7_rec1 (watching tv, social interaction)
    0:14 ~ 1:51 (social_interaction)
    1:51 ~ End (4:26)(social_interaction)
    loc1_script2_seq7_rec2
    0:16 ~ End(4:26) (social_interaction)
    loc1_script2_seq8_rec1
    0:14 ~ 0:59 (social_interaction)
    loc1_script2_seq8_rec2
    0:14 ~ 0:59 (social_interaction)
    loc1_script3_seq2_rec1
    Start ~ End (01:22) (object_manipulation)
    loc2_script2_seq1_rec1
    0:30 ~ 1:14 (social_interaction)
    1:14 ~ 1:16 (object_manipulation)
    1:16 ~ End (2:14) (social_interaction)
    loc2_script2_seq1_rec2
    1:15 ~ End (2:14) (social_interaction)
    loc2_script2_seq3_rec1
    0:38 ~ End(1:20)(social_interaction)
    loc2_script2_seq3_rec2
    0:47 ~ End (1:20) (object_manipulation)
    loc2_script2_seq4_rec1
    Start ~ 0:42 (social_interaction)
    0:42 ~ End (1:42) (object_manipulation)
    loc2_script2_seq5_rec1
    0:18 ~ End (4:15) (collaborative_task)
    loc2_script2_seq5_rec2
    Start ~ End (4:15) (collaborative_task)
    loc2_script2_seq6_rec1
    Start ~ End (1:03) (collaborative_task)
    loc2_script2_seq6_rec2
    Start ~ 0:54 (collaborative_task)
    loc2_script2_seq8_rec1
    Start ~ 1:07 (social_interaction)
    loc2_script2_seq8_rec2
    Start ~ 1:07 (social_interaction)
    loc2_script3_seq1_rec2
    Start ~ 0:37 (social_interaction)
    loc2_script3_seq2_rec1
    0:15 ~ End (1:22) (social_interaction)
    loc2_script3_seq2_rec2
    0:12 ~ End (1:22) (social_interaction)
    loc2_script3_seq4_rec1
    1:14 ~ End (4:10) (object_manipulation)
    loc2_script3_seq4_rec2
    2:10 ~ 3:57 (object_manipulation)
    loc3_script2_seq1_rec1
    0:22 ~ End (1:54) (social_interaction)
    loc3_script2_seq1_rec2
    0:58 ~ End (1:54) (social_interaction)
    loc3_script2_seq3_rec1
    1:22 ~ End(3:10) (object_manipulation)
    loc3_script2_seq3_rec2
    1:22 ~ End(3:10) (object_manipulation)
    loc3_script2_seq4_rec1
    1:04 ~ End (1:56) (object_manipulation)
    loc3_script2_seq4_rec2
    1:04 ~ End (1:56) (object_manipulation)
    loc3_script2_seq5_rec1
    Start ~ End (2:11) (collaborative_task)
    loc3_script2_seq5_rec2
    Start ~ End (2:11) (collaborative_task)
    loc3_script2_seq7_rec1
    0:18 ~ End (1:40) (object_manipulation)
    loc3_script2_seq7_rec2
    0:10 ~ End (1:40) (object_manipulation)
    loc3_script3_seq1_rec2
    Start ~ End (1:05) (social_interaction)
    loc3_script3_seq2_rec1
    0:10 ~ End(2:49) (social_interaction)
    loc3_script3_seq2_rec2
    Start ~ End (2:49) (social_interaction)
    loc3_script3_seq4_rec1
    3:36 ~ End (4:41) (object_manipulation)
    loc3_script3_seq4_rec2
    1:42 ~ End (4:41) (object_manipulation)
    """

    # Parse the raw data into the required dictionary format
    video_processing_data = parse_processing_data(raw_processing_data)

    # Run the splitting process
    split_videos(video_path, video_processing_data, output_path)