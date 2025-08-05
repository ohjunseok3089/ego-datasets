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
                        subclip.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)

                    except Exception as e:
                        print(f"    - ERROR processing time range '{time_range_str}' for {video_file}: {e}")
                        continue

        except Exception as e:
            print(f"--- ERROR: Failed to process video file {video_file}: {e} ---")
            continue

    print("\nVideo processing complete.")
if __name__ == "__main__":
    # Define the base path for source videos and the output directory path
    video_path = "/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/"
    output_path = "/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/parts/"

    # Raw data provided by the user
    raw_processing_data = """
    vid_001__day_1__con_1__person_1_part1.MP4
    Start ~ 01:04 (social_interaction)
    01:06~01:19 (social_interaction)
    01:21~01:28 (social_interaction)
    01:48 ~ 02:00 (object_manipulation)
    02:18 ~ 02:23 (social_interaction)
    02:43 ~ 02:51 (social_interaction)
    vid_002__day_1__con_1__person_1_part2.MP4
    Start ~ 01:33 (social_interaction)
    02:26~04:39 (social_interaction)
    vid_003__day_1__con_1__person_1_part3.MP4
    Start ~ 00:38 (object_manipulation)
    00:38 ~ 01:00 (social_interaction)
    01:10 ~ 02:05 (social_interaction)
    02:15 ~ End (social_interaction)
    vid_004__day_1__con_1__person_1_part4.MP4
    Start ~ End (social_interaction)
    vid_005__day_1__con_1__person_1_part5.MP4
    Start ~ End (social_interaction)
    vid_006__day_1__con_1__person_2_part1.MP4
    Start ~ 01:04 (social_interaction)
    01:06~01:28 (social_interaction)
    03:38 ~ 04:15 (social_interaction)
    vid_007__day_1__con_1__person_2_part2.MP4
    Start ~ 01:40 (social_interaction)
    02:26~04:39 (social_interaction)
    vid_008__day_1__con_1__person_2_part3.MP4
    Start ~ 0:13 (object_manipulation)
    0:15 ~ 0:38 (object_manipulation)
    0:38 ~ 01:00 (social_interaction)
    01:13 ~ 01:40 (social_interaction)
    01:10 ~ 02:05 (social_interaction)
    02:15 ~ End (social_interaction)
    vid_009__day_1__con_1__person_2_part4.MP4
    Start ~ End (social_interaction)
    vid_010__day_1__con_1__person_2_part5.MP4
    Start ~ End (social_interaction)
    vid_011__day_1__con_1__person_3_part1.MP4
    Start ~ 01:04 (social_interaction)
    01:06~01:19 (social_interaction)
    01:21~01:28 (social_interaction)
    01:48 ~ 02:00 (object_manipulation)
    02:43 ~ 02:51 (social_interaction)
    vid_012__day_1__con_1__person_3_part2.MP4
    Start ~ 01:40 (social_interaction)
    02:26~04:39 (social_interaction)
    vid_013__day_1__con_1__person_3_part3.MP4
    0:12 ~ 0:32 (object_manipulation)
    0:34 ~ 01:00 (social_interaction)
    01:10 ~ 01:34 (object_manipulcation)
    01:38 ~ 02:05 (social_interaction)
    02:15 ~ End (social_interaction)
    vid_014__day_1__con_1__person_3_part4.MP4
    Start ~ End (social_interaction)
    vid_015__day_1__con_1__person_3_part5.MP4
    Start ~ End (social_interaction)
    vid_016__day_1__con_2__person_1_part1.MP4
    Start ~ 00:58 (social_interaction)
    01:17 ~ 01:26 (social_interaction)
    01:40 ~ 02:28 (social_interaction)
    02:36 ~ 03:10 (social_interaction)
    03:22 ~ End (social_interaction)
    vid_017__day_1__con_2__person_1_part2.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_018__day_1__con_2__person_1_part3.MP4
    Start ~ 02:10 (social_interaction)
    02:28 ~ 03:24 (social_interaction)
    vid_019__day_1__con_2__person_1_part4.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_020__day_1__con_2__person_1_part5.MP4
    Start ~ 0:29 (social_interaction)
    vid_021__day_1__con_2__person_2_part1.MP4
    Start ~ 01:00 (social_interaction)
    01:22 ~ 02:28 (social_interaction)
    02:39 ~ 03:05 (social_interaction)
    03:22 ~ End (social_interaction)
    vid_022__day_1__con_2__person_2_part2.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_023__day_1__con_2__person_2_part3.MP4
    Start ~ 03:24 (social_interaction)
    vid_024__day_1__con_2__person_2_part4.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_025__day_1__con_2__person_2_part5.MP4
    Start ~ 0:20 (social_interaction)
    vid_026__day_1__con_2__person_3_part1.MP4
    Start ~ 01:00 (social_interaction)
    01:17 ~ 01:26 (social_interaction)
    01:40 ~ 02:28 (social_interaction)
    02:36 ~ 03:10 (social_interaction)
    03:22 ~ End (social_interaction)
    vid_027__day_1__con_2__person_3_part2.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_028__day_1__con_2__person_3_part3.MP4
    Start ~ 02:10 (social_interaction)
    02:28 ~ 03:24 (social_interaction)
    vid_029__day_1__con_2__person_3_part4.MP4
    Start ~ 02:20 (social_interaction)
    02:35 ~ End (social_interaction)
    vid_030__day_1__con_2__person_3_part5.MP4
    Start ~ End (social_interaction)
    vid_031__day_1__con_3__person_1_part1.MP4
    Start ~ 01:55 (social_interaction)
    02:00 ~ 02:07 (object_manipulation)
    03:13 ~ End (social_interaction)
    vid_032__day_1__con_3__person_1_part2.MP4
    Start ~ 01:24 (social_interaction)
    01:52 ~ 02:23 (social_interaction)
    02:35 ~ 03:07 (social_interaction)
    03:20 ~ 03:34 (social_interaction)
    04:15 ~ End (social_interaction)
    vid_033__day_1__con_3__person_1_part3.MP4
    Start ~ End (social_interaction)
    vid_034__day_1__con_3__person_1_part4.MP4
    Start ~ End (social_interaction)
    vid_035__day_1__con_3__person_2_part1.MP4
    Start ~ 01:55 (social_interaction)
    02:02 ~ 02:07 (object_manipulation)
    02:08 ~ 02:40 (social_interaction)
    02:49 ~ 03:02 (social_interaction)
    03:35 ~ End (social_interaction)
    vid_036__day_1__con_3__person_2_part2.MP4
    Start ~ 01:24 (social_interaction)
    01:52 ~ 02:40 (social_interaction)
    04:15 ~ End (social_interaction)
    vid_037__day_1__con_3__person_2_part3.MP4
    Start ~ End (social_interaction)
    vid_038__day_1__con_3__person_2_part4.MP4
    Start ~ End (social_interaction)
    vid_039__day_1__con_3__person_3_part1.MP4
    Start ~ 01:55 (social_interaction)
    02:00 ~ 02:06 (object_manipulation)
    02:07 ~ 2:20 (social_interaction)
    03:13 ~ End (social_interaction)
    vid_040__day_1__con_3__person_3_part2.MP4
    Start ~ 01:24 (social_interaction)
    01:52 ~ 03:07 (social_interaction)
    03:23 ~ 03:29 (social_interaction)
    04:15 ~ End (social_interaction)
    vid_041__day_1__con_3__person_3_part3.MP4
    Start ~ End (social_interaction)
    vid_042__day_1__con_3__person_3_part4.MP4
    Start ~ End (social_interaction)
    vid_043__day_1__con_4__person_1_part1.MP4
    Start ~ 01:15 (social_interaction)
    01:20 ~. 1:30 (social_interaction)
    01:44 ~ 01:51 (social_interaction)
    02:08 ~ 02:56 (social_interaction)
    03:21 ~ End (social_interaction)
    vid_044__day_1__con_4__person_1_part2.MP4
    Start ~ 0:20 (social_interaction)
    1:13 ~ 1:23 (social_interaction)
    1:42 ~ 1:58 (social_interaction)
    2:12 ~ 2:19 (social_interaction)
    2:27 ~ 2:30 (social_interaction)
    2:44 ~ End (social_interaction)
    vid_045__day_1__con_4__person_1_part3.MP4
    Start ~ 0:43 (social_interaction)
    0:46 ~ 02:15 (social_interaction)
    02:20 03:19 (social_interaction)
    04:20 ~ End (social_interaction)
    vid_046__day_1__con_4__person_1_part4.MP4
    Start ~ 01:50 (social_interaction)
    02:08 ~ 03:44 (social_interaction)
    vid_047__day_1__con_4__person_2_part1.MP4
    Start ~ 01:39 (social_interaction)
    02:36 ~ 2:53 (social_interaction)
    3:24 ~ End (social_interaction)
    vid_048__day_1__con_4__person_2_part2.MP4
    02:16 ~ End (social_interaction)
    vid_049__day_1__con_4__person_2_part3.MP4
    Start ~ 03:19 (social_interaction)
    04:18 ~ End (social_interaction)
    vid_050__day_1__con_4__person_2_part4.MP4
    Start ~ End (social_interaction)
    vid_051__day_1__con_4__person_3_part1.MP4
    Start ~ 01:38 (social_interaction)
    01:47 ~ 01:55 (social_interaction)
    03:26 ~ End (social_interaction)
    vid_052__day_1__con_4__person_3_part2.MP4
    Start ~ 0:50 (social_interaction)
    0-1:13 ~ 01:23 (social_interaction)
    01:31 ~ 02:35 (social_interaction)
    02:45 ~ End (social_interaction)
    vid_053__day_1__con_4__person_3_part3.MP4
    start ~ 03:22 (social_interaction)
    04:15 ~ End (social_interaction)
    vid_054__day_1__con_4__person_3_part4.MP4
    start ~ End (social_interaction)
    vid_055__day_1__con_5__person_1_part1.MP4
    start ~ 01:09 (social_interaction)
    01:22 ~ 01:31 (social_interaction)
    01:51 ~ 02:00 (object_manipulation)
    02:03 ~ 02:32 (object_manipulation)
    02:48 ~ 02:58 (social_interaction)
    03:04 ~ 03:11 (object_manipulation)
    03:13 ~ 03:45 (object_manipulation)
    04:00 ~ 04:33 (social_interaction)
    vid_056__day_1__con_5__person_1_part2.MP4
    00:33 ~ 00:39 (social_interaction)
    01:28 ~ 02:15 (social_interaction)
    03:42 ~ End (collaborative_task)
    vid_057__day_1__con_5__person_1_part3.MP4
    Start ~ 03:20 (collaborative_task)
    03:54 ~ End (social_interaction)
    vid_058__day_1__con_5__person_1_part4.MP4
    Start ~ 02:10 (social_interaction)
    02:19 ~ 02:25 (object_manipulation)
    02:44 ~ 02:56 (social_interaction)
    03:07 ~ End (social_interaction)
    vid_059__day_1__con_5__person_1_part5.MP4
    Start ~ End (social_interaction)
    vid_060__day_1__con_5__person_2_part1.MP4
    start ~ 01:09 (social_interaction)
    02:03 ~ 02:35 (object_manipulation)
    02:53 ~ 03:34 (object_manipulation)
    04:00 ~ 04:37 (social_interaction)
    vid_061__day_1__con_5__person_2_part2.MP4
    00:33 ~ 00:39 (social_interaction)
    01:33 ~ 02:15 (social_interaction)
    03:38 ~ End (collaborative_task)
    vid_062__day_1__con_5__person_2_part3.MP4
    Start ~ 03:43 (collaborative_task)
    03:54 ~ End (social_interaction)
    vid_063__day_1__con_5__person_2_part4.MP4
    Start ~ 02:10 (social_interaction)
    02:44 ~ 02:57 (object_manipulation)
    03:07 ~ End (social_interaction)
    vid_064__day_1__con_5__person_2_part5.MP4
    Start ~ End (social_interaction)
    vid_065__day_2__con_1__person_1_part1.MP4
    Start ~ 01:40 (social_interaction)
    01:48 ~ 02:05 (social_interaction)
    02:16 ~ 03:07 (object_manipulation)
    03:17 ~ 03:27 (social_interaction)
    03:40 ~ 03:50 (social_interaction)
    03:54 ~ 04:14 (object_manipulation)
    vid_066__day_2__con_1__person_1_part2.MP4
    0:22 ~ 0:33 (social_interaction)
    0:42 ~ 0:58 (social_interaction)
    01:32 ~ 03:10 (social_interaction)
    03:35 ~ 04:24 (social_interaction)
    04:35 ~ End (social_interaction)
    vid_067__day_2__con_1__person_1_part3.MP4
    Start ~ 01:44 (social_interaction)
    01:55 ~ 02:10 (social_interaction)
    03:15 ~ End (collaborative_task)
    vid_068__day_2__con_1__person_1_part4.MP4
    Start ~ End (collaborative_task)
    vid_069__day_2__con_1__person_1_part5.MP4
    Start ~ 02:24 (collaborative_task)
    02:32 ~ 02:57 (social_interaction)
    03:03 ~ End (social_interaction)
    vid_070__day_2__con_1__person_2_part1.MP4
    Start ~ 02:10(social_interaction)
    02:11 ~ 02:37 (object_manipulation)
    02:40 ~ 03:03 (object_manipulation)
    03:03 ~ 03:10 (social_interaction)
    03:22 ~ 03:47 (social_interaction)
    03:55 ~ 04:05 (object_manipulation)
    vid_071__day_2__con_1__person_2_part2.MP4
    Start ~ 01:05 (social_interaction)
    01:20 ~ 03:18 (social_interaction)
    03:35 ~ End (social_interaction)
    vid_072__day_2__con_1__person_2_part3.MP4
    Start ~ 02:12 (social_interaction)
    02:30 ~ End (collaborative_task)
    vid_073__day_2__con_1__person_2_part4.MP4
    Start ~ End (collaborative_task)
    vid_074__day_2__con_1__person_2_part5.MP4
    Start ~ 02:24 (collaborative_task)
    02:36 ~ 03:00 (social_interaction)
    03:07 ~ End (social_interaction)
    vid_075__day_2__con_1__person_3_part1.MP4
    Start ~ 01:41 (social_interaction)
    02:15 ~ 02:35 (object_manipulation)
    03:16 ~ 03:55 (social_interaction)
    03:55 ~ 04:13 (object_manipulation)
    vid_076__day_2__con_1__person_3_part2.MP4
    Start ~ 0:08 (social_interaction)
    0:10 ~ 0:37 (social_interaction)
    01:25 ~ 03:17 (social_interaction)
    03:29 ~ End (social_interaction)
    vid_077__day_2__con_1__person_3_part3.MP4
    Start ~ 02:12 (social_interaction)
    02:30 ~ End (collaborative_task)
    vid_078__day_2__con_1__person_3_part4.MP4
    Start ~ End (collaborative_task)
    vid_079__day_2__con_1__person_3_part5.MP4
    Start ~ 02:24 (collaborative_task)
    02:34 ~ 03:00 (social_interaction)
    03:07 ~ End (social_interaction)
    vid_080__day_2__con_2__person_1_part1.MP4
    0:04 ~ 03:57 (social_interaction)
    04:09 ~ End (object_manipulation)
    vid_081__day_2__con_2__person_1_part2.MP4
    Start ~ 00:10 (object_manipulation)
    00:20 ~ End (social_interaction)
    vid_082__day_2__con_2__person_1_part3.MP4
    Start ~ 02:14 (social_interaction)
    02:37 ~ End (collaborative_task)
    vid_083__day_2__con_2__person_1_part4.MP4
    Start ~ 03:10 (collaborative_task)
    03:13 ~ End (collaborative_task)
    vid_084__day_2__con_2__person_2_part1.MP4
    0:02 ~ 03:56 (social_interaction)
    vid_085__day_2__con_2__person_2_part2.MP4
    0:20 ~ End (social_interaction)
    vid_086__day_2__con_2__person_2_part3.MP4
    Start ~ 02:14 (social_interaction)
    02:33 ~ End (collaborative_task)
    vid_087__day_2__con_2__person_2_part4.MP4
    Start ~ 03:12 (collaborative_task)
    03:13 ~ 03:25 (collaborative_task)
    """

    # Parse the raw data into the required dictionary format
    video_processing_data = parse_processing_data(raw_processing_data)

    # Run the splitting process
    split_videos(video_path, video_processing_data, output_path)