import os
import re
from moviepy.editor import VideoFileClip

def parse_processing_data(raw_data_string):
    """
    Parses a multiline string of video processing instructions into a dictionary.
    """
    data = {}
    current_file = None
    # Clean up common typos in filenames
    raw_data_string = raw_data_string.replace('.MP4.', '.MP4')
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
    # Remove spaces for robustness, e.g., "07 : 14" -> "07:14"
    time_str = time_str.replace(' ', '')
    parts = time_str.split(':')
    if len(parts) == 2:
        mins, secs = map(int, parts)
        return mins * 60 + secs
    elif len(parts) == 3:
        hrs, mins, secs = map(int, parts)
        return hrs * 3600 + mins * 60 + secs
    else:
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

    # Scan for existing files to implement skipping logic
    print("Scanning for existing files to skip...")
    try:
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
                        # Clean up data entry issues like "End (24:21)" before the category
                        time_range_str = re.sub(r'\(End\s*[\d:]*\)', '', time_range_str)
                        
                        match = re.match(r'(.+?)\s*\((.+)\)', time_range_str)
                        if not match:
                            print(f"    - WARNING: Invalid format for time range string, skipping: '{time_range_str}'")
                            continue

                        time_part, category = match.groups()
                        # Fix missing closing parenthesis and other typos
                        category = category.strip().replace(')', '').replace('manipulcation', 'manipulation')

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
                        
                        # Check if the file already exists and skip if it does
                        if output_filename in existing_files:
                            print(f"  - SKIPPING (already exists): {output_filename}")
                            continue

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
    # --- Configuration for 20-minute videos ---
    video_path = "/mas/robots/prg-egocom/EGOCOM/720p/20min/"
    output_path = "/mas/robots/prg-egocom/EGOCOM/720p/20min/parts/"

    # Raw data for the 20-minute video parts
    raw_processing_data_20min = """
    vid_088__day_2__con_3__person_1.MP4
    0:02 ~ 0:44 (social_interaction)
    0:50 ~ 01:20 (social_interaction)
    01:56 ~ 02:15 (object_manipulation)
    02:16 ~ 03:30 (social_interaction)
    03:59 ~ 05:27 (social_interaction)
    05:51 ~ 06:01 (object_manipulation)
    06:25 ~ 14:12 (social_interaction)
    14:22 ~ 14:42 (social_interaction)
    14:53 ~ End (collaborative_task)
    vid_089__day_2__con_3__person_2.MP4
    Start ~ 01:30 (social_interaction)
    03:59 ~ 05:40 (social_interaction)
    06:10 ~ 06:19 (social_interaction)
    06:27 ~ 14:44 (social_interaction)
    14:55 ~ End (collaborative_task)
    vid_090__day_2__con_4__person_1.MP4
    0:09 ~ 01:26 (social_interaction)
    01:26 ~ 02:05 (object_manipulation)
    02:20 ~ 02:27 (social_interaction)
    02:52 ~ 03:22 (social_interaction)
    03:27 ~ 3:48 (object_manipulation)
    04:27 ~ 16:37 (social_interaction)
    16:51 ~ End (collaborative_task)
    vid_091__day_2__con_4__person_2.MP4
    0:09 ~ 01:26 (social_interaction)
    01:26 ~ 02:05 (object_manipulation)
    02:20 ~ 02:44 (social_interaction)
    02:57 ~ 03:25 (social_interaction)
    03:29 ~ 03:44 (object_manipulation)
    04:26 ~ 16:37 (social_interaction)
    16:58 ~ End (collaborative_task)
    vid_092__day_2__con_4__person_3.MP4
    0:09 ~ 01:26 (social_interaction)
    01:26 ~ 02:05 (object_manipulation)
    02:15 ~ 02:34 (social_interaction)
    02:48 ~ 02:54 (social_interaction)
    03:55 ~ 04:08 (social_interaction)
    04:26 ~ 16:37 (social_interaction)
    16:58 ~ End (collaborative_task)
    vid_093__day_2__con_5__person_1.MP4
    Start ~ 0:35 (social_interaction)
    0:41 ~ 01:45 (social_interaction)
    02:00 ~ 02:12 (social_interaction)
    02:22 ~ 02:47 (social_interaction)
    03:11 ~ 03:34 (social_interaction)
    03:58 ~ 04:15 (social_interaction)
    04:22 ~ 04:40 (social_interaction)
    05:04 ~ 05:14 (social_interaction)
    05:20 ~ 05:40 (social_interaction)
    06:07 ~ 06:18 (social_interaction)
    06:30 ~ 13:00 (social_interaction)
    13:11 ~ 13:59 (social_interaction)
    14:18 ~ 20:08 (social_interaction)
    vid_094__day_2__con_5__person_2.MP4
    Start ~ 0:05 (social_interaction)
    0:09 ~ 01:49 (social_interaction)
    01:59 ~ 02:10 (social_interaction)
    02:22 ~ 02:31 (social_interaction)
    03:09 ~ 03:35 (social_interaction)
    04:05 ~ 04:54 (social_interaction)
    05:04 ~ 05:13 (social_interaction)
    05:18 ~ 06:00 (social_interaction)
    06:11 ~ 06:18 (social_interaction)
    06:35 ~ 20:08 (social_interaction)
    vid_095__day_2__con_5__person_3.MP4
    Start ~ 02:00 (social_interaction)
    02:30 ~ 02:45 (social_interaction)
    03:10 ~ 03:35 (social_interaction)
    04:10 ~ 04:25 (social_interaction)
    04:38 ~ 04:50 (social_interaction)
    05:00 ~ 05:50 (social_interaction)
    06:09 ~ 06:19 (social_interaction)
    06:30 ~ 13:00 (social_interaction)
    13:18 ~ 14:08 (social_interaction)
    14:27 ~ 20:08 (social_interaction)
    vid_096__day_2__con_6__person_1.MP4
    Start ~ 01:00 (social_interaction)
    01:07 ~ 05:47 (social_interaction)
    05:47 ~ 08:30 (collaborative_task)
    08:30 ~ 11:40 (social_interaction)
    12:00 ~ 13:34 (social_interaction)
    13:45 ~ 20:50 (collaborative_task)
    vid_098__day_2__con_6__person_3.MP4
    Start ~ 05:47 (social_interaction)
    05:47 ~ 08:30 (collaborative_task)
    08:30 ~ 11:40 (social_interaction)
    12:00 ~ 13:50 (social_interaction)
    14:00 ~ 16:28 (collaborative_task)
    16:33 ~ End (collaborative_task)
    vid_099__day_2__con_7__person_1.MP4
    Start ~ 01:20 (social_interaction)
    01:33 ~ 03:10 (social_interaction)
    03:16 ~ 03:30 (social_interaction)
    03:38 ~ 03:43 (social_interaction)
    03:49 ~ 04:25 (social_interaction)
    04:33 ~ 04:41 (object_manipulation)
    04:41 ~ 05:36 (social_interaction)
    05:45 ~ 09:45 (collaborative_task)
    09:53 ~ 16:50 (social_interaction)
    17:08 ~ End (collaborative_task)
    vid_100__day_2__con_7__person_2.MP4
    Start ~ 01:20 (social_interaction)
    01:33 ~ 03:50 (social_interaction)
    04:00 ~ 04:25 (social_interaction)
    04:33 ~ 04:41 (object_manipulation)
    04:41 ~ 05:36 (social_interaction)
    05:45 ~ 09:45 (collaborative_task)
    09:45 ~ 17:00 (social_interaction)
    17:12 ~ End (collaborative_task)
    vid_101__day_3__con_1__person_1.MP4
    0:10 ~ 0:27 (social_interaction)
    0:40 ~ 01:25 (social_interaction)
    01:50 ~ 03:30 (social_interaction)
    03:30 ~ 04:00 (object_manipulation)
    04:49 ~ 05:30 (object_manipulation)
    06:20 ~ 07:10 (object_manipulation)
    07:20 ~ 07:26 (social_interaction)
    08:08 ~ 09:20 (social_interaction)
    09:33 ~ 10:00 (object_manipulation)
    10:36 ~ 16:39 (social_interaction)
    16:50 ~ End (social_interaction)
    vid_102__day_3__con_1__person_2.MP4
    Start ~ 0:27 (social_interaction)
    0:43 ~ 01:30 (social_interaction)
    01:37 ~ 03:25 (social_interaction)
    03:55 ~ 06:05 (object_manipulation)
    06:16 ~ 07:15 (social_interaction)
    07:25 ~ 09:22 (social_interaction)
    09:35 ~ 10:00 (social_interaction)
    10:00 ~ End (social_interaction)
    vid_103__day_3__con_1__person_3.MP4
    Start ~ 0:27 (social_interaction)
    0:40 ~ 01:30 (social_interaction)
    01:47 ~ 03:25 (social_interaction)
    04:06 ~ 04:16 (object_manipulation)
    04:31 ~ 06:10 (object_manipulation)
    06:20 ~ 06:35 (social_interaction)
    06:48 ~ 06:58 (social_interaction)
    07:30 ~ 09:18 (social_interaction)
    09:33 ~ 18:08 (social_interaction)
    18:35 ~ End (social_interaction)
    vid_104__day_3__con_2__person_1.MP4
    Start ~ 0:16 (social_interaction)
    0:25 ~ 01:06 (social_interaction)
    01:10 ~ 01:50 (object_manipulation)
    01:55 ~ 02:15 (social_interaction)
    03:03 ~ 05:00 (social_interaction)
    05:53 ~ 06:00 (social_interaction)
    06:16 ~ 06:43 (object_manipulation)
    06:43 ~ 07:15 (social_interaction)
    07:23 ~ 22:08 (social_interaction)
    vid_106__day_3__con_2__person_3.MP4
    Start ~ 0:20 (social_interaction)
    0:26 ~ 01:07 (social_interaction)
    01:17 ~ 01:50 (object_manipulation)
    01:50 ~ 02:59 (social_interaction)
    03:11 ~ 05:41 (social_interaction)
    05:48 ~ 06:00 (social_interaction)
    06:11 ~ 06:50 (object_manipulation)
    06:55 ~ 07:15 (social_interaction)
    07:23 ~ 18:10 (social_interaction)
    18:37 ~ 22:08 (social_interaction)
    vid_107__day_3__con_3__person_1.MP4
    0:06 ~ 1:04 (social_interaction)
    1:06 ~ 01:50 (social_interaction)
    01:58 ~ 02:05 (object_manipulation)
    02:22 ~ 04:59 (social_interaction)
    06:18 ~ 14:35 (collaborative_task)
    15:04 ~ End (collaborative_task)
    vid_108__day_3__con_3__person_2.MP4
    0:07 ~ 01:01 (social_interaction)
    01:03 ~ 01:58 (social_interaction)
    01:58 ~ 02:05 (object_manipulation)
    02:05 ~ 02:10 (social_interaction)
    02:30 ~ 05:05 (social_interaction)
    06:10 ~ 14:35 (collaborative_task)
    14:51 ~ End (collaborative_task)
    vid_109__day_3__con_3__person_3.MP4
    0:05 ~ 01:05 (social_interaction)
    01:18 ~ 01:40 (social_interaction)
    02:25 ~ 04:59 (social_interaction)
    06:14 ~ 14:35 (collaborative_task)
    15:08 ~ End (collaborative_task)
    vid_110__day_3__con_4__person_1.MP4
    0:03 ~ 04:47 (social_interaction)
    05:00 ~ 06:45 (object_manipulation)
    06:58 ~ 10:20 (collaborative_task)
    10:25 ~ 18:00 (social_interaction)
    18:08 ~ 19:17 (social_interaction)
    19:28 ~ 24:53 (collaborative_task)
    24:57 ~ End (collaborative_task)
    vid_111__day_3__con_4__person_2.MP4
    0:03 ~ 04:48 (social_interaction)
    05:00 ~ 06:46 (object_manipulation)
    06:53 ~ 10:20 (collaborative_task)
    10:20 ~ 19:19 (social_interaction)
    19:30 ~ End (collaborative_task)
    vid_112__day_3__con_4__person_3.MP4
    0:03 ~ 04:48 (social_interaction)
    05:00 ~ 06:46 (object_manipulation)
    06:53 ~ 10:20 (collaborative_task)
    10:20 ~ 19:19 (social_interaction)
    19:32 ~ End (collaborative_task)
    vid_113__day_3__con_5__person_1.MP4
    Start ~ 06:21 (social_interaction)
    06:33 ~ 06:57 (social_interaction)
    07:05 ~ 09:07 (social_interaction)
    09:13 ~ 09:46 (object_manipulation)
    09:52 ~ 11:03 (social_interaction)
    11:08 ~ 16:23 (collaborative_task)
    16:37 ~ End (collaborative_task)
    vid_114__day_3__con_5__person_2.MP4
    Start ~ 06:21 (social_interaction)
    06:35 ~ 06:41 (social_interaction)
    06:46 ~ 06:55 (social_interaction)
    07:00 ~ 09:07 (social_interaction)
    09:07 ~ 09:50 (object_manipulation)
    10:05 ~ 10:38 (social_interaction)
    10:48 ~ 16:23 (collaborative_task)
    16:35 ~ End (collaborative_task)
    vid_115__day_3__con_5__person_3.MP4
    Start ~ 06:21 (social_interaction)
    06:35 ~ 06:59 (social_interaction)
    07:00 ~ 09:10 (social_interaction)
    09:30 ~ 09:50 (object_manipulation)
    09:56 ~ 10:00 (social_interaction)
    10:05 ~ 10:35 (social_interaction)
    10:58 ~ 16:24 (collaborative_task)
    16:32 ~ End (collaborative_task)
    vid_116__day_3__con_6__person_1.MP4
    Start ~ 02:00 (social_interaction)
    02:10 ~ 02:19 (social_interaction)
    02:49 ~ 03:20 (object_manipulation)
    03:57 ~ 05:35 (object_manipulation)
    05:48 ~ 06:18 (social_interaction)
    06:25 ~ 06:41 (social_interaction)
    06:53 ~ 07:20 (object_manipulation)
    07:25 ~ 07:48 (object_manipulation)
    08:00 ~ 09:53 (social_interaction)
    10:00 ~ 13:30 (social_interaction)
    13:50 ~ 17:37 (social_interaction)
    17:50 ~ End (collaborative_task)
    vid_117__day_3__con_6__person_2.MP4
    Start ~ 02:05 (social_interaction)
    02:14 ~ 02:27 (social_interaction)
    02:41 ~ 03:30 (object_manipulation)
    03:35 ~ 03:43 (object_manipulation)
    03:58 ~ 05:20 (object_manipulation)
    05:48 ~ 06:18 (social_interaction)
    06:25 ~ 06:41 (social_interaction)
    06:53 ~ 07:20 (object_manipulation)
    07:25 ~ 07:48 (object_manipulation)
    08:00 ~ 09:53 (social_interaction)
    10:00 ~ 13:30 (social_interaction)
    13:45 ~ 17:37 (social_interaction)
    17:50 ~ End (collaborative_task)
    vid_118__day_3__con_6__person_3.MP4
    0:04 ~ 0:35 (social_interaction)
    0:36 ~ 02:00 (social_interaction)
    02:41 ~ 03:35 (object_manipulation)
    03:38 ~ 03:54 (object_manipulation)
    03:57 ~ 05:20 (object_manipulation)
    05:28 ~ 05:35 (object_manipulation)
    05:43 ~ 06:16 (social_interaction)
    06:38 ~ 07:20 (object_manipulation)
    07:25 ~ 07:48 (object_manipulation)
    07:57 ~ 09:53 (social_interaction)
    10:00 ~ 13:30 (social_interaction)
    13:53 ~ 17:37 (social_interaction)
    17:48 ~ End (collaborative_task)
    vid_119__day_4__con_1__person_1.MP4
    0:02 ~ 03:00 (social_interaction)
    03:22 ~ 04:41 (social_interaction)
    04:42 ~ 06:00 (object_manipulation)
    06:10 ~ 10:34 (collaborative_task)
    10:34 ~ End (social_interaction)
    vid_120__day_4__con_1__person_2.MP4
    0:03 ~ 03:00 (social_interaction)
    03:20 ~ 04:41 (social_interaction)
    04:41 ~ 06:00 (object_manipulation)
    06:10 ~ 10:34 (collaborative_task)
    10:34 ~ End (social_interaction)
    vid_121__day_4__con_1__person_3.MP4
    0:03 ~ 03:00 (social_interaction)
    03:22 ~ 04:41 (social_interaction)
    04:41 ~ 06:00 (object_manipulation)
    06:08 ~ 10:34 (collaborative_task)
    10:34 ~ End (social_interaction)
    vid_122__day_4__con_2__person_1.MP4
    0:06 ~ 04:08 (social_interaction)
    04:25 ~ 05:04 (social_interaction)
    05:09 ~ 05:28 (social_interaction)
    05:37 ~ 05:55 (object_manipulation)
    05:56 ~ 06:13 (object_manipulation)
    06:14 ~ 07:08 (object_manipulation)
    07:14 ~ 07:56 (object_manipulation)
    08:10 ~ 13:24 (social_interaction)
    13:35 ~ 18:36 (social_interaction)
    18:43 ~ End (collaborative_task)
    vid_123__day_4__con_2__person_2.MP4
    0:05 ~ 04:10 (social_interaction)
    04:22 ~ 05:35 (social_interaction)
    05:44 ~ 07:58 (object_manipulation)
    08:13 ~ 10:54 (social_interaction)
    11:09 ~ 11:29 (social_interaction)
    11:37 ~ 12:45 (social_interaction)
    12:53 ~ 13:12 (social_interaction)
    13:15 ~ 13:24 (social_interaction)
    13:36 ~ 13:55 (social_interaction)
    14:04 ~ 18:43 (social_interaction)
    18:43 ~ End (collaborative_task)
    vid_124__day_4__con_3__person_1.MP4
    Start ~ 01:00 (social_interaction)
    01:00 ~ 02:55 (object_manipulation)
    03:10 ~ 03:49 (social_interaction)
    04:23 ~ 06:56 (object_manipulation)
    07:15 ~ 16:26 (social_interaction)
    16:30 ~ 19:00 (social_interaction)
    19:05 ~ End (social_interaction)
    vid_125__day_4__con_3__person_2.MP4
    Start ~ 01:00 (social_interaction)
    01:00 ~ 02:55 (object_manipulation)
    03:09 ~ 03:20 (social_interaction)
    03:30 ~ 06:56 (object_manipulation)
    06:56 ~ 19:00 (social_interaction)
    19:07 ~ End (social_interaction)
    vid_126__day_4__con_3__person_3.MP4
    Start ~ 01:01 (social_interaction)
    01:03 ~ 01:37 (object_manipulation)
    1:35 ~ 02:57 (object_manipulation)
    03:09 ~ 03:30 (social_interaction)
    03:30 ~ 06:56 (object_manipulation)
    06:56 ~ 20:36 (social_interaction)
    vid_127__day_4__con_4__person_1.MP4
    Start ~ 01:42 (social_interaction)
    01:50 ~ 02:08 (object_manipulation)
    02:18 ~ 02:29 (object_manipulation)
    02:37 ~ 04:46 (social_interaction)
    04:47 ~ 08:11 (social_interaction)
    08:16 ~ 14:20 (social_interaction)
    14:35 ~ End (collaborative_task)
    vid_128__day_4__con_4__person_2.MP4
    Start ~ 01:42 (social_interaction)
    01:50 ~ 02:08 (object_manipulation)
    02:15 ~ 02:29 (object_manipulation)
    02:29 ~ 08:11 (social_interaction)
    08:16 ~ 14:20 (social_interaction)
    14:36 ~ End (collaborative_task)
    vid_129__day_4__con_4__person_3.MP4
    Start ~ 01:39 (social_interaction)
    01:50 ~ 02:07 (object_manipulation)
    02:13 ~ 02:29 (object_manipulation)
    02:29 ~ 08:11 (social_interaction)
    08:16 ~ 14:20 (social_interaction)
    14:32 ~ End (collaborative_task)
    vid_130__day_4__con_5__person_1.MP4
    Start ~ 02:21 (social_interaction)
    02:21 ~ 02:31 (object_manipulation)
    02:31 ~ 02:57 (social_interaction)
    03:10 ~ 03:25 (social_interaction)
    03:30 ~ 04:26 (social_interaction)
    04:36 ~ 04:58 (object_manipulation)
    05:10 ~ 06:05 (object_manipulation)
    06:25 ~ 06:45 (object_manipulation)
    06:50 ~ 09:13 (object_manipulation)
    09:20 ~ 10:13 (object_manipulation)
    10:17 ~ 16:10 (object_manipulation)
    16:28 ~ 20:35 (collaborative_task)
    vid_131__day_4__con_5__person_2.MP4
    0:03 ~ 02:12 (social_interaction)
    02:16 ~ 02:21 (social_interaction)
    02:32 ~ 02:57 (social_interaction)
    03:10 ~ 03:26 (social_interaction)
    03:30 ~ 04:26 (social_interaction)
    04:36 ~ 05:00 (object_manipulation)
    05:10 ~ 05:25 (object_manipulation)
    05:29 ~ 06:05 (object_manipulation)
    06:22 ~ 06:37 (object_manipulation)
    06:40 ~ 15:42 (object_manipulation)
    15:53 ~ 16:10 (object_manipulation)
    16:40 ~ 20:43 (collaborative_task)
    vid_132__day_4__con_5__person_3.MP4
    0:04 ~ 02:14 (social_interaction)
    02:16 ~ 02:21 (social_interaction)
    02:21 ~ 02:31 (object_manipulation)
    02:31 ~ 02:55 (social_interaction)
    03:05 ~ 03:20 (social_interaction)
    03:25 ~ 04:26 (social_interaction)
    04:36 ~ 05:03 (object_manipulation)
    05:33 ~ 06:05 (object_manipulation)
    06:25 ~ 06:31 (object_manipulation)
    07:04 ~ 16:10 (object_manipulation)
    16:35 ~ 17:09 (collaborative_task)
    17:12 ~ 20:43 (collaborative_task)
    vid_133__day_4__con_6__person_1.MP4
    Start ~ 0:25 (social_interaction)
    0:35 ~ 0:40 (object_manipulation)
    0:44 ~ 01:01 (object_manipulation)
    01:13 ~ 01:57 (object_manipulation)
    02:06 ~ 02:09 (object_manipulation)
    02:15 ~ 02:26 (object_manipulation)
    02:30 ~ 02:42 (object_manipulation)
    03:04 ~ 03:56 (object_manipulation)
    04:16 ~ 13:58 (social_interaction)
    13:59 ~ 14:05 (social_interaction)
    14:13 ~ 14:22 (social_interaction)
    14:37 ~ End (collaborative_task)
    vid_134__day_4__con_6__person_2.MP4
    Start ~ 0:07 (social_interaction)
    0:12 ~ 0:24 (social_interaction)
    1:45 ~ 1:56 (object_manipulation)
    02:06 ~ 02:09 (object_manipulation)
    02:15 ~ 02:22 (object_manipulation)
    02:27 ~ 02:42 (object_manipulation)
    02:49 ~ 04:05 (object_manipulation)
    04:16 ~ 08:53 (social_interaction)
    08:56 ~ 09:05 (social_interaction)
    09:08 ~ 13:53 (social_interaction)
    14:04 ~ 14:06 (social_interaction)
    14:30 ~ End (collaborative_task)
    vid_135__day_4__con_6__person_3.MP4
    Start ~ 0:25 (social_interaction)
    0:35 ~ 1:24 (object_manipulation)
    1:26 ~ 1:30 (object_manipulation)
    1:38 ~ 02:01 (object_manipulation)
    02:04 ~ 02:12 (object_manipulation)
    02:16 ~ 02:26 (object_manipulation)
    02:37 ~ 02:44 (object_manipulation)
    02:56 ~ 03:42 (object_manipulation)
    03:50 ~ 04:00 (object_manipulation)
    04:16 ~ 13:58 (social_interaction)
    13:59 ~ 14:05 (social_interaction)
    14:13 ~ 14:22 (social_interaction)
    14:37 ~ End (collaborative_task)
    vid_136__day_5__con_1__person_1.MP4
    0:04 ~ End (social_interaction)
    vid_137__day_5__con_1__person_2.MP4
    Start ~ End (social_interaction)
    vid_138__day_5__con_2__person_1.MP4
    0:01 ~ 02:08 (social_interaction)
    02:30 ~ 08:10 (object_manipulation)
    08:15 ~ 09:00 (object_manipulation)
    09:05 ~ 10:06 (object_manipulation)
    10:09 ~ 10:18 (object_manipulation)
    10:23 ~ 10:42 (object_manipulation)
    10:50 ~ 11:46 (object_manipulation)
    11:59 ~ End (social_interaction)
    vid_139__day_5__con_2__person_2.MP4
    0:02 ~ 02:08 (social_interaction)
    02:34 ~ 10:10 (object_manipulation)
    10:15 ~ 11:46 (object_manipulation)
    11:59 ~ End (social_interaction)
    vid_140__day_5__con_3__person_1.MP4
    0:10 ~ 0:14 (social_interaction)
    0:19 ~ 10:15 (social_interaction)
    10:34 ~ 12:24 (object_manipulation)
    12:31 ~ 12:45 (object_manipulation)
    12:50 ~ 12:57 (object_manipulation)
    13:01 ~ 13:10 (object_manipulation)
    13:17 ~ 15:05 (object_manipulation)
    15:20 ~ End (social_interaction)
    vid_141__day_5__con_3__person_2.MP4
    00:05 ~ 10:15 (social_interaction)
    10:26 ~ 12:14 (object_manipulation)
    12:20 ~ 12:55 (object_manipulation)
    13:10 ~ 15:10 (object_manipulation)
    15:20 ~ End (social_interaction)
    vid_142__day_5__con_3__person_3.MP4
    0:06 ~ 10:15 (social_interaction)
    10:29 ~ 15:10 (object_manipulation)
    15:16 ~ End (social_interaction)
    vid_143__day_5__con_4__person_1.MP4
    Start ~ 04:22 (social_interaction)
    04:40 ~ 04:59 (object_manipulation)
    05:01 ~ 05:08 (object_manipulation)
    05:09 ~ 05:18 (object_manipulation)
    09:50 ~ 16:43 (social_interaction)
    16:54 ~ End (collaborative_task)
    vid_144__day_5__con_4__person_2.MP4
    Start ~ 04:21 (social_interaction)
    04:38 ~ 06:29 (object_manipulation)
    06:33 ~ 07:55 (object_manipulation)
    08:02 ~ 09:40 (object_manipulation)
    09:53 ~ 16:43 (social_interaction)
    16:54 ~ End (collaborative_task)
    vid_145__day_5__con_4__person_3.MP4
    Start ~ 04:22 (social_interaction)
    09:50 ~ 16:43 (social_interaction)
    16:52 ~ End (collaborative_task)
    vid_146__day_5__con_5__person_1.MP4
    Start ~ 01:12 (social_interaction)
    01:17 ~ 02:28 (object_manipulation)
    02:33 ~ 02:55 (social_interaction)
    03:18 ~ 04:55 (social_interaction)
    05:03 ~ 06:14 (object_manipulation)
    06:24 ~ 06:27 (social_interaction)
    06:36 ~ 07:04 (object_manipulation)
    07:10 ~ End (social_interaction)
    vid_147__day_5__con_5__person_2.MP4
    Start ~ 01:12 (social_interaction)
    01:12 ~ 02:19 (object_manipulation)
    02:28 ~ 02:33 (object_manipulation)
    02:37 ~ 04:08 (social_interaction)
    04:14 ~ 05:00 (social_interaction)
    05:00 ~ 07:04 (object_manipulation)
    07:04 ~ End (social_interaction)
    vid_148__day_5__con_5__person_3.MP4
    Start ~ 01:12 (social_interaction)
    01:12 ~ 02:33 (object_manipulation)
    02:33 ~ 05:00 (social_interaction)
    05:00 ~ 07:04 (object_manipulation)
    07:04 ~ End (social_interaction)
    vid_149__day_5__con_6__person_1.MP4
    0:06 ~ 07:23 (social_interaction)
    07:31 ~ End (social_interaction)
    vid_150__day_5__con_6__person_2.MP4
    Start ~ End (social_interaction)
    vid_151__day_5__con_6__person_3.MP4
    Start ~ End (social_interaction)
    vid_152__day_5__con_7__person_1.MP4
    0:03 ~ 01:55 (collaborative_task)
    02:38 ~ End (collaborative_task)
    vid_153__day_5__con_7__person_2.MP4
    0:03 ~ 01:55 (collaborative_task)
    03:14 ~ 08:24 (collaborative_task)
    08:27 ~ End (collaborative_task)
    vid_154__day_5__con_7__person_3.MP4
    0:05 ~ 01:23 (collaborative_task)
    02:00 ~ 02:16 (collaborative_task)
    03:24 ~ 08:25 (collaborative_task)
    08:28 ~ End (collaborative_task)
    vid_155__day_5__con_8__person_1.MP4
    Start ~ 02:50 (collaborative_task)
    03:10 ~ End (collaborative_task)
    vid_156__day_5__con_8__person_2.MP4
    Start ~ 0:30 (collaborative_task)
    01:43 ~ 02:40 (collaborative_task)
    03:05 ~ 17:42 (collaborative_task)
    vid_157__day_5__con_8__person_3.MP4
    Start ~ 02:06 (collaborative_task)
    02:20 ~ End (collaborative_task)
    vid_158__day_6__con_1__person_1.MP4
    11:29 ~ 13:35 (social_interaction)
    20:43 ~ 21:30 (object_manipulation)
    vid_159__day_6__con_1__person_2.MP4
    00:24 ~ 01:25 (object_manipulation)
    01:35 ~ 01:45 (object_manipulation)
    01:58 ~ 02:02 (object_manipulation)
    02:08 ~ 02:35 (object_manipulation)
    02:41 ~ 03:01 (object_manipulation)
    03:07 ~ 03:26 (object_manipulation)
    05:12 ~ 05:50 (object_manipulation)
    05:58 ~ 07:16 (object_manipulation)
    08:35 ~ 10:28 (object_manipulation)
    10:31 ~ 11:07 (object_manipulation)
    11:29 ~ 13:35 (social_interaction)
    13:55 ~ 15:40 (object_manipulation)
    15:54 ~ 16:13 (object_manipulation)
    17:00 ~ 18:25 (object_manipulation)
    19:00 ~ 19:17 (object_manipulation)
    19:30 ~ 21:39 (object_manipulation)
    21:57 ~ End (object_manipulation)
    vid_160__day_6__con_1__person_3.MP4
    00:24 ~ 01:45 (object_manipulation)
    01:54 ~ 07:10 (object_manipulation)
    07:55 ~ 08:05 (object_manipulation)
    08:18 ~ 08:45 (object_manipulation)
    9:50 ~ 10:06 (object_manipulation)
    11:29 ~ 13:35 (social_interaction)
    13:53 ~ 14:07 (object_manipulation)
    14:15 ~ 14:42 (object_manipulation)
    14:47 ~ 15:02 (object_manipulation)
    15:12 ~ 15:22 (object_manipulation)
    17:04 ~ 18:05 (object_manipulation)
    19:07 ~ 19:16 (object_manipulation)
    19:38 ~ 20:34 (object_manipulation)
    21:54 ~ End (object_manipulation)
    vid_161__day_6__con_2__person_1.MP4
    0:05 ~ 03:03 (collaborative_task)
    03:23 ~ 06:36 (collaborative_task)
    07:13 ~ End (collaborative_task)
    vid_162__day_6__con_2__person_2.MP4
    0:05 ~ 02:36 (collaborative_task)
    03:14 ~ 03:44 (collaborative_task)
    03:54 ~ End (collaborative_task)
    vid_163__day_6__con_2__person_3.MP4
    0:10 ~ 02:40 (collaborative_task)
    04:00 ~ 06:39 (collaborative_task)
    06:48 ~ End (collaborative_task)
    vid_164__day_6__con_3__person_1.MP4
    0:07 ~ 02:30 (collaborative_task)
    02:31 ~ 03:45 (collaborative_task)
    04:11 ~ End (collaborative_task)
    vid_165__day_6__con_3__person_2.MP4
    0:14 ~ 02:27 (collaborative_task)
    02:44 ~ 03:11 (collaborative_task)
    03:44 ~ End (collaborative_task)
    vid_166__day_6__con_3__person_3.MP4
    0:08 ~ 03:25 (collaborative_task)
    03:48 ~ End (collaborative_task)
    vid_167__day_6__con_4__person_1.MP4
    01:11 ~ 01:30 (collaborative_task)
    01:56 ~ End (collaborative_task)
    vid_168__day_6__con_4__person_2.MP4
    Start ~ 01:11 (collaborative_task)
    02:00 ~ End (collaborative_task)
    vid_169__day_6__con_4__person_3.MP4
    Start ~ 01:11 (collaborative_task)
    02:08 ~ 17:23 (collaborative_task)
    vid_170__day_6__con_5__person_1.MP4
    Start ~ 01:26 (collaborative_task)
    02:05 ~ 02:25 (collaborative_task)
    02:30 ~ 12:17 (collaborative_task)
    12:23 ~ End (collaborative_task)
    vid_171__day_6__con_5__person_2.MP4
    Start ~ 01:58 (collaborative_task)
    02:02 ~ 02:28 (collaborative_task)
    02:40 ~ 12:17 (collaborative_task)
    12:26 ~ 20:40 (collaborative_task)
    20:52 ~ End (collaborative_task)
    vid_172__day_6__con_5__person_3.MP4
    Start ~ 0:28 (collaborative_task)
    0:36 ~ 01:30 (collaborative_task)
    01:39 ~ 01:55 (collaborative_task)
    02:19 ~ 21:14 (collaborative_task)
    vid_173__day_6__con_6__person_1.MP4
    Start ~ 0:23 (collaborative_task)
    0:35 ~ 01:46 (collaborative_task)
    01:55 ~ End (collaborative_task)
    vid_174__day_6__con_6__person_2.MP4
    Start ~ 0:23 (collaborative_task)
    0:34 ~ 01:02 (collaborative_task)
    01:10 ~ 02:05 (collaborative_task)
    02:19 ~ End (collaborative_task)
    vid_175__day_6__con_6__person_3.MP4
    Start ~ 0:30 (collaborative_task)
    0:36 ~ 01:16 (collaborative_task)
    01:56 ~ 02:13 (collaborative_task)
    02:30 ~ End (collaborative_task)
    """

    # Parse the raw data into the required dictionary format
    video_processing_data = parse_processing_data(raw_processing_data_20min)

    # Run the splitting process
    split_videos(video_path, video_processing_data, output_path)