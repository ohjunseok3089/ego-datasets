import os
import torch
import argparse
import imageio.v3 as iio
import numpy as np
from PIL import Image
import cv2
import imageio

from cotracker.utils.visualizer import Visualizer, read_video_from_path
from track_red import detect_red_circle
from cotracker.predictor import CoTrackerOnlinePredictor

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)
FRAMES_INTERVAL = 0.5

def extract_video_info(video_path):
    """Extracts FPS and total number of frames from a video file."""
    try:
        reader = imageio.get_reader(video_path)
        fps = reader.get_meta_data()['fps']
        num_frames = reader.get_length()
        reader.close()
        return fps, num_frames
    except Exception as e:
        print(f"Error loading video {video_path}: {e}")
        raise ValueError(f"Failed to load video: {video_path}")

def extract_frames(video, seconds, fps, start_frame, num_frames):
    """Extracts a segment of frames from the full video."""
    frames_to_extract = int(fps * seconds)
    end_frame = start_frame + frames_to_extract
    if end_frame > num_frames:
        end_frame = num_frames
    video = video[start_frame:end_frame]
    return video, end_frame
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        help="path to a video",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="CoTracker model parameters",
    )
    parser.add_argument("--grid_size", type=int, default=10, help="Regular grid size")
    parser.add_argument(
        "--grid_query_frame",
        type=int,
        default=0,
        help="Compute dense and grid tracks starting from this frame",
    )
    parser.add_argument(
        "--mask_path",
        default=None,
        help="path to a mask",
    )
    parser.add_argument(
        "--save_dir",
        default="/mas/robots/prg-egocom/EGOCOM/720p/5min_parts/co-tracker",
        help="base directory to save output videos",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise ValueError("Video file does not exist")

    print("Loading model...")
    if args.checkpoint is not None:
        model = CoTrackerOnlinePredictor(checkpoint=args.checkpoint)
    else:
        model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    print("Model loaded.")

    print(f"Moving model to device: {DEFAULT_DEVICE}")
    model = model.to(DEFAULT_DEVICE)
    print("Model moved to device.")

    window_frames = []
        
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, queries):
        """Processes a chunk of video frames with the CoTracker model."""
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        result = model(
            video_chunk,
            is_first_step=is_first_step,
            grid_size=grid_size,
            grid_query_frame=grid_query_frame,
            queries=queries,
        )
        return result


    try:
        fps, num_frames = extract_video_info(args.video_path)
        full_vid = read_video_from_path(args.video_path)
        
        if full_vid is None or len(full_vid) == 0:
            raise ValueError("Failed to load video or video is empty")
    except Exception as e:
        print(f"Error processing video {args.video_path}: {e}")
        print("Skipping this video due to corruption or loading issues.")
        exit(1)

    seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_dir = f"{args.save_dir}/{seq_name}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output will be saved to directory: {save_dir}")
    
    frame_height, frame_width = full_vid[0].shape[:2]
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    
    queries = torch.tensor([
        [[0., center_x, center_y]]
    ], device=DEFAULT_DEVICE)
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Center coordinates: ({center_x}, {center_y})")
    
    start_frame = 0
    last_frame_from_previous_batch = None
    
    while start_frame < num_frames:
        print(f"Processing frames from {start_frame} to {min(start_frame + int(fps * FRAMES_INTERVAL), num_frames)}")
        video, end_frame = extract_frames(full_vid, FRAMES_INTERVAL, fps, start_frame, num_frames)
        
        if end_frame <= start_frame or len(video) == 0:
            print(f"No frames to process in segment {start_frame} to {end_frame}, skipping...")
            break
        
        # Reset model state for the new segment
        if hasattr(model, 'reset'):
            model.reset()
        else:
            if hasattr(model, 'model') and hasattr(model.model, 'init_video_online_processing'):
                model.model.init_video_online_processing()
            if hasattr(model, 'queries'):
                delattr(model, 'queries')
            if hasattr(model, 'N'):
                delattr(model, 'N')
            if hasattr(model, 'model') and hasattr(model.model, 'reset'):
                model.model.reset()
        print("Model state reset for this segment.")
        
        window_frames = []
        if last_frame_from_previous_batch is not None:
            window_frames.append(last_frame_from_previous_batch)
        
        is_first_step = True
        
        for i, frame in enumerate(video):
            window_frames.append(frame)
            if len(window_frames) % model.step == 0 and len(window_frames) >= model.step:
                print(f"Calling _process_step at frame index {i} (is_first_step={is_first_step})")
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                    queries=queries,
                )
                is_first_step = False
        
        if len(window_frames) > 0 and (is_first_step or len(window_frames) % model.step != 0):
            print("Calling _process_step for final frames...")
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                queries=queries,
            )

        print("Tracks are computed.")
        
        if pred_tracks is not None: 
            # --- REVISED LOGIC TO DYNAMICALLY REMOVE FROZEN FRAMES ---
            # `video` holds the new frames for this segment.
            # `window_frames` is `[optional_last_frame] + video`.

            # 1. Find the number of frozen frames at the start of the new segment (`video`).
            frozen_in_video = 0
            if len(video) > 1:
                first_frame_of_video_np = np.asarray(video[0])
                count = 1  # The first frame always matches itself.
                for i in range(1, len(video)):
                    if np.array_equal(first_frame_of_video_np, np.asarray(video[i])):
                        count += 1
                    else:
                        break
                # A "frozen sequence" must have at least 2 identical frames.
                if count > 1:
                    frozen_in_video = count
            
            # 2. Calculate the total number of frames to trim from the start of `window_frames`.
            num_to_trim = frozen_in_video
            if last_frame_from_previous_batch is not None:
                # For visualization, we also trim the carry-over frame to get a clean start.
                num_to_trim += 1

            if num_to_trim > 0:
                print(f"Found {num_to_trim} static frames at the beginning. Trimming for visualization.")
                trimmed_vis_frames = window_frames[num_to_trim:]
                trimmed_vis_tracks = pred_tracks[:, num_to_trim:]
                trimmed_vis_visibility = pred_visibility[:, num_to_trim:]
            else:
                print("No static starting frames detected.")
                trimmed_vis_frames = window_frames
                trimmed_vis_tracks = pred_tracks
                trimmed_vis_visibility = pred_visibility

            if not trimmed_vis_frames:
                print("Segment is empty after trimming frozen frames. Skipping visualization.")
                start_frame = end_frame
                continue

            # Prepare tensor for the initial visualization
            trimmed_video_tensor = torch.tensor(np.stack(trimmed_vis_frames), device=DEFAULT_DEVICE).permute(
                0, 3, 1, 2
            )[None]

            vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
            output_filename = f"{seq_name}_{start_frame}_{end_frame}"
            print(f"Saving initial visualization to {output_filename}.mp4")
            vis.visualize(
                trimmed_video_tensor, trimmed_vis_tracks, trimmed_vis_visibility, query_frame=args.grid_query_frame, filename=output_filename
            )
            
            # --- RED CIRCLE DETECTION & FINAL TRUNCATION ---
            cap = cv2.VideoCapture(os.path.join(save_dir, output_filename + ".mp4"))
            actual_end_frame = end_frame
            frame_idx_in_trimmed_vid = 0
            red_circle_disappeared = False
            while True:
                ret, cv_frame = cap.read()
                if not ret:
                    break
                if detect_red_circle(cv_frame) is None:
                    print(f"Red circle not detected in frame {frame_idx_in_trimmed_vid} of the visualized video. Truncating.")
                    frames_to_keep = num_to_trim + frame_idx_in_trimmed_vid
                    actual_end_frame = start_frame + frames_to_keep
                    red_circle_disappeared = True
                    break
                frame_idx_in_trimmed_vid += 1
            cap.release()
            
            if red_circle_disappeared:
                os.remove(os.path.join(save_dir, output_filename + ".mp4"))
                print(f"Removed intermediate file: {output_filename}.mp4")

                if frames_to_keep > 0:
                    final_truncated_frames = window_frames[:frames_to_keep]
                    final_truncated_tracks = pred_tracks[:, :frames_to_keep]
                    final_truncated_visibility = pred_visibility[:, :frames_to_keep]
                    
                    final_video_tensor = torch.tensor(np.stack(final_truncated_frames), device=DEFAULT_DEVICE).permute(
                        0, 3, 1, 2
                    )[None]
                    
                    new_output_filename = f"{seq_name}_{start_frame}_{actual_end_frame}"
                    print(f"Saving final truncated video to {new_output_filename}.mp4")
                    vis.visualize(
                        final_video_tensor, final_truncated_tracks, final_truncated_visibility, query_frame=args.grid_query_frame, filename=new_output_filename
                    )
                else:
                    print("Final truncated video has zero length. Not saving.")

            if len(window_frames) > 0:
                last_frame_from_previous_batch = window_frames[-1]
            
            start_frame = actual_end_frame
        else:
            print("No tracks were predicted for this segment, skipping visualization.")
            if len(window_frames) > 0:
                last_frame_from_previous_batch = window_frames[-1]
            
            start_frame = end_frame
        print(f"Finished segment. Next start frame will be: {start_frame}")

    print(f"Completed processing all frames.")