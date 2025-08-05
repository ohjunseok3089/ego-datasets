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
# FROZEN_FRAMES = 7 # This constant is no longer needed.

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
    # Mask processing logic remains unchanged
    # if args.mask_path is not None:
    #     ...
    # else:
    #     segm_mask = None
        
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
            # segm_mask=torch.from_numpy(segm_mask)[None, None],
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
            # Reinitialize the model's online processing state properly
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
            # Prepare tensor for the initial visualization
            video_tensor = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
                0, 3, 1, 2
            )[None]

            vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
            output_filename = f"{seq_name}_{start_frame}_{end_frame}"
            print(f"Saving initial visualization to {output_filename}.mp4")
            vis.visualize(
                video_tensor, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=output_filename
            )
            
            # --- RED CIRCLE DETECTION & FROZEN FRAME TRIMMING ---
            # First detect and trim frozen frames based on red circle movement
            frames_to_trim_start = 0
            if len(window_frames) > 1:
                first_red_circle = detect_red_circle(window_frames[0])
                if first_red_circle is not None:
                    first_red_pos = (float(first_red_circle[0]), float(first_red_circle[1]))
                    # Check for stationary red circle at the beginning
                    for i in range(1, len(window_frames)):
                        current_red_circle = detect_red_circle(window_frames[i])
                        if current_red_circle is not None:
                            current_red_pos = (float(current_red_circle[0]), float(current_red_circle[1]))
                            # If red circle moved significantly, stop trimming
                            if abs(current_red_pos[0] - first_red_pos[0]) > 2.0 or abs(current_red_pos[1] - first_red_pos[1]) > 2.0:
                                frames_to_trim_start = i - 1 if i > 1 else 0
                                break
                        else:
                            # Red circle disappeared, stop trimming
                            frames_to_trim_start = i - 1 if i > 1 else 0
                            break
                    else:
                        # All frames have stationary red circle, keep only last frame
                        frames_to_trim_start = len(window_frames) - 1
            
            # Apply frozen frame trimming
            if frames_to_trim_start > 0:
                print(f"Trimming {frames_to_trim_start} frozen frames from start based on red circle position.")
                window_frames = window_frames[frames_to_trim_start:]
                pred_tracks = pred_tracks[:, frames_to_trim_start:]
                pred_visibility = pred_visibility[:, frames_to_trim_start:]
                start_frame += frames_to_trim_start
            
            # Re-create video tensor after trimming
            if len(window_frames) > 0:
                video_tensor = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(0, 3, 1, 2)[None]
                
                # Re-visualize with trimmed data
                vis.visualize(
                    video_tensor, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=output_filename
                )
            
            # Now check for red circle disappearance for final truncation
            cap = cv2.VideoCapture(os.path.join(save_dir, output_filename + ".mp4"))
            actual_end_frame = end_frame
            frame_idx_in_vid = 0
            red_circle_disappeared = False
            while True:
                ret, cv_frame = cap.read()
                if not ret:
                    break
                if detect_red_circle(cv_frame) is None:
                    print(f"Red circle not detected in frame {frame_idx_in_vid} of the visualized video. Truncating.")
                    frames_to_keep = frame_idx_in_vid
                    actual_end_frame = start_frame + frames_to_keep
                    red_circle_disappeared = True
                    break
                frame_idx_in_vid += 1
            cap.release()
            
            if red_circle_disappeared:
                # Remove the initial (potentially longer) video file
                os.remove(os.path.join(save_dir, output_filename + ".mp4"))
                print(f"Removed intermediate file: {output_filename}.mp4")

                if frames_to_keep > 0:
                    # Create the final, correctly truncated video from the *original* segment data
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
