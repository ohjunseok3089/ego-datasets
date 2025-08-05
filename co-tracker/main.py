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
FROZEN_FRAMES = 7

def extract_video_info(video_path):
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
    # if args.mask_path is not None:
    #     segm_mask = np.array(Image.open(args.mask_path))
    #     print(f"Original segm_mask shape: {segm_mask.shape}")
    #     if segm_mask.ndim == 4:
    #         segm_mask_gray = segm_mask[..., 0, 0] if segm_mask.shape[3] == 1 else segm_mask[..., 0]
    #     elif segm_mask.ndim == 3 and segm_mask.shape[2] == 3:
    #         segm_mask_gray = cv2.cvtColor(segm_mask, cv2.COLOR_RGB2GRAY)
    #     elif segm_mask.ndim == 3 and segm_mask.shape[2] == 4:
    #         segm_mask_gray = segm_mask[..., 0]
    #     else:
    #         segm_mask_gray = segm_mask

    #     segm_mask_model = cv2.resize(segm_mask_gray, (512, 384))  
    #     print(f"Model input mask shape: {segm_mask_model.shape}")

    #     segm_mask = segm_mask_model
    #     print("Mask processed.")
    # else:
    #     segm_mask = None
    #     print("No mask provided.")
        
    def _process_step(window_frames, is_first_step, grid_size, grid_query_frame, queries):
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

    # Get the base name of the video file to use as the directory name
    seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
    
    # Construct the full save directory path as requested
    save_dir = f"{args.save_dir}/{seq_name}"
    
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output will be saved to directory: {save_dir}")
    
    # Calculate center coordinates for queries
    frame_height, frame_width = full_vid[0].shape[:2]
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    
    queries = torch.tensor([
        [[0., center_x, center_y]]
    ])
    if torch.cuda.is_available():
        queries = queries.cuda()
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Center coordinates: ({center_x}, {center_y})")
    
    start_frame = 0
    last_frame_from_previous_batch = None
    
    while start_frame < num_frames:
        print(f"Processing frames from {start_frame} to {min(start_frame + int(fps * FRAMES_INTERVAL), num_frames)}")
        video, end_frame = extract_frames(full_vid, FRAMES_INTERVAL, fps, start_frame, num_frames)
        
        # Skip if no frames to process
        if end_frame <= start_frame or len(video) == 0:
            print(f"No frames to process in segment {start_frame} to {end_frame}, skipping...")
            break
        
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
            print(f"Added last frame from previous batch for overlap (frame {start_frame - 1})")
        
        is_first_step = True
        
        for i, frame in enumerate(video):
            if i % model.step == 0 and i != 0:
                print(f"Calling _process_step at frame {i} (is_first_step={is_first_step})")
                pred_tracks, pred_visibility = _process_step(
                    window_frames,
                    is_first_step,
                    grid_size=args.grid_size,
                    grid_query_frame=args.grid_query_frame,
                    queries=queries,
                )
                print(f"_process_step completed at frame {i}")
                is_first_step = False
            window_frames.append(frame)
        
        if len(window_frames) > 0:
            print("Calling _process_step for final frames...")
            pred_tracks, pred_visibility = _process_step(
                window_frames,
                is_first_step,
                grid_size=args.grid_size,
                grid_query_frame=args.grid_query_frame,
                queries=queries,
            )
            print("_process_step for final frames completed.")

        print("Tracks are computed")
        # TODO : Delete frozen frame
        
        if pred_tracks is not None: 
            # save a video with predicted tracks
            print("Preparing video tensor for visualization...")
            video_tensor = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
                0, 3, 1, 2
            )[None]
            print("Saving video with predicted tracks...")
            vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
            output_filename = f"{seq_name}_{start_frame}_{end_frame}"
            vis.visualize(
                video_tensor, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=output_filename
            )
            # Check for red circle detection in the saved video and truncate if necessary
            cap = cv2.VideoCapture(os.path.join(save_dir, output_filename + ".mp4"))
            actual_end_frame = end_frame
            i = 0
            while True:
                ret, cv_frame = cap.read()
                if i < FROZEN_FRAMES:
                    i += 1
                    continue
                if not ret:
                    break
                red_circle = detect_red_circle(cv_frame)
                if red_circle is None:
                    print(f"Red circle not detected in frame {start_frame + i}. Truncating video backward.")
                    actual_end_frame = start_frame + i - FROZEN_FRAMES
                    
                    frames_to_keep = actual_end_frame - start_frame
                    if frames_to_keep > 0:
                        truncated_window_frames = window_frames[:frames_to_keep]
                        print(f"Truncated window_frames to {frames_to_keep} frames (up to frame {actual_end_frame})")
                        
                        truncated_video_tensor = torch.tensor(np.stack(truncated_window_frames), device=DEFAULT_DEVICE).permute(
                            0, 3, 1, 2
                        )[None]
                        new_output_filename = f"{seq_name}_{start_frame}_{actual_end_frame}.mp4"
                        vis.visualize(
                            truncated_video_tensor, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=new_output_filename
                        )
                        print(f"Truncated video saved to {os.path.join(save_dir, new_output_filename)}")
                        
                        os.remove(os.path.join(save_dir, output_filename + ".mp4"))
                        print(f"Removed original file: {output_filename}")
                    break
                i += 1
            cap.release()
            
            if len(window_frames) > 0:
                last_frame_from_previous_batch = window_frames[-1]
            
            start_frame = actual_end_frame
        else:
            print("No tracks were predicted for this segment, skipping visualization.")
            if len(window_frames) > 0:
                last_frame_from_previous_batch = window_frames[-1]
            
            start_frame = end_frame
        print(f"Processed frames from {start_frame} to {end_frame}")

    print(f"Processed all frames from 0 to {num_frames}")