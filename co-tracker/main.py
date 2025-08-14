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

FRAMES_INTERVAL = 15
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

def load_video_chunk(video_path, start_frame, chunk_size, total_frames):
    """Load a small chunk of video frames to manage memory usage"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Failed to open video for chunk loading")
    
    # Skip to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    end_frame = min(start_frame + chunk_size, total_frames)
    consecutive_fails = 0
    max_consecutive_fails = 10  # Allow some failed frames but not too many
    
    for i in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            consecutive_fails += 1
            if consecutive_fails > max_consecutive_fails:
                print(f"Too many consecutive failed frame reads ({consecutive_fails}), stopping chunk at frame {i}")
                break
            continue
        
        if frame is None or frame.size == 0:
            consecutive_fails += 1
            if consecutive_fails > max_consecutive_fails:
                print(f"Too many consecutive empty frames ({consecutive_fails}), stopping chunk at frame {i}")
                break
            continue
            
        consecutive_fails = 0  # Reset counter on successful read
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    
    if len(frames) == 0:
        return None, start_frame
    
    return np.stack(frames), start_frame + len(frames)

def extract_frames(video, frames_to_extract, start_frame, num_frames):
    end_frame = start_frame + frames_to_extract + 1
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

    # Get the base name of the video file to check if already processed
    seq_name = os.path.splitext(os.path.basename(args.video_path))[0]
    save_dir = f"{args.save_dir}/{seq_name}"
    
    # Check if this video has already been fully processed
    if os.path.exists(save_dir):
        existing_files = [f for f in os.listdir(save_dir) if f.endswith('.mp4')]
        if len(existing_files) > 0:
            # Extract frame ranges from existing files to check completion
            processed_frames = []
            for file in existing_files:
                # Extract start_end pattern from filename
                parts = file.replace('.mp4', '').split('_')
                if len(parts) >= 3:
                    try:
                        start_frame = int(parts[-2])
                        end_frame = int(parts[-1])
                        processed_frames.extend(range(start_frame, end_frame))
                    except ValueError:
                        continue
            
            # Quick check of total video length to see if processing is complete
            try:
                cap = cv2.VideoCapture(args.video_path)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()
                    
                    # If we have processed at least 90% of frames, consider it complete
                    coverage = len(set(processed_frames)) / total_frames if total_frames > 0 else 0
                    
                    if coverage >= 0.9:
                        print(f"Video {seq_name} already processed with {len(existing_files)} output files ({coverage:.1%} coverage)")
                        print("Skipping this video. Delete the output directory to reprocess.")
                        exit(0)
                    else:
                        print(f"Video {seq_name} partially processed ({coverage:.1%} coverage), restarting from beginning...")
                        # Remove incomplete processing results
                        import shutil
                        shutil.rmtree(save_dir)
                else:
                    print(f"Cannot verify completion for {seq_name}, reprocessing...")
                    import shutil 
                    shutil.rmtree(save_dir)
            except Exception as e:
                print(f"Error checking completion for {seq_name}: {e}, reprocessing...")
                import shutil
                shutil.rmtree(save_dir)

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


    # Get video info without loading all frames
    try:
        print("Getting video information...")
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Validate video properties
        if fps <= 0 or num_frames <= 0 or frame_width <= 0 or frame_height <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties: fps={fps}, frames={num_frames}, size={frame_width}x{frame_height}")
        
        # Try to read first frame to verify video is actually readable
        ret, test_frame = cap.read()
        cap.release()
        
        if not ret or test_frame is None:
            raise ValueError("Cannot read first frame - video may be corrupted")
        
        print(f"Video info: {num_frames} frames at {fps} FPS, resolution: {frame_width}x{frame_height}")
        
    except Exception as e:
        print(f"Failed to get video information with OpenCV: {e}")
        print("Trying with imageio as fallback...")
        try:
            fps, num_frames = extract_video_info(args.video_path)
            # Get frame dimensions from first frame
            test_chunk, _ = load_video_chunk(args.video_path, 0, 1, num_frames)
            if test_chunk is not None:
                frame_height, frame_width = test_chunk[0].shape[:2]
                print(f"Video info (imageio): {num_frames} frames at {fps} FPS, resolution: {frame_width}x{frame_height}")
            else:
                raise ValueError("Failed to load test frame")
        except Exception as e2:
            print(f"Both OpenCV and imageio failed: {e2}")
            print("This video appears to be corrupted or unreadable.")
            print("Exiting with error code 1 to indicate video corruption.")
            exit(1)

    # Create the directory if it doesn't exist (seq_name and save_dir already defined above)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output will be saved to directory: {save_dir}")
    
    # Calculate center coordinates for queries
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    
    queries = torch.tensor([
        [[0., center_x, center_y]]
    ])
    if torch.cuda.is_available():
        queries = queries.cuda()
    print(f"Video dimensions: {frame_width}x{frame_height}")
    print(f"Center coordinates: ({center_x}, {center_y})")
    
    # Process video in small chunks to manage memory usage
    chunk_start = 0
    
    while chunk_start < num_frames:
        # Load small chunk of video (FRAMES_INTERVAL frames)
        chunk_end = min(chunk_start + FRAMES_INTERVAL, num_frames)
        print(f"Loading video chunk: frames {chunk_start} to {chunk_end-1}")
        
        try:
            video_chunk, actual_end = load_video_chunk(args.video_path, chunk_start, FRAMES_INTERVAL, num_frames)
            if video_chunk is None:
                print(f"Failed to load chunk starting at frame {chunk_start}, skipping to next chunk")
                chunk_start += FRAMES_INTERVAL
                continue
                
            print(f"Successfully loaded chunk: {len(video_chunk)} frames")
            
        except Exception as e:
            print(f"Error loading chunk at frame {chunk_start}: {e}")
            print("Trying to skip ahead to avoid corrupted section...")
            chunk_start += FRAMES_INTERVAL
            continue
        
        # Process this chunk 
        actual_start_frame = chunk_start
        end_frame = chunk_start + len(video_chunk)
        print(f"Processing chunk: frames {actual_start_frame} to {end_frame-1}")
        
        video = video_chunk
        
        # Skip if no frames to process
        if end_frame <= actual_start_frame or len(video) == 0:
            print(f"No frames to process in segment {actual_start_frame} to {end_frame}, skipping...")
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
        
        if pred_tracks is not None: 
            # save a video with predicted tracks
            print("Preparing video tensor for visualization...")
            video_tensor = torch.tensor(np.stack(window_frames), device=DEFAULT_DEVICE).permute(
                0, 3, 1, 2
            )[None]
            print("Saving video with predicted tracks...")
            vis = Visualizer(save_dir=save_dir, pad_value=120, linewidth=3)
            output_filename = f"{seq_name}_{actual_start_frame}_{end_frame - 2}"
            vis.visualize(
                video_tensor, pred_tracks, pred_visibility, query_frame=args.grid_query_frame, filename=output_filename
            )
            # Post-process the saved video in-place without re-visualizing:
            output_path = os.path.join(save_dir, output_filename + ".mp4")
            cap = cv2.VideoCapture(output_path)
            out_fps = cap.get(cv2.CAP_PROP_FPS)
            if out_fps is None or out_fps <= 0:
                out_fps = fps
            frames = []
            i = 0
            removed_original_due_to_no_red = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if i < FROZEN_FRAMES:
                    i += 1
                    continue
                red_circle = detect_red_circle(frame)
                if red_circle is None:
                    print(f"No red circle detected at frame {i}, removing video {output_filename}.mp4")
                    os.remove(output_path)
                    end_frame = actual_start_frame + i - FROZEN_FRAMES + 1
                    output_filename = f"{seq_name}_{actual_start_frame}_{end_frame - 2}"
                    output_path = os.path.join(save_dir, output_filename + ".mp4")
                    break
                frames.append(frame)
                i += 1
            cap.release()
            if len(frames) > 0:
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
                for f in frames:
                    writer.write(f)
                writer.release()
                print(f"Saved video {output_filename}.mp4")
        
        # Progress to next chunk with slight overlap
        chunk_start = actual_end - 1 if actual_end > chunk_start + 1 else actual_end
        print(f"Completed chunk, moving to next chunk starting at frame {chunk_start}")
    
    print(f"Processed all frames from 0 to {num_frames}")