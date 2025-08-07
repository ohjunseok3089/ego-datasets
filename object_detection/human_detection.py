import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import argparse
import os
import numpy as np
from strongsort import StrongSORT

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1, y1, x2, y2 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    
    inter_area = (xi2 - xi1) * (yi2 - yi1)
    
    # Calculate union area
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

def process_video_with_yolo(video_path, output_video_path, output_csv_path):
    model = YOLO("yolo12x.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    # Initialize StrongSORT tracker
    tracker = StrongSORT(model_weights='osnet_x1_0', device=device, fp16=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    detection_results = []
    frame_number = 0
    
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=0, conf=0.5, verbose=False)
        
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id == 0:  # Only person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf.item())
                        detections.append([x1, y1, x2, y2, confidence])
        
        # Update tracker with detections
        if detections:
            detections = np.array(detections)
            tracks = tracker.update(detections, frame)
            
            # Sort tracks by confidence and take top 3
            tracks_with_conf = [(track, track[4]) for track in tracks]
            tracks_with_conf.sort(key=lambda x: x[1], reverse=True)
            
            for track, confidence in tracks_with_conf[:3]:  # EGOCOM only needs 3 detections
                x1, y1, x2, y2, conf, track_id = track
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                
                class_name = model.names[0]  # person
                person_label = f"{class_name}_id_{track_id}"
                
                detection_results.append({
                    'frame': frame_number,
                    'class_name': person_label,
                    'confidence': f"{confidence:.2f}",
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{person_label} {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Update tracker with empty detections to handle disappearances
            tracker.update(np.empty((0, 5)), frame) 
        
        out.write(frame)
        
        frame_number += 1
        if frame_number % 100 == 0:
            print(f"Processed {frame_number} frames")
            
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    if detection_results:
        df = pd.DataFrame(detection_results)
        df.to_csv(output_csv_path, index=False)
        print(f"Detection results saved to {output_csv_path}")
    else:
        print("No detections found in the video")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Object Detection")
    parser.add_argument('--input', '-i', type=str, required=True, help="Input video file path")
    parser.add_argument('--output', '-o', type=str, required=True, help="Output directory path")
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output
    
    os.makedirs(output_dir, exist_ok=True)
    video_filename_base = os.path.splitext(os.path.basename(input_path))[0]
    output_video_path = os.path.join(output_dir, f"{video_filename_base}_detected.mp4")
    output_csv_path = os.path.join(output_dir, f"{video_filename_base}_detections.csv")
    
    process_video_with_yolo(input_path, output_video_path, output_csv_path)