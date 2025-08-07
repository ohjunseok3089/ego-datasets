import cv2
import pandas as pd
import torch
import argparse
import os
import numpy as np

from ultralytics import YOLO
from strongsort import StrongSort

def process_video_with_yolo(video_path, output_video_path, output_csv_path):
    model = YOLO("yolo12x.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    # Initialize StrongSort tracker
    tracker = StrongSort()
    
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
        
        # Make detections like the example
        boxes = []
        scores = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    class_id = int(box.cls.item())
                    if class_id == 0:  # Only person class
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        w, h = x2 - x1, y2 - y1
                        confidence = float(box.conf.item())
                        boxes.append([x1, y1, w, h])  # [x, y, w, h] format like example
                        scores.append(confidence)
        
        # Update tracker with detections
        if boxes:
            boxes = np.array(boxes)
            scores = np.array(scores)
            tracks = tracker.update(boxes, scores, frame)
            
            # Sort tracks by confidence and take top 3
            if len(tracks) > 0:
                tracks_sorted = sorted(tracks, key=lambda x: x[4], reverse=True)
                
                for track in tracks_sorted[:3]:  # EGOCOM only needs 3 detections
                    x1, y1, x2, y2, track_id = track
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    track_id = int(track_id)
                    
                    # Find confidence for this track (approximate)
                    confidence = 0.8  # Default confidence since tracks may not preserve exact conf
                    
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
            tracker.update(np.empty((0, 4)), np.empty((0,)), frame) 
        
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