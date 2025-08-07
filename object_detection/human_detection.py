import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import argparse
import os
import numpy as np
from strongsort.strong_sort import StrongSORT

def process_video_with_yolo(video_path, output_video_path, output_csv_path):
    model = YOLO("yolo12x.pt")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model.to(device)
    
    # Initialize StrongSORT tracker
    strong_sort = StrongSORT(
        model_weights='osnet_x0_25_msmt17.pt',
        device=device,
        fp16=True
    )
    
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
        
        # Get YOLO detections (detection only, no tracking)
        results = model(frame, classes=0, conf=0.5, verbose=False)
        
        # Prepare detections for StrongSORT
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                boxes_with_conf = [(box, float(box.conf.item())) for box in result.boxes]
                boxes_with_conf.sort(key=lambda x: x[1], reverse=True)
                
                for box, confidence in boxes_with_conf[:3]: # EGOCOM only needs 3 detections
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    # StrongSORT expects [x1, y1, x2, y2, conf, class_id]
                    detections.append([x1, y1, x2, y2, confidence, 0])  # class_id=0 for person
        
        # Convert to numpy array
        if detections:
            detections = np.array(detections)
            # Update StrongSORT tracker
            tracks = strong_sort.update(detections, frame)
            
            # Process tracked objects
            for track in tracks:
                x1, y1, x2, y2, track_id = map(int, track[:5])
                confidence = track[4] if len(track) > 4 else 0.0
                
                class_name = "person"
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
            # No detections, still need to update tracker
            tracks = strong_sort.update(np.empty((0, 6)), frame) 
        
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