import cv2
import pandas as pd
from ultralytics import YOLO
import torch
import argparse
import os

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
    
    # Track previous frame detections for IoU matching
    prev_detections = []
    next_track_id = 1
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame, classes=0, conf=0.5, stream=True, verbose=False)
        
        current_detections = []
        
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                # Sort by confidence and take top 3
                boxes_with_conf = [(box, float(box.conf.item())) for box in result.boxes]
                boxes_with_conf.sort(key=lambda x: x[1], reverse=True)
                
                for box, confidence in boxes_with_conf[:3]: # EGOCOM only needs 3 detections.
                    class_id = int(box.cls.item())
                    class_name = model.names[class_id]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    current_box = (x1, y1, x2, y2)
                    
                    # Find best match with previous frame using IoU
                    best_match_id = None
                    best_iou = 0
                    iou_threshold = 0.3  # Adjust this threshold as needed
                    
                    for prev_det in prev_detections:
                        prev_box = (prev_det['x1'], prev_det['y1'], prev_det['x2'], prev_det['y2'])
                        iou = calculate_iou(current_box, prev_box)
                        if iou > best_iou and iou > iou_threshold:
                            best_iou = iou
                            best_match_id = prev_det['track_id']
                    
                    # Assign track ID
                    if best_match_id is not None:
                        track_id = best_match_id
                    else:
                        track_id = next_track_id
                        next_track_id += 1
                    
                    person_label = f"{class_name}_id_{track_id}"
                    
                    current_detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'track_id': track_id,
                        'confidence': confidence
                    })
                    
                    detection_results.append({
                        'frame': frame_number,
                        'class_name': person_label,
                        'confidence': f"{confidence:.2f}",
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                    })
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{person_label} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Update previous detections for next frame
        prev_detections = current_detections 
        
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