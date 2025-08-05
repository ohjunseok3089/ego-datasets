import cv2
import pandas as pd
import os
import numpy as np

def draw_boxes_on_video(csv_path, video_path, output_video_path):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: CSV file not found. Path: {csv_path}")
        return
    
    boxes_by_frame = df.groupby('frame_number').apply(lambda x: x.to_dict('records')).to_dict()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video file. Path: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Starting bounding box drawing... (This may take time depending on video length)")

    person_colors = {}
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx in boxes_by_frame:
            for box_data in boxes_by_frame[frame_idx]:
                person_id = box_data['person_id']
                x1, y1, x2, y2 = int(box_data['x1']), int(box_data['y1']), int(box_data['x2']), int(box_data['y2'])

                if person_id not in person_colors:
                    person_colors[person_id] = np.random.randint(0, 256, size=3).tolist()
                
                color = person_colors[person_id]
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                label = str(person_id)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        out.write(frame)
        frame_idx += 1

    print(f"Task completed! Result saved to '{output_video_path}'")
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    CLEANED_CSV_FILE_PATH = '/Volumes/T7 Shield Portable/Github/ego-datasets/egocom/egocom_data_processing/cleaned_csvs/vid_001__day_1__con_1__person_1_part1_global_gallery_cleaned.csv'

    ORIGINAL_VIDEO_PATH = '/Volumes/T7 Shield Portable/Github/EGOCOM/EGOCOM/720p/5min_parts/vid_001__day_1__con_1__person_1_part1.MP4'
    
    OUTPUT_VIDEO_PATH = "vid_001_with_boxes_output.mp4"

    draw_boxes_on_video(
        csv_path=CLEANED_CSV_FILE_PATH,
        video_path=ORIGINAL_VIDEO_PATH,
        output_video_path=OUTPUT_VIDEO_PATH
    )