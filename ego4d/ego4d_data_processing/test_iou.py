#!/usr/bin/env python3

def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    print(f"Box1: [{x1_1}, {y1_1}, {x2_1}, {y2_1}]")
    print(f"Box2: [{x1_2}, {y1_2}, {x2_2}, {y2_2}]")
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    print(f"Intersection bounds: [{x1_i}, {y1_i}, {x2_i}, {y2_i}]")
    
    if x2_i <= x1_i or y2_i <= y1_i:
        print("No intersection")
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    print(f"Intersection: {intersection}")
    print(f"Area1: {area1}, Area2: {area2}")
    print(f"Union: {union}")
    
    iou = intersection / union if union > 0 else 0.0
    print(f"IoU: {iou}")
    return iou

# Test with actual coordinates from frame 498
gt_box = [2.0, 131.1, 412.5, 587.2]
detected_box = [37, 426, 152, 573]

print("=== Frame 498 IoU Calculation ===")
iou = calculate_iou(gt_box, detected_box)

print(f"\nFinal IoU: {iou:.4f}")
print(f"Would match with threshold 0.3? {iou >= 0.3}")
print(f"Would match with threshold 0.1? {iou >= 0.1}")
print(f"Would match with threshold 0.05? {iou >= 0.05}")
