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
# Original GT: frame_number,person_id,x1,y1,x2,y2,confidence -> 498,1,1.99,131.06,412.5,587.2,1
original_gt = [2.0, 131.1, 412.5, 587.2]  # [x1, y1, x2, y2] 
detected_box = [37, 426, 152, 573]        # [x1, y1, x2, y2]

print("=== Frame 498 IoU Calculation ===")
print("Original interpretation:")
iou1 = calculate_iou(original_gt, detected_box)

print(f"\nFinal IoU (original): {iou1:.4f}")

# Try different coordinate order interpretations
print("\n" + "="*50)
print("Trying different GT coordinate orders:")

# Maybe GT is [y1, x1, y2, x2]?
gt_swapped1 = [131.1, 2.0, 587.2, 412.5]  # [y1, x1, y2, x2]
print(f"\nGT as [y1, x1, y2, x2]: {gt_swapped1}")
iou2 = calculate_iou(gt_swapped1, detected_box)
print(f"IoU: {iou2:.4f}")

# Maybe GT is [x1, x2, y1, y2]?
gt_swapped2 = [2.0, 412.5, 131.1, 587.2]  # [x1, x2, y1, y2] 
print(f"\nGT as [x1, x2, y1, y2]: {gt_swapped2}")
iou3 = calculate_iou(gt_swapped2, detected_box)
print(f"IoU: {iou3:.4f}")

# Maybe GT is [y1, y2, x1, x2]?
gt_swapped3 = [131.1, 587.2, 2.0, 412.5]  # [y1, y2, x1, x2]
print(f"\nGT as [y1, y2, x1, x2]: {gt_swapped3}")
iou4 = calculate_iou(gt_swapped3, detected_box)
print(f"IoU: {iou4:.4f}")

print(f"\n" + "="*50)
print("SUMMARY:")
print(f"Original [x1, y1, x2, y2]: IoU = {iou1:.4f}")
print(f"Swapped  [y1, x1, y2, x2]: IoU = {iou2:.4f}")
print(f"Swapped  [x1, x2, y1, y2]: IoU = {iou3:.4f}")
print(f"Swapped  [y1, y2, x1, x2]: IoU = {iou4:.4f}")

best_iou = max(iou1, iou2, iou3, iou4)
print(f"\nBest IoU: {best_iou:.4f}")
print(f"Would match with threshold 0.3? {best_iou >= 0.3}")
print(f"Would match with threshold 0.1? {best_iou >= 0.1}")
