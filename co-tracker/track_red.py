import cv2
import numpy as np
def detect_red_circle(image, target_radius: int = 3):
    if image is None:
        return None

    h, w = image.shape[:2]
    
    target_bgr = np.array([[[48, 28, 255]]], dtype=np.uint8)
    target_hsv = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0]
    target_h, target_s, target_v = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
    
    # Convert image to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Define tolerance ranges for the specific red color
    h_range = 10
    s_range = 30
    v_range = 40
    
    h_min = max(0, target_h - h_range)
    h_max = min(179, target_h + h_range)
    s_min = max(0, target_s - s_range)
    s_max = min(255, target_s + s_range)
    v_min = max(0, target_v - v_range)
    v_max = min(255, target_v + v_range)
    
    # Create mask for target red color
    lower_red = np.array([h_min, s_min, v_min], dtype=np.uint8)
    upper_red = np.array([h_max, s_max, v_max], dtype=np.uint8)
    red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
    
    # Handle red wrap-around in HSV
    if target_h < 10:
        lower_red2 = np.array([max(0, target_h + 170), s_min, v_min], dtype=np.uint8)
        upper_red2 = np.array([179, s_max, v_max], dtype=np.uint8)
        red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(red_mask, red_mask2)
    
    # Clean up the mask
    kernel = np.ones((3,3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5: continue
        perimeter = cv2.arcLength(contour, True) # gets contour's perimeter
        if perimeter == 0: continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        if 1 <= radius <= target_radius + 5:
            circles.append([int(x), int(y), int(radius)])

    if not circles:
        return None
    
    # Since there's only one circle, use it directly
    circle = circles[0]
    center_x = int(circle[0])
    center_y = int(circle[1])
    radius = int(circle[2])
    
    return (center_x, center_y, radius)

def calculate_head_movement(
    prev_red_pos,
    curr_red_pos,
    image_width,
    image_height,
    video_fov_degrees=104.0,
    method: str = "linear",
    center: tuple = None,
):
    if prev_red_pos is None or curr_red_pos is None:
        return None

    if method == "pinhole":
        # Use pinhole model: x = f * tan(theta). Delta angle = atan(x2/f) - atan(x1/f)
        if center is None:
            cx = image_width * 0.5
            cy = image_height * 0.5
        else:
            cx, cy = float(center[0]), float(center[1])
        hfov_rad = np.radians(float(video_fov_degrees))
        vfov_rad = hfov_rad  # assume symmetry for circular ROI
        fx = (image_width * 0.5) / np.tan(hfov_rad * 0.5)
        fy = (image_height * 0.5) / np.tan(vfov_rad * 0.5)

        x1 = float(prev_red_pos[0]) - cx
        x2 = float(curr_red_pos[0]) - cx
        y1 = float(prev_red_pos[1]) - cy
        y2 = float(curr_red_pos[1]) - cy

        yaw_rad = -(np.arctan2(x2, fx) - np.arctan2(x1, fx))
        pitch_rad = -(np.arctan2(y2, fy) - np.arctan2(y1, fy))

        return {
            "horizontal": {"radians": float(yaw_rad), "degrees": float(np.degrees(yaw_rad))},
            "vertical": {"radians": float(pitch_rad), "degrees": float(np.degrees(pitch_rad))},
        }

    # Fallback to previous linear-per-pixel mapping
    horizontal_pixel_change = curr_red_pos[0] - prev_red_pos[0]
    vertical_pixel_change = curr_red_pos[1] - prev_red_pos[1]

    aspect_ratio = image_width / image_height
    vertical_fov_degrees = video_fov_degrees / aspect_ratio

    horizontal_pixels_per_degree = image_width / video_fov_degrees
    horizontal_angle_degrees = -horizontal_pixel_change / horizontal_pixels_per_degree
    horizontal_radians = np.radians(horizontal_angle_degrees)

    vertical_pixels_per_degree = image_height / vertical_fov_degrees
    vertical_angle_degrees = -vertical_pixel_change / vertical_pixels_per_degree
    vertical_radians = np.radians(vertical_angle_degrees)

    return {
        "horizontal": {
            "radians": horizontal_radians,
            "degrees": horizontal_angle_degrees
        },
        "vertical": {
            "radians": vertical_radians,
            "degrees": vertical_angle_degrees
        }
    } 

def remap_position_from_movement(start_pos, head_movement, image_width, image_height, video_fov_degrees=104.0):
    if start_pos is None or head_movement is None or np.isnan(head_movement['horizontal']['radians']):
        return None
    
    horizontal_radians = head_movement['horizontal']['radians']
    vertical_radians = head_movement['vertical']['radians']

    horizontal_angle_degrees = np.degrees(horizontal_radians)
    vertical_angle_degrees = np.degrees(vertical_radians)

    aspect_ratio = image_width / image_height
    vertical_fov_degrees = video_fov_degrees / aspect_ratio
    horizontal_pixels_per_degree = image_width / video_fov_degrees
    vertical_pixels_per_degree = image_height / vertical_fov_degrees

    horizontal_pixel_change = -horizontal_angle_degrees * horizontal_pixels_per_degree
    vertical_pixel_change = -vertical_angle_degrees * vertical_pixels_per_degree

    predicted_x = start_pos[0] + horizontal_pixel_change
    predicted_y = start_pos[1] + vertical_pixel_change

    return (predicted_x, predicted_y)