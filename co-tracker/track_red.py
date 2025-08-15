import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any

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


def calculate_head_movement_with_fisheye_correction(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
    image_width: int,
    image_height: int,
    fov_degrees: float = 110.0,
    distortion_model: str = "equidistant"
) -> Optional[Dict[str, Any]]:
    """
    Calculate head movement with proper fisheye lens distortion correction for Aria cameras.
    """
    
    # Convert pixel coordinates to normalized coordinates [-1, 1]
    def pixel_to_normalized(x: float, y: float) -> Tuple[float, float]:
        norm_x = (2.0 * x / image_width) - 1.0
        norm_y = (2.0 * y / image_height) - 1.0
        return norm_x, norm_y
    
    # Convert normalized coordinates to angular coordinates using fisheye model
    def normalized_to_angular(norm_x: float, norm_y: float) -> Tuple[float, float]:
        # Distance from center
        r = np.sqrt(norm_x**2 + norm_y**2)
        
        # Maximum radius corresponds to half FOV
        max_radius = 1.0  # normalized coordinate system goes from -1 to 1
        half_fov_rad = np.deg2rad(fov_degrees / 2.0)
        
        if r > max_radius:
            # Clamp to avoid extrapolation errors
            r = max_radius
            norm_x = norm_x * (max_radius / np.sqrt(norm_x**2 + norm_y**2))
            norm_y = norm_y * (max_radius / np.sqrt(norm_x**2 + norm_y**2))
        
        if r < 1e-8:  # Avoid division by zero at center
            return 0.0, 0.0
        
        # Apply inverse fisheye projection (equidistant model for Aria)
        # r = f * θ (linear relationship between radius and angle)
        theta = r * half_fov_rad
        
        # Convert back to Cartesian angular coordinates
        phi = np.arctan2(norm_y, norm_x)  # azimuth angle
        
        # Convert spherical to Cartesian angular coordinates
        angular_x = theta * np.cos(phi)  # horizontal angle
        angular_y = theta * np.sin(phi)  # vertical angle
        
        return angular_x, angular_y
    
    # Convert both positions to angular coordinates
    prev_norm_x, prev_norm_y = pixel_to_normalized(prev_pos[0], prev_pos[1])
    curr_norm_x, curr_norm_y = pixel_to_normalized(curr_pos[0], curr_pos[1])
    
    prev_ang_x, prev_ang_y = normalized_to_angular(prev_norm_x, prev_norm_y)
    curr_ang_x, curr_ang_y = normalized_to_angular(curr_norm_x, curr_norm_y)
    
    # Calculate angular differences (head movement)
    # Note: For head movement, the direction is inverted
    # (if object moves right in image, head moved left)
    delta_horizontal = -(curr_ang_x - prev_ang_x)  # yaw (left/right)
    delta_vertical = -(curr_ang_y - prev_ang_y)    # pitch (up/down)
    
    return {
        "horizontal": {
            "radians": float(delta_horizontal),
            "degrees": float(np.rad2deg(delta_horizontal))
        },
        "vertical": {
            "radians": float(delta_vertical),
            "degrees": float(np.rad2deg(delta_vertical))
        }
    }

def calculate_head_movement(prev_red_pos, curr_red_pos, image_width, image_height, video_fov_degrees=104.0):
    """
    Calculate head movement with automatic Aria detection and fisheye correction.
    """
    if prev_red_pos is None or curr_red_pos is None:
        return None
    
    # Check if this is Aria RGB camera (1408x1408, 110° FOV)
    is_aria_rgb = (image_width == 1408 and image_height == 1408)
    
    if is_aria_rgb:
        # Use fisheye correction for Aria RGB camera
        print("Detected Aria RGB camera - using fisheye correction")
        return calculate_head_movement_with_fisheye_correction(
            prev_red_pos, curr_red_pos, image_width, image_height, fov_degrees=110.0
        )
    else:
        # Use linear method for other cameras
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

def remap_position_from_movement_fisheye(start_pos, head_movement, image_width, image_height, fov_degrees=110.0):
    """
    Remap position using fisheye correction for Aria cameras.
    """
    if start_pos is None or head_movement is None:
        return None
    
    if np.isnan(head_movement['horizontal']['radians']) or np.isnan(head_movement['vertical']['radians']):
        return None
    
    # Convert starting position to normalized coordinates
    start_norm_x = (2.0 * start_pos[0] / image_width) - 1.0
    start_norm_y = (2.0 * start_pos[1] / image_height) - 1.0
    
    # Convert to angular coordinates
    start_r = np.sqrt(start_norm_x**2 + start_norm_y**2)
    half_fov_rad = np.deg2rad(fov_degrees / 2.0)
    
    if start_r < 1e-8:
        start_ang_x, start_ang_y = 0.0, 0.0
    else:
        start_theta = start_r * half_fov_rad
        start_phi = np.arctan2(start_norm_y, start_norm_x)
        start_ang_x = start_theta * np.cos(start_phi)
        start_ang_y = start_theta * np.sin(start_phi)
    
    # Apply head movement (note: movement is inverted, so we add it back)
    new_ang_x = start_ang_x - head_movement['horizontal']['radians']
    new_ang_y = start_ang_y - head_movement['vertical']['radians']
    
    # Convert back to pixel coordinates
    new_theta = np.sqrt(new_ang_x**2 + new_ang_y**2)
    if new_theta < 1e-8:
        new_norm_x, new_norm_y = 0.0, 0.0
    else:
        new_phi = np.arctan2(new_ang_y, new_ang_x)
        new_r = new_theta / half_fov_rad
        new_norm_x = new_r * np.cos(new_phi)
        new_norm_y = new_r * np.sin(new_phi)
    
    # Convert normalized coordinates back to pixels
    predicted_x = (new_norm_x + 1.0) * image_width / 2.0
    predicted_y = (new_norm_y + 1.0) * image_height / 2.0
    
    return (predicted_x, predicted_y)

def remap_position_from_movement(start_pos, head_movement, image_width, image_height, video_fov_degrees=104.0):
    """
    Remap position from head movement with automatic Aria detection.
    """
    if start_pos is None or head_movement is None or np.isnan(head_movement['horizontal']['radians']):
        return None
    
    # Check if this is Aria RGB camera
    is_aria_rgb = (image_width == 1408 and image_height == 1408)
    
    if is_aria_rgb:
        # Use fisheye correction for Aria RGB camera
        return remap_position_from_movement_fisheye(
            start_pos, head_movement, image_width, image_height, fov_degrees=110.0
        )
    else:
        # Use linear method for other cameras
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

# Test function to compare methods
def test_aria_vs_linear():
    """
    Test function to show the difference between linear and fisheye methods.
    """
    print("Testing Aria RGB vs Linear methods:")
    print("=" * 50)
    
    # Aria RGB specs
    aria_w, aria_h = 1408, 1408
    
    # Test movements at different positions
    test_cases = [
        ((704, 704), (754, 704), "Center - 50px right"),
        ((200, 704), (250, 704), "Left edge - 50px right"),
        ((1200, 704), (1250, 704), "Right edge - 50px right"),
        ((704, 704), (804, 704), "Center - 100px right"),
        ((1000, 1000), (1100, 1100), "Near corner - diagonal"),
    ]
    
    for prev_pos, curr_pos, desc in test_cases:
        # Linear method
        linear_result = calculate_head_movement(prev_pos, curr_pos, 640, 480, 104.0)  # Force linear
        
        # Fisheye method (force Aria detection)
        fisheye_result = calculate_head_movement(prev_pos, curr_pos, aria_w, aria_h, 104.0)
        
        print(f"\n{desc}:")
        if linear_result:
            print(f"  Linear:   {linear_result['horizontal']['degrees']:6.2f}° H, {linear_result['vertical']['degrees']:6.2f}° V")
        if fisheye_result:
            print(f"  Fisheye:  {fisheye_result['horizontal']['degrees']:6.2f}° H, {fisheye_result['vertical']['degrees']:6.2f}° V")

if __name__ == "__main__":
    test_aria_vs_linear()