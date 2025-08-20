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

def estimate_aria_intrinsics_from_fov(image_width: int, image_height: int, fov_degrees: float = 110.0) -> Dict[str, float]:
    """
    Estimate Aria RGB camera intrinsic parameters from FOV.
    
    Based on typical fisheye camera parameters and Aria's 110° FOV.
    Real Aria cameras have fx ≈ fy ≈ 700-900 pixels for 1408x1408 images.
    """
    # Principal point at center (this is standard)
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # For fisheye cameras, focal length estimation is more complex
    # but we can use empirical values based on typical Aria calibrations
    if image_width == 1408 and image_height == 1408:
        # Typical values for Aria RGB based on community reports
        # These values result in more reasonable angular calculations
        fx = fy = 800.0  # Empirical value that works better than FOV-based estimation
        print(f"Using empirical Aria RGB intrinsics: fx=fy={fx}")
    else:
        # Fallback to FOV-based estimation for other cameras
        half_fov_rad = np.deg2rad(fov_degrees / 2.0)
        fx = (image_width / 2.0) / half_fov_rad
        fy = (image_height / 2.0) / half_fov_rad
        print(f"Using FOV-based estimation: fx={fx:.1f}, fy={fy:.1f}")
    
    return {
        "fx": fx,
        "fy": fy, 
        "cx": cx,
        "cy": cy
    }

def calculate_head_movement_spherical_model(
    prev_pos: Tuple[float, float],
    curr_pos: Tuple[float, float],
    fx: float,
    fy: float,
    cx: float,
    cy: float
) -> Optional[Dict[str, Any]]:
    """
    Calculate head movement using Aria's spherical camera model.
    
    Aria spherical model:
    u = fx * θ * cos(φ) + cx
    v = fy * θ * sin(φ) + cy
    
    Where θ is the polar angle and φ is the azimuth angle.
    """
    
    def pixel_to_angles(u: float, v: float) -> Tuple[float, float]:
        """Convert pixel coordinates to spherical angles (θ, φ)"""
        # Normalize pixel coordinates relative to principal point
        u_norm = (u - cx) / fx
        v_norm = (v - cy) / fy
        
        # Calculate spherical coordinates
        theta = np.sqrt(u_norm**2 + v_norm**2)  # polar angle
        phi = np.arctan2(v_norm, u_norm)        # azimuth angle
        
        return theta, phi
    
    # Convert both positions to spherical angles
    prev_theta, prev_phi = pixel_to_angles(prev_pos[0], prev_pos[1])
    curr_theta, curr_phi = pixel_to_angles(curr_pos[0], curr_pos[1])
    
    # Convert to Cartesian angular coordinates for easier head movement calculation
    prev_ang_x = prev_theta * np.cos(prev_phi)  # horizontal component
    prev_ang_y = prev_theta * np.sin(prev_phi)  # vertical component
    
    curr_ang_x = curr_theta * np.cos(curr_phi)
    curr_ang_y = curr_theta * np.sin(curr_phi)
    
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
    Calculate head movement with Aria spherical model for 1408x1408 images.
    """
    if prev_red_pos is None or curr_red_pos is None:
        return None
    
    # Check if this is Aria RGB camera (1408x1408)
    is_aria_rgb = (image_width == 1408 and image_height == 1408)
    
    # if is_aria_rgb:
    #     # Use Aria's spherical model
    #     print("Detected Aria RGB camera - using spherical model")
        
    #     # Estimate intrinsic parameters from FOV
    #     intrinsics = estimate_aria_intrinsics_from_fov(image_width, image_height, fov_degrees=110.0)
        
    #     print(f"Estimated Aria intrinsics: fx={intrinsics['fx']:.1f}, fy={intrinsics['fy']:.1f}, cx={intrinsics['cx']:.1f}, cy={intrinsics['cy']:.1f}")
        
    #     # Use spherical model for calculation
    #     return calculate_head_movement_spherical_model(
    #         prev_red_pos, curr_red_pos,
    #         intrinsics["fx"], intrinsics["fy"], 
    #         intrinsics["cx"], intrinsics["cy"]
    #     )
    # else:
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

def remap_position_from_movement_spherical(
    start_pos: Tuple[float, float],
    head_movement: Dict[str, Any],
    fx: float, fy: float, cx: float, cy: float
) -> Optional[Tuple[float, float]]:
    """
    Remap position using spherical model.
    """
    if start_pos is None or head_movement is None:
        return None
    
    if np.isnan(head_movement['horizontal']['radians']) or np.isnan(head_movement['vertical']['radians']):
        return None
    
    # Convert starting position to spherical angles
    u_norm = (start_pos[0] - cx) / fx
    v_norm = (start_pos[1] - cy) / fy
    
    start_theta = np.sqrt(u_norm**2 + v_norm**2)
    start_phi = np.arctan2(v_norm, u_norm)
    
    # Convert to Cartesian angular coordinates
    start_ang_x = start_theta * np.cos(start_phi)
    start_ang_y = start_theta * np.sin(start_phi)
    
    # Apply head movement (note: movement is inverted, so we add it back)
    new_ang_x = start_ang_x - head_movement['horizontal']['radians']
    new_ang_y = start_ang_y - head_movement['vertical']['radians']
    
    # Convert back to spherical coordinates
    new_theta = np.sqrt(new_ang_x**2 + new_ang_y**2)
    new_phi = np.arctan2(new_ang_y, new_ang_x) if new_theta > 1e-8 else 0.0
    
    # Convert back to pixel coordinates using spherical model
    predicted_u = fx * new_theta * np.cos(new_phi) + cx
    predicted_v = fy * new_theta * np.sin(new_phi) + cy
    
    return (predicted_u, predicted_v)

def remap_position_from_movement(start_pos, head_movement, image_width, image_height, video_fov_degrees=104.0):
    """
    Remap position from head movement with Aria spherical model support.
    """
    if start_pos is None or head_movement is None or np.isnan(head_movement['horizontal']['radians']):
        return None
    
    # Check if this is Aria RGB camera
    is_aria_rgb = (image_width == 1408 and image_height == 1408)
    
    # if is_aria_rgb:
    #     # Use Aria's spherical model
    #     intrinsics = estimate_aria_intrinsics_from_fov(image_width, image_height, fov_degrees=110.0)
        
    #     return remap_position_from_movement_spherical(
    #         start_pos, head_movement,
    #         intrinsics["fx"], intrinsics["fy"],
    #         intrinsics["cx"], intrinsics["cy"]
    #     )
    # else:
    # Use linear method for other cameras
    horizontal_radians = head_movement['horizontal']['radians']
    vertical_radians = head_movement['vertical']['radians']

    horizontal_angle_degrees = np.degrees(horizontal_radians)
    vertical_angle_degrees = np.degrees(vertical_radians)

    # For square images (like Aria 1408x1408), both FOVs should be the same
    if image_width == image_height:
        horizontal_fov_degrees = video_fov_degrees
        vertical_fov_degrees = video_fov_degrees
    else:
        # For rectangular images, calculate vertical FOV based on aspect ratio
        aspect_ratio = image_width / image_height
        horizontal_fov_degrees = video_fov_degrees
        vertical_fov_degrees = video_fov_degrees / aspect_ratio
    
    horizontal_pixels_per_degree = image_width / horizontal_fov_degrees
    vertical_pixels_per_degree = image_height / vertical_fov_degrees

    horizontal_pixel_change = -horizontal_angle_degrees * horizontal_pixels_per_degree
    vertical_pixel_change = -vertical_angle_degrees * vertical_pixels_per_degree

    predicted_x = start_pos[0] + horizontal_pixel_change
    predicted_y = start_pos[1] + vertical_pixel_change

    return (predicted_x, predicted_y)