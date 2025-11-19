#!/usr/bin/env python3
"""
CARLA Traffic-Light Detection + Model-Gated Autopilot (with CARLA fallback)

- Spawns a vehicle using try_spawn_actor.
- Attaches an RGB camera and streams frames into pygame.
- Runs a custom Ultralytics YOLO model on each frame.
- Autopilot drives, but we gate GO/STOP using the model:
    red   -> STOP
    green -> GO
    yellow-> GO
  If model disagrees with CARLA's current traffic-light state for this vehicle,
  we TRUST CARLA (fallback).

Notes:
- Uses Traffic Manager with "ignore lights" so autopilot won't independently stop.
- We override controls to brake when STOP is required; release when GO.

Requirements:
  pip install ultralytics pygame opencv-python numpy
  CARLA 0.9.x (client libs on PYTHONPATH)
"""

import random
import time
import weakref
from collections import deque, Counter

import numpy as np
import pygame
import cv2

import carla
from ultralytics import YOLO

# -----------------------------
# Config
# -----------------------------
WEIGHTS_PATH = "traffic_light.pt"  # your trained model
CLASS_NAMES = ["green", "red", "yellow"]  # class order in your model
WINDOW_W, WINDOW_H = 1000, 800
CAM_FOV = "90"
FPS = 30
CONF_THRESH = 0.4
IOU_THRESH = 0.45
CARLA_HOST = "localhost"
CARLA_PORT = 2000
Z_LIFT = 0.5
MAX_SPAWN_TRIES = 80

# smoothing for model outputs (frames)
SMOOTH_N = 3              # majority vote over last N frames
MIN_BOX_PIXELS = 16 * 16  # ignore tiny boxes

# Vehicle speed settings
VEHICLE_SPEED_LIMIT = 120  # km/h - maximum speed for the vehicle

# ROI/Zoom Detection Settings
USE_ROI_ZOOM = True       # Enable ROI-based detection for better accuracy
USE_DUAL_DETECTION = True # Process both zoomed and non-zoomed frames for better coverage
ROI_TOP_RATIO = 0.30       # Start from top of image (0%)
ROI_BOTTOM_RATIO = 0.6    # End at 60% of image height (upper portion)
ROI_LEFT_RATIO = 0.3      # Start from 30% from left (narrower focus)
ROI_RIGHT_RATIO = 0.8     # End at 70% from right (narrower focus)
ZOOM_SCALE = 1.75         # Scale factor for zoomed detection (1.5x zoom)

# Enhanced stopping behavior
STOP_DISTANCE_THRESHOLD = 30.0  # meters - start slowing when red light detected
RED_LIGHT_BRAKE_FORCE = 0.9    # Brake force when red light detected
GRADUAL_BRAKE_DISTANCE = 15.0   # meters - distance to start gradual braking
MIN_STOP_SPEED = 0.5            # km/h - speed threshold to consider stopped

# Yellow light behavior
YELLOW_LIGHT_BRAKE_FORCE = 0.5  # Brake force to slow down on yellow (gentler than red)
YELLOW_SPEED_REDUCTION = 0.4    # Target speed reduction (60% of current speed)

# Stop/Go transition settings
GREEN_LIGHT_DELAY = 0.5         # seconds - wait before resuming after green detected
RED_LIGHT_HOLD_TIME = 0.3       # seconds - hold brake state to prevent flickering
RESUME_THROTTLE = 0.5           # Initial throttle when resuming (0.0-1.0)
RESUME_DURATION = 2.0           # seconds - how long to apply initial throttle

# -----------------------------
# Helper: safe vehicle spawn
# -----------------------------
def spawn_vehicle_safe(world: carla.World, vehicle_bp: carla.ActorBlueprint,
                       max_tries: int = MAX_SPAWN_TRIES, z_lift: float = Z_LIFT):
    spawns = world.get_map().get_spawn_points()
    if not spawns:
        raise RuntimeError("No spawn points available in the current map.")
    random.shuffle(spawns)

    for sp in spawns[:max_tries]:
        sp.location.z += z_lift
        veh = world.try_spawn_actor(vehicle_bp, sp)
        if veh is not None:
            return veh

    for sp in spawns:
        sp.location.z += z_lift
        veh = world.try_spawn_actor(vehicle_bp, sp)
        if veh is not None:
            return veh
    return None

# -----------------------------
# Camera callback
# -----------------------------
class CameraBuffer:
    def __init__(self):
        self.latest = None

    def __call__(self, image: carla.Image):
        # Carla gives BGRA uint8; keep BGR for OpenCV
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        self.latest = array[:, :, :3].copy()

# -----------------------------
# Utility: majority vote
# -----------------------------
def majority_state(history: deque[str]) -> str | None:
    if not history:
        return None
    c = Counter(history)
    # Break ties with order preference: red > green > yellow (more conservative)
    # but you can change this if you prefer
    pref = ["red", "green", "yellow"]
    best_count = max(c.values())
    candidates = [s for s, k in c.items() if k == best_count]
    for p in pref:
        if p in candidates:
            return p
    return candidates[0]

# -----------------------------
# ROI/Zoom Functions
# -----------------------------
def extract_roi(image: np.ndarray, top_ratio=0.0, bottom_ratio=0.6, 
                left_ratio=0.2, right_ratio=0.8):
    """
    Extract region of interest from image (upper center portion where TLs appear)
    Returns: roi_image, (x_offset, y_offset) for coordinate mapping
    """
    h, w = image.shape[:2]
    y1 = int(h * top_ratio)
    y2 = int(h * bottom_ratio)
    x1 = int(w * left_ratio)
    x2 = int(w * right_ratio)
    
    roi = image[y1:y2, x1:x2].copy()
    return roi, (x1, y1)

def zoom_image(image: np.ndarray, scale: float = 1.5):
    """
    Zoom into center of image by scale factor
    Returns: zoomed_image, (x_offset, y_offset) for coordinate mapping back
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h / scale), int(w / scale)
    
    # Calculate crop coordinates (center crop)
    y1 = (h - new_h) // 2
    y2 = y1 + new_h
    x1 = (w - new_w) // 2
    x2 = x1 + new_w
    
    cropped = image[y1:y2, x1:x2]
    # Resize back to original size for consistent processing
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    
    return zoomed, (x1, y1), scale

def map_bbox_to_original(bbox, offset, scale=1.0):
    """
    Map bounding box from ROI/zoomed coordinates back to original image
    """
    # Convert to regular Python numbers to avoid PyTorch tensor issues
    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    x_off, y_off = offset
    
    # If zoomed, scale back
    if scale != 1.0:
        x1 = int(x1 / scale) + x_off
        y1 = int(y1 / scale) + y_off
        x2 = int(x2 / scale) + x_off
        y2 = int(y2 / scale) + y_off
    else:
        x1 = int(x1) + x_off
        y1 = int(y1) + y_off
        x2 = int(x2) + x_off
        y2 = int(y2) + y_off
    
    return (x1, y1, x2, y2)

# -----------------------------
# Utility: carla TL enum -> str
# -----------------------------
def carla_tl_to_str(tl_state: carla.TrafficLightState | None) -> str | None:
    if tl_state is None:
        return None
    name = tl_state.name.lower()
    if "red" in name:
        return "red"
    if "green" in name:
        return "green"
    if "yellow" in name or "amber" in name:
        return "yellow"
    return None

# -----------------------------
# Decision: effective state
# -----------------------------
def decide_effective_state(model_state: str | None, carla_state: str | None) -> str | None:
    """
    Returns 'red'/'green'/'yellow' or None if unknown.
    Rule:
      - If both present and DIFFER, TRUST CARLA.
      - Else use whichever is present (model preferred if equal).
    """
    if carla_state and model_state and carla_state != model_state:
        return carla_state
    return model_state or carla_state

# -----------------------------
# Control helpers
# -----------------------------
def apply_stop(vehicle: carla.Vehicle, brake_force: float = 1.0):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake_force, hand_brake=False))

def apply_gradual_brake(vehicle: carla.Vehicle, distance: float, max_distance: float = GRADUAL_BRAKE_DISTANCE):
    """Apply gradual braking based on distance to traffic light"""
    # Calculate brake force based on distance (closer = more brake)
    brake_force = max(0.3, min(1.0, 1.0 - (distance / max_distance)))
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=brake_force, hand_brake=False))

def release_brake(vehicle: carla.Vehicle):
    # Release brake; autopilot will command throttle/steer next tick
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))

def apply_resume_throttle(vehicle: carla.Vehicle, throttle: float = 0.5):
    """Apply gentle throttle to help vehicle resume from stop"""
    # Get current control to preserve steering
    control = vehicle.get_control()
    vehicle.apply_control(carla.VehicleControl(
        throttle=throttle, 
        steer=control.steer,  # Preserve autopilot steering
        brake=0.0, 
        hand_brake=False
    ))

def get_vehicle_speed_kmh(vehicle: carla.Vehicle) -> float:
    """Get vehicle speed in km/h"""
    vel = vehicle.get_velocity()
    return 3.6 * np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate union area
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area

# -----------------------------
# Main
# -----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("CARLA TL - Model-Gated Autopilot (CARLA fallback)")
    display = pygame.display.set_mode(
        (WINDOW_W, WINDOW_H), pygame.HWSURFACE | pygame.DOUBLEBUF
    )
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Load YOLO model
    model = YOLO(WEIGHTS_PATH)

    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)

    # World & TM
    world = client.get_world()
    # Pick a town if you want a specific one:
    world = client.load_world("Town03")

    tm = client.get_trafficmanager()
    # If you're running synchronous simulation externally, set True + world settings.
    tm.set_synchronous_mode(False)

    blueprint_library = world.get_blueprint_library()

    # Choose a vehicle blueprint
    vehicle_bp = blueprint_library.filter("vehicle.*")[0]

    # Try to spawn safely
    vehicle = spawn_vehicle_safe(world, vehicle_bp)
    if vehicle is None:
        world = client.load_world(world.get_map().name)
        time.sleep(1.0)
        blueprint_library = world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*")[0]
        vehicle = spawn_vehicle_safe(world, vehicle_bp)
        if vehicle is None:
            raise RuntimeError("Could not find a free spawn point after retrying.")

    # Spectator top-down (optional)
    try:
        spectator = world.get_spectator()
        transform = vehicle.get_transform()
        spectator.set_transform(
            carla.Transform(transform.location + carla.Location(z=30),
                            carla.Rotation(pitch=-90))
        )
    except Exception:
        pass

    # Register with Traffic Manager and enable autopilot
    vehicle.set_autopilot(True, tm.get_port())

    # IMPORTANT: let TM ignore native TL logic so it won't fight our gating
    tm.ignore_lights_percentage(vehicle, 100)  # 100% ignore traffic lights

    # Set vehicle speed limit
    tm.set_desired_speed(vehicle, VEHICLE_SPEED_LIMIT / 3.6)  # Convert km/h to m/s
    print(f"Vehicle speed limit set to: {VEHICLE_SPEED_LIMIT} km/h")

    # Camera sensor
    camera_bp = blueprint_library.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", str(WINDOW_W))
    camera_bp.set_attribute("image_size_y", str(WINDOW_H))
    camera_bp.set_attribute("fov", CAM_FOV)

    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

    cam_buffer = CameraBuffer()
    camera.listen(weakref.proxy(cam_buffer))

    # State smoothing
    recent_model_states: deque[str] = deque(maxlen=SMOOTH_N)
    last_effective_state = None
    last_apply_ts = 0.0
    stopped_for_red = False  # Track if we're currently stopped at red light
    green_detected_time = 0.0  # Time when green was first detected
    resume_start_time = 0.0  # Time when we started resuming
    last_red_time = 0.0  # Last time red was detected (for hysteresis)

    print("Model-gated autopilot started. Press 'q' to quit.")
    print(f"ROI Zoom Detection: {'ENABLED' if USE_ROI_ZOOM else 'DISABLED'}")
    print(f"Dual Detection (Zoom + No-Zoom): {'ENABLED' if USE_DUAL_DETECTION else 'DISABLED'}")
    if USE_ROI_ZOOM:
        print(f"  ROI: Top={ROI_TOP_RATIO*100:.0f}% Bottom={ROI_BOTTOM_RATIO*100:.0f}% Left={ROI_LEFT_RATIO*100:.0f}% Right={ROI_RIGHT_RATIO*100:.0f}%")
        print(f"  Zoom Scale: {ZOOM_SCALE}x")

    running = True
    try:
        while running:
            clock.tick(FPS)

            # Pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    running = False

            img = cam_buffer.latest
            if img is None:
                display.fill((0, 0, 0))
                pygame.display.flip()
                continue

            # Prepare images for detection
            img_original = img.copy()  # Keep original for display
            detection_images = []  # List of (image, offset, scale, source_name)
            
            # Always include ROI-based detection
            if USE_ROI_ZOOM:
                # Extract ROI (focus on upper center where TLs are)
                roi_img, roi_offset = extract_roi(
                    img, ROI_TOP_RATIO, ROI_BOTTOM_RATIO, 
                    ROI_LEFT_RATIO, ROI_RIGHT_RATIO
                )
                
                # Add non-zoomed ROI detection (good for nearby lights)
                if USE_DUAL_DETECTION:
                    detection_images.append((roi_img, roi_offset, 1.0, "ROI"))
                
                # Add zoomed ROI detection (good for distant lights)
                if ZOOM_SCALE > 1.0:
                    zoomed_img, zoom_offset, scale = zoom_image(roi_img, ZOOM_SCALE)
                    # Combine offsets
                    combined_offset = (roi_offset[0] + zoom_offset[0], roi_offset[1] + zoom_offset[1])
                    detection_images.append((zoomed_img, combined_offset, scale, "Zoomed"))
                else:
                    detection_images.append((roi_img, roi_offset, 1.0, "ROI"))
            else:
                # If ROI/Zoom disabled, just use full image
                detection_images.append((img, (0, 0), 1.0, "Full"))
            
            # Run YOLO inference on all detection images
            all_detected_boxes = []
            
            for detection_img, offset, scale, source in detection_images:
                results = model(
                    detection_img,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    verbose=False,
                    stream=False
                )

                # Process detections from this image
                if results:
                    r = results[0]
                    if hasattr(r, "boxes") and r.boxes is not None:
                        for box in r.boxes:
                            if box.cls is None:
                                continue
                            cls_id = int(box.cls[0])
                            if not (0 <= cls_id < len(CLASS_NAMES)):
                                continue
                            cls_name = CLASS_NAMES[cls_id]
                            conf = float(box.conf[0])
                            x1, y1, x2, y2 = box.xyxy[0]
                            w, h = float(x2 - x1), float(y2 - y1)
                            if w * h < MIN_BOX_PIXELS:
                                continue  # ignore tiny boxes
                            
                            # Map box back to original coordinates
                            if USE_ROI_ZOOM:
                                orig_box = map_bbox_to_original([x1, y1, x2, y2], offset, scale)
                            else:
                                orig_box = (int(x1), int(y1), int(x2), int(y2))
                            
                            all_detected_boxes.append((orig_box, cls_name, conf, source))

            # Choose the highest-confidence valid TL class from all detections
            model_state_frame = None
            detected_boxes = []  # Store unique boxes for visualization
            
            # Remove duplicate detections (using IoU threshold)
            unique_boxes = []
            for box_data in all_detected_boxes:
                box_coords, cls_name, conf, source = box_data
                
                # Check if this box overlaps significantly with existing boxes
                is_duplicate = False
                for existing_box, existing_cls, existing_conf, existing_source in unique_boxes:
                    iou = calculate_iou(box_coords, existing_box)
                    if iou > 0.5:  # Same detection from different images
                        # Keep the one with higher confidence
                        if conf > existing_conf:
                            unique_boxes.remove((existing_box, existing_cls, existing_conf, existing_source))
                            unique_boxes.append((box_coords, cls_name, conf, source))
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    unique_boxes.append((box_coords, cls_name, conf, source))
            
            # Find best detection
            best_conf = -1.0
            best_cls_name = None
            
            for box_coords, cls_name, conf, source in unique_boxes:
                detected_boxes.append((box_coords, cls_name, conf, source))
                if conf > best_conf:
                    best_conf = conf
                    best_cls_name = cls_name

            if best_cls_name is not None:
                model_state_frame = best_cls_name  # 'red'|'green'|'yellow'

            # Draw boxes on original image
            for box_coords, cls_name, conf, source in detected_boxes:
                x1, y1, x2, y2 = box_coords
                
                # Color based on state
                if cls_name == "red":
                    color = (0, 0, 255)  # Red
                elif cls_name == "green":
                    color = (0, 255, 0)  # Green
                elif cls_name == "yellow":
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (255, 0, 255)  # Magenta for unknown
                
                cv2.rectangle(img_original, (x1, y1), (x2, y2), color, 3)
                
                # Add label with background and source
                label = f"{cls_name.upper()} {conf:.2f} [{source}]"
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(img_original, (x1, y1 - label_h - 10), 
                            (x1 + label_w, y1), color, -1)
                cv2.putText(img_original, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            # Update smoothing buffer
            if model_state_frame is not None:
                recent_model_states.append(model_state_frame)
            model_state = majority_state(recent_model_states)  # None|red|green|yellow

            # Get CARLA traffic light affecting this vehicle
            carla_state = None
            tl_actor = vehicle.get_traffic_light()
            if tl_actor is not None:
                try:
                    carla_state = carla_tl_to_str(tl_actor.get_state())
                except RuntimeError:
                    # Occasionally TL can be invalidated when changing maps, etc.
                    carla_state = None

            # Decide effective state (model unless mismatch, then CARLA)
            effective = decide_effective_state(model_state, carla_state)

            # Get vehicle speed
            vehicle_speed = get_vehicle_speed_kmh(vehicle)
            is_stopped = vehicle_speed < MIN_STOP_SPEED

            # Apply control according to effective state with enhanced logic
            # red->STOP, green/yellow->GO
            now = time.time()
            hud_note = "GO"
            brake_info = ""
            
            if effective == "red":
                last_red_time = now
                green_detected_time = 0.0  # Reset green timer
                
                if not stopped_for_red:
                    # Just detected red light, start stopping procedure
                    stopped_for_red = True
                    hud_note = "RED DETECTED - STOPPING"
                    brake_info = f"Braking ({RED_LIGHT_BRAKE_FORCE:.1f})"
                    # Progressive braking based on speed
                    if vehicle_speed > 30:
                        apply_stop(vehicle, 0.8)  # Gentler brake at high speed
                    else:
                        apply_stop(vehicle, RED_LIGHT_BRAKE_FORCE)
                else:
                    # Already stopping/stopped for red
                    if is_stopped:
                        hud_note = "STOPPED AT RED"
                        brake_info = "Holding (1.0)"
                        apply_stop(vehicle, 1.0)  # Full brake to hold position
                    else:
                        hud_note = "STOPPING FOR RED"
                        # Progressive braking as we slow down
                        brake_force = min(1.0, RED_LIGHT_BRAKE_FORCE + (1.0 - RED_LIGHT_BRAKE_FORCE) * (1.0 - vehicle_speed / 30))
                        brake_info = f"Braking ({brake_force:.1f})"
                        apply_stop(vehicle, brake_force)
                last_apply_ts = now
                
            elif effective in ("green", "yellow"):
                # Green or yellow detected
                if stopped_for_red:
                    # We were stopped at red, now light changed to green/yellow
                    
                    # IMPORTANT: If we detect YELLOW while stopped, it's likely RED/YELLOW confusion
                    # Stay stopped and don't move until we get stable GREEN detection
                    if effective == "yellow" and is_stopped:
                        hud_note = "YELLOW (STOPPED - HOLDING)"
                        brake_info = "Holding (1.0) - waiting for green"
                        apply_stop(vehicle, 1.0)  # Keep holding, ignore yellow flickering
                        last_apply_ts = now
                        # Don't start green timer for yellow, stay in stopped_for_red state
                    else:
                        # Either moving and yellow, or detected green
                        # Start timer when green first detected
                        if green_detected_time == 0.0 and effective == "green":
                            green_detected_time = now
                        
                        if effective == "green":
                            time_since_green = now - green_detected_time
                            
                            # Wait a bit before resuming (hysteresis to prevent false positives)
                            if time_since_green < GREEN_LIGHT_DELAY:
                                hud_note = f"GREEN - WAITING ({GREEN_LIGHT_DELAY - time_since_green:.1f}s)"
                                brake_info = "Holding, verifying green"
                                apply_stop(vehicle, 1.0)  # Keep holding
                            else:
                                # Time to resume!
                                if resume_start_time == 0.0:
                                    resume_start_time = now
                                
                                time_since_resume = now - resume_start_time
                                
                                # Apply initial throttle for smooth start
                                if time_since_resume < RESUME_DURATION:
                                    hud_note = "GREEN - RESUMING"
                                    brake_info = f"Throttle ({RESUME_THROTTLE:.1f})"
                                    apply_resume_throttle(vehicle, RESUME_THROTTLE)
                                else:
                                    # Fully resumed, let autopilot take over
                                    hud_note = "GREEN - DRIVING"
                                    brake_info = "Autopilot control"
                                    release_brake(vehicle)
                                    stopped_for_red = False
                                    green_detected_time = 0.0
                                    resume_start_time = 0.0
                        else:
                            # Yellow while moving (not stopped yet)
                            hud_note = "YELLOW - SLOWING DOWN"
                            brake_info = f"Braking ({YELLOW_LIGHT_BRAKE_FORCE:.1f})"
                            apply_stop(vehicle, YELLOW_LIGHT_BRAKE_FORCE)
                        
                        last_apply_ts = now
                else:
                    # Not stopped, cruising
                    if effective == "yellow":
                        # Yellow light - slow down (prepare to stop)
                        hud_note = "YELLOW - SLOWING DOWN"
                        brake_info = f"Braking ({YELLOW_LIGHT_BRAKE_FORCE:.1f})"
                        apply_stop(vehicle, YELLOW_LIGHT_BRAKE_FORCE)
                    else:
                        # Green light - drive normally
                        hud_note = "DRIVING"
                        brake_info = "Autopilot control"
                        release_brake(vehicle)
                    resume_start_time = 0.0
                    
                last_apply_ts = now
                
            else:
                # Unknown state - be conservative
                time_since_last_red = now - last_red_time if last_red_time > 0 else 999
                
                if stopped_for_red and is_stopped:
                    # If we were stopped for red and now state is unknown, stay stopped
                    # But only for a short time
                    if time_since_last_red < RED_LIGHT_HOLD_TIME:
                        hud_note = "STOPPED (holding)"
                        brake_info = "Holding, state unclear"
                        apply_stop(vehicle, 1.0)
                    else:
                        # Been too long, cautiously resume
                        hud_note = "NO DETECTION - CAUTIOUS RESUME"
                        brake_info = "Releasing slowly"
                        release_brake(vehicle)
                        stopped_for_red = False
                        green_detected_time = 0.0
                        resume_start_time = 0.0
                else:
                    # Not stopped, no detection - let autopilot proceed
                    hud_note = "NO DETECTION"
                    brake_info = "Autopilot control"
                    if stopped_for_red:
                        stopped_for_red = False
                        green_detected_time = 0.0
                        resume_start_time = 0.0

            # Compose HUD lines
            detection_sources = set([source for _, _, _, source in detected_boxes])
            detection_info = f"Detections: {len(detected_boxes)} from {', '.join(detection_sources) if detection_sources else 'None'}"
            
            # Add state timing info
            state_timing = ""
            if green_detected_time > 0:
                state_timing = f" (Green timer: {now - green_detected_time:.1f}s)"
            elif resume_start_time > 0:
                state_timing = f" (Resuming: {now - resume_start_time:.1f}s)"
            
            hud_lines = [
                f"FPS: {int(clock.get_fps())}",
                f"Speed: {vehicle_speed:.1f} km/h {'[STOPPED]' if is_stopped else '[MOVING]'}",
                f"Model (smoothed): {model_state or 'None'}",
                f"CARLA TL: {carla_state or 'None'}",
                f"Effective: {effective or 'None'}  ->  {hud_note}{state_timing}",
                f"Control: {brake_info}",
                detection_info,
                f"Dual Mode: {'ON' if USE_DUAL_DETECTION else 'OFF'} | ROI/Zoom: {'ON' if USE_ROI_ZOOM else 'OFF'}",
                "Rule: If model != CARLA => use CARLA",
            ]
            
            # Draw ROI box if enabled
            if USE_ROI_ZOOM:
                h, w = img_original.shape[:2]
                roi_x1 = int(w * ROI_LEFT_RATIO)
                roi_y1 = int(h * ROI_TOP_RATIO)
                roi_x2 = int(w * ROI_RIGHT_RATIO)
                roi_y2 = int(h * ROI_BOTTOM_RATIO)
                # Draw semi-transparent ROI box
                cv2.rectangle(img_original, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                            (255, 255, 0), 2)
                cv2.putText(img_original, "ROI", (roi_x1 + 5, roi_y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # Blit to pygame
            img_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
            surface = pygame.surfarray.make_surface(img_rgb.swapaxes(0, 1))
            display.blit(surface, (0, 0))

            y = 10
            for line in hud_lines:
                text_surf = font.render(line, True, (255, 255, 255))
                display.blit(text_surf, (10, y))
                y += 20

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        print("\nCleaning up")
        try:
            camera.stop()
        except Exception:
            pass
        for actor in [camera, vehicle]:
            try:
                if actor is not None:
                    actor.destroy()
            except Exception:
                pass
        pygame.quit()
        print("Done!")

if __name__ == "__main__":
    main()
