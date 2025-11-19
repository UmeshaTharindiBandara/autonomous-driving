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
SMOOTH_N = 8              # majority vote over last N frames
MIN_BOX_PIXELS = 16 * 16  # ignore tiny boxes

# Vehicle speed settings
VEHICLE_SPEED_LIMIT = 120  # km/h - maximum speed for the vehicle

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
def apply_stop(vehicle: carla.Vehicle):
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0, hand_brake=False))

def release_brake(vehicle: carla.Vehicle):
    # Release brake; autopilot will command throttle/steer next tick
    vehicle.apply_control(carla.VehicleControl(throttle=0.0, steer=0.0, brake=0.0, hand_brake=False))

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

    print("Model-gated autopilot started. Press 'q' to quit.")

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

            # Run YOLO inference (on BGR image)
            results = model(
                img,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                verbose=False,
                stream=False
            )

            # Choose the highest-confidence valid TL class (optional: filter by size)
            model_state_frame = None
            if results:
                r = results[0]
                if hasattr(r, "boxes") and r.boxes is not None:
                    best_conf = -1.0
                    best_cls_name = None
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
                        if conf > best_conf:
                            best_conf = conf
                            best_cls_name = cls_name

                    if best_cls_name is not None:
                        model_state_frame = best_cls_name  # 'red'|'green'|'yellow'

                    # Draw boxes (optional)
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls_id = int(box.cls[0]) if box.cls is not None else -1
                        cls_name = CLASS_NAMES[cls_id] if 0 <= cls_id < len(CLASS_NAMES) else str(cls_id)
                        conf = float(box.conf[0]) if box.conf is not None else 0.0
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

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

            # Apply control according to effective state
            # red->STOP, green/yellow->GO
            now = time.time()
            hud_note = "GO"
            if effective == "red":
                hud_note = "STOP"
                apply_stop(vehicle)
                last_apply_ts = now
            elif effective in ("green", "yellow"):
                # release brake if we recently forced a stop
                release_brake(vehicle)
                last_apply_ts = now
            else:
                # Unknown -> do nothing special; let autopilot proceed
                hud_note = "UNKNOWN"

            # Compose HUD lines
            hud_lines = [
                f"FPS: {int(clock.get_fps())}",
                f"Model (smoothed): {model_state or 'None'}",
                f"CARLA TL: {carla_state or 'None'}",
                f"Effective: {effective or 'None'}  ->  {hud_note}",
                "Rule: If model != CARLA => use CARLA",
            ]

            # Blit to pygame
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
