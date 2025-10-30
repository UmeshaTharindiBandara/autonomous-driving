import carla
import torch
import os
import cv2
import numpy as np
import time
import math
import csv
from collections import deque
from model.model import parsingNet
from utils.common import *
from utils.config import *
import scipy.special
import torchvision.transforms as transforms
from data.constant import culane_row_anchor, tusimple_row_anchor
from PIL import Image
from detection.yolo_distance_detector import YOLODistanceDetector
from detection.yolo_lane_filter import YOLOLaneFilter

# ==== Medium-article controller tunables ====
LANE_WIDTH_M   = 3.7          # nominal lane width
LOOK_Y_OFFSET  = 60           # pixels above BEV bottom where we evaluate center
PID_KP, PID_KI, PID_KD = 0.55, 0.02, 0.22
PID_I_LIM      = 0.6
STEER_RATE     = 0.03         # max change per frame
STEER_LIMIT    = 0.25         # final clamp [-0.25, 0.25]
EMA_ALPHA      = 0.30         # smooth error (meters) to reduce jitter
MISS_WINDOW    = 20           # frames window for lane-loss
MISS_THRESH    = 2            # stop only if <=2 frames in window had any lane
STEER_SIGN     = -1.0         # flip sign if sim steering is inverted

# ==== Obstacle Spawning Settings ====
AUTO_SPAWN_OBSTACLES = True   # Automatically spawn obstacles on startup
NUM_SPAWN_VEHICLES   = 10     # Number of vehicles to spawn on roads
NUM_SPAWN_PEDESTRIANS = 0     # Number of pedestrians to spawn (0 = disabled for now)
NUM_STATIC_OBSTACLES = 3      # Number of static vehicles in ego lane for testing

# ==== MDE Collision Avoidance Tunables ==== (NEW SECTION)
STOP_DISTANCE_M = 15.0        # Stop if obstacle is closer than 15 meters
LOOK_AHEAD_Y_FRAC = 0.70      # Ignore bottom 30% of image for obstacle depth
PIXEL_COUNT_THRESHOLD = 50    # Stop if > 50 pixels in danger zone are too close
DANGER_ZONE_COLOR = (0, 0, 255) # Red for the overlay

# ==== Manual driving tunables ====
MAN_STEER_STEP     = 0.04     # per press steer delta
MAN_STEER_DECAY    = 0.90     # steer return-to-center per frame
MAN_THR_STEP       = 0.05     # per press throttle delta
MAN_THR_DECAY      = 0.96     # passive throttle decay
MAN_BRAKE_STEP     = 0.08     # per press brake delta
MAN_BRAKE_DECAY    = 0.90     # passive brake decay
MAN_MAX_THR        = 0.85
MAN_MAX_BRAKE      = 1.00


class ROISelector:
    """Class for selecting and managing ROI points"""
    def __init__(self, csv_file="roi_points.csv"):
        self.csv_file = csv_file
        self.roi_points = []
        self.temp_points = []
        self.selecting = False
        self.window_name = "Select ROI Points"
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.temp_points) < 3:
            self.temp_points.append((x, y))
            print(f"Point {len(self.temp_points)}: ({x}, {y})")
            img_copy = param.copy()
            for i, point in enumerate(self.temp_points):
                cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                label = "Left" if i==0 else ("Right" if i==1 else "Top")
                cv2.putText(img_copy, label, (point[0]+10, point[1]-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(img_copy, self.temp_points[i-1], point, (0, 255, 0), 2)
                if len(self.temp_points) == 3:
                    cv2.line(img_copy, self.temp_points[2], self.temp_points[0], (0, 255, 0), 2)
            cv2.imshow(self.window_name, img_copy)
    
    def load_from_csv(self):
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'r') as file:
                    reader = csv.reader(file)
                    _ = next(reader)
                    row = next(reader)
                    self.roi_points = [
                        (int(row[0]), int(row[1])),  # Left
                        (int(row[2]), int(row[3])),  # Right
                        (int(row[4]), int(row[5]))   # Top
                    ]
                print(f"✓ ROI points loaded from {self.csv_file}")
                print(f"   Left: {self.roi_points[0]}, Right: {self.roi_points[1]}, Top: {self.roi_points[2]}")
                return True
            except Exception as e:
                print(f"⚠️ Error loading ROI points: {e}")
                return False
        return False
    
    def save_to_csv(self):
        try:
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['left_x', 'left_y', 'right_x', 'right_y', 'top_x', 'top_y'])
                writer.writerow([
                    self.roi_points[0][0], self.roi_points[0][1],
                    self.roi_points[1][0], self.roi_points[1][1],
                    self.roi_points[2][0], self.roi_points[2][1]
                ])
            print(f"✓ ROI points saved to {self.csv_file}")
            return True
        except Exception as e:
            print(f"⚠️ Error saving ROI points: {e}")
            return False
    
    def select_roi(self, sample_image):
        print("\n🔧 ROI Selection Mode")
        print("1) Left-bottom  2) Right-bottom  3) Top-center")
        print("[r] reset  [s] save  [q] cancel")
        img_copy = sample_image.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, img_copy)
        cv2.imshow(self.window_name, img_copy)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.temp_points = []
                cv2.imshow(self.window_name, sample_image)
                print("↺ Selection reset")
            elif key == ord('s') and len(self.temp_points) == 3:
                self.roi_points = self.temp_points.copy()
                self.save_to_csv()
                cv2.destroyWindow(self.window_name)
                return True
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return False
            
    def create_roi_mask(self, img_shape):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if len(self.roi_points) == 3:
            pts = np.array(self.roi_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def filter_lanes_by_roi(self, lanes, img_shape):
        if len(self.roi_points) != 3 or len(lanes) == 0:
            return lanes
        mask = self.create_roi_mask(img_shape)
        filtered_lanes = []
        for lane in lanes:
            points_in_roi = 0
            total_points = len(lane)
            bottom_half = sorted(lane, key=lambda p: p[1], reverse=True)[:total_points//2 + 1]
            for point in bottom_half:
                if 0 <= point[0] < img_shape[1] and 0 <= point[1] < img_shape[0]:
                    if mask[point[1], point[0]] > 0:
                        points_in_roi += 1
            threshold = 0.3 if len(lanes) == 1 else 0.4
            if points_in_roi > len(bottom_half) * threshold:
                filtered_lanes.append(lane)
        if len(filtered_lanes) > 2:
            center_x = img_shape[1] // 2
            lane_scores = []
            for lane in filtered_lanes:
                bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:5]
                if len(bottom_points) > 0:
                    avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                    distance_to_center = abs(avg_x - center_x)
                    score = distance_to_center - len(lane) * 2
                    lane_scores.append((lane, score))
            lane_scores.sort(key=lambda x: x[1])
            filtered_lanes = [ls[0] for ls in lane_scores[:2]]
        return filtered_lanes


class PID:
    def __init__(self, kp, ki, kd, i_limit=0.6, rate_limit=0.03, out_limit=0.25):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.i_limit   = i_limit
        self.rate_limit= rate_limit
        self.out_limit = out_limit
        self.i_term = 0.0
        self.prev_e = None
        self.prev_t = None
    def step(self, e, now, last_out=None):
        if self.prev_t is None:
            self.prev_t = now
            self.prev_e = e
            u = self.kp * e
        else:
            dt = max(1e-3, now - self.prev_t)
            p = self.kp * e
            self.i_term += self.ki * e * dt
            self.i_term = max(-self.i_limit, min(self.i_limit, self.i_term))
            d = self.kd * (e - self.prev_e) / dt
            u = p + self.i_term + d
            self.prev_e = e
            self.prev_t = now
        u = -u * STEER_SIGN
        u = max(-self.out_limit, min(self.out_limit, u))
        if last_out is None:
            return u
        delta = u - last_out
        if delta > self.rate_limit:  u = last_out + self.rate_limit
        if delta < -self.rate_limit: u = last_out - self.rate_limit
        return u


class LaneDetectionController:
    def __init__(self, cfg_path="configs/tusimple.py", model_path="tusimple_18.pth"):
        self.setup_lane_detection(cfg_path, model_path)
        
        # --- YOLO-based obstacle detection with distance estimation ---
        try:
            self.yolo_detector = YOLODistanceDetector(
                model_path='yolo11n.pt',  # Fast model; use yolo11s.pt for better accuracy
                conf_threshold=0.5
            )
            # Calibrate for CARLA camera (1280x720, 90° FOV)
            self.yolo_detector.calibrate_focal_length(1280, 720, fov_degrees=90)
            print("✓ YOLO obstacle detector initialized")
        except Exception as e:
            print(f"⚠️ Failed to initialize YOLO detector: {e}")
            print("   Install with: pip install ultralytics")
            self.yolo_detector = None
        
        # --- NEW: Lane-aware YOLO filter ---
        self.yolo_lane_filter = YOLOLaneFilter(img_width=self.img_w, img_height=self.img_h)
        
        self.roi_selector = ROISelector()
        self.setup_carla()
        self.setup_control_parameters()
        self.setup_bev()
        self.setup_roi_initial()

        # ===== Mode & manual state =====
        self.mode = 'manual'  # Start in manual, press L for auto
        self.awaiting_roi_choice = False
        self.manual_throttle = 0.0
        self.manual_brake    = 0.0
        self.manual_steer    = 0.0
        self.manual_reverse  = False
        
    def setup_lane_detection(self, cfg_path, model_path):
        import sys
        sys.argv = ['script_name', cfg_path, '--test_model', model_path]
        args, self.cfg = merge_config()
        if self.cfg.dataset == 'CULane':
            self.cls_num_per_lane = 18
            self.img_w, self.img_h = 1640, 590
            self.row_anchor = culane_row_anchor
        elif self.cfg.dataset == 'Tusimple':
            self.cls_num_per_lane = 56
            self.img_w, self.img_h = 1280, 720
            self.row_anchor = tusimple_row_anchor
        else:
            raise NotImplementedError
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        self.net = parsingNet(
            pretrained=False, 
            backbone=self.cfg.backbone,
            cls_dim=(self.cfg.griding_num+1, self.cls_num_per_lane, 4),
            use_aux=False
        ).cuda()
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            if 'model' in state_dict:
                state_dict = state_dict['model']
            compatible_state_dict = { (k[7:] if k.startswith('module.') else k): v
                                      for k,v in state_dict.items() }
            self.net.load_state_dict(compatible_state_dict, strict=False)
            self.net.eval()
            print(f"✓ Successfully loaded model weights from {model_path}")
        except Exception as e:
            print(f"⚠️ Error loading weights: {e}")
            raise
    
    def setup_carla(self):
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04')
        bp = self.world.get_blueprint_library()
        veh_bp = bp.filter('vehicle.tesla.model3')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        self.spawn_point = spawn_points[0]
        self.vehicle = self.world.spawn_actor(veh_bp, self.spawn_point)
        camera_bp = bp.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(self.img_w))
        camera_bp.set_attribute('image_size_y', str(self.img_h))
        camera_bp.set_attribute('fov', '90')
        cam_tf = carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(camera_bp, cam_tf, attach_to=self.vehicle)
        self.camera_data = None
        self.camera.listen(self._camera_callback)
        
        # Store spawned obstacles for cleanup
        self.spawned_actors = []
        
        # Auto-spawn obstacles if enabled
        if AUTO_SPAWN_OBSTACLES:
            self.spawn_traffic_obstacles()
        
    def spawn_traffic_obstacles(self):
        """Spawn vehicles and pedestrians for testing obstacle detection"""
        print("\n🚗 Auto-spawning traffic obstacles...")
        
        spawn_start_time = time.time()
        timeout = 30  # 30 second timeout for spawning
        
        try:
            bp = self.world.get_blueprint_library()
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Spawn vehicles
            vehicle_bps = bp.filter('vehicle.*')
            vehicles_spawned = 0
            
            # Use random spawn points, avoiding index 0 (where ego vehicle is)
            available_points = spawn_points[1:min(NUM_SPAWN_VEHICLES * 2 + 1, len(spawn_points))]
            import random
            random.shuffle(available_points)
            
            for i, spawn_point in enumerate(available_points[:NUM_SPAWN_VEHICLES]):
                try:
                    vehicle_bp = random.choice(vehicle_bps)
                    
                    # Set color variety
                    if vehicle_bp.has_attribute('color'):
                        color = random.choice(vehicle_bp.get_attribute('color').recommended_values)
                        vehicle_bp.set_attribute('color', color)
                    
                    vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
                    if vehicle is not None:
                        self.spawned_actors.append(vehicle)
                        vehicles_spawned += 1
                        
                        # Set ALL vehicles to autopilot for realistic traffic
                        vehicle.set_autopilot(True)
                except:
                    pass
            
            print(f"   ✓ Spawned {vehicles_spawned} vehicles (all on autopilot)")
            
            # Spawn static obstacles in ego vehicle's lane for testing
            static_spawned = 0
            if NUM_STATIC_OBSTACLES > 0:
                print(f"   🚧 Spawning {NUM_STATIC_OBSTACLES} static obstacles in driving lane...")
                
                # Use Tesla for visibility
                tesla_bps = [bp for bp in vehicle_bps if 'tesla' in bp.id.lower()]
                if not tesla_bps:
                    tesla_bps = vehicle_bps
                
                # Get ego vehicle spawn point and use CARLA waypoint system for accurate lane positioning
                ego_spawn = spawn_points[0]
                
                # Get the map and find the waypoint at ego spawn location
                carla_map = self.world.get_map()
                ego_waypoint = carla_map.get_waypoint(ego_spawn.location)
                
                # MODIFIED: Use waypoint system for accurate lane-centered spawning
                distances = [50, 80, 120]  # meters ahead
                
                for i, distance in enumerate(distances[:NUM_STATIC_OBSTACLES]):
                    try:
                        # Get waypoint ahead using CARLA's next() method
                        # This automatically keeps us centered in the lane
                        current_waypoint = ego_waypoint
                        remaining_distance = distance
                        
                        # Move forward in small steps to stay on the road
                        while remaining_distance > 0:
                            step = min(2.0, remaining_distance)  # 2m steps
                            next_waypoints = current_waypoint.next(step)
                            if len(next_waypoints) > 0:
                                current_waypoint = next_waypoints[0]
                                remaining_distance -= step
                            else:
                                break
                        
                        # Get the transform from the final waypoint (already lane-centered)
                        spawn_transform = current_waypoint.transform
                        spawn_transform.location.z += 0.5  # Raise slightly to avoid ground collision
                        
                        # Spawn vehicle
                        vehicle_bp = random.choice(tesla_bps)
                        
                        # Set distinct color for visibility
                        if vehicle_bp.has_attribute('color'):
                            colors = ['255,0,0', '255,255,0', '0,255,255']  # Red, Yellow, Cyan
                            if i < len(colors):
                                vehicle_bp.set_attribute('color', colors[i])
                        
                        vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_transform)
                        
                        if vehicle is not None:
                            self.spawned_actors.append(vehicle)
                            static_spawned += 1
                            print(f"      → Static obstacle {static_spawned} at ~{distance}m ahead (Lane-centered via waypoint)")
                            
                            # Keep vehicle static (no autopilot)
                            # Apply handbrake to ensure it stays put
                            vehicle.apply_control(carla.VehicleControl(
                                throttle=0.0,
                                brake=1.0,
                                hand_brake=True
                            ))
                        else:
                            print(f"      ⚠️ Failed to spawn at {distance}m (spawn returned None)")
                            
                    except Exception as e:
                        print(f"      ⚠️ Failed to spawn at {distance}m: {e}")
                
                print(f"   ✓ Spawned {static_spawned} static obstacles")
            
            vehicles_spawned += static_spawned
            
            # Spawn pedestrians (simplified - skip if problematic)
            pedestrians_spawned = 0
            
            if NUM_SPAWN_PEDESTRIANS > 0:
                print(f"   🚶 Attempting to spawn {NUM_SPAWN_PEDESTRIANS} pedestrians...")
                walker_bps = bp.filter('walker.pedestrian.*')
                
                # Spawn pedestrians around spawn point 0 area
                ref_loc = spawn_points[0].location
                
                for i in range(NUM_SPAWN_PEDESTRIANS * 3):  # Try 3x times
                    if pedestrians_spawned >= NUM_SPAWN_PEDESTRIANS:
                        break
                    
                    try:
                        # Random position around reference
                        angle = random.uniform(0, 2 * 3.14159)
                        distance = random.uniform(10, 50)
                        
                        spawn_loc = carla.Location(
                            x=ref_loc.x + distance * math.cos(angle),
                            y=ref_loc.y + distance * math.sin(angle),
                            z=ref_loc.z + 1.0
                        )
                        
                        spawn_transform = carla.Transform(
                            spawn_loc,
                            carla.Rotation(yaw=random.uniform(0, 360))
                        )
                        
                        walker_bp = random.choice(walker_bps)
                        walker = self.world.try_spawn_actor(walker_bp, spawn_transform)
                        
                        if walker is not None:
                            self.spawned_actors.append(walker)
                            
                            # Spawn controller
                            walker_controller_bp = bp.find('controller.ai.walker')
                            controller = self.world.try_spawn_actor(walker_controller_bp, carla.Transform(), walker)
                            
                            if controller is not None:
                                self.spawned_actors.append(controller)
                                pedestrians_spawned += 1
                                
                                # Start walking (simple, no navigation)
                                controller.start()
                                controller.set_max_speed(1.0 + random.random() * 0.5)
                    except Exception as e:
                        # Silently skip failed pedestrians
                        pass
                
                print(f"   ✓ Spawned {pedestrians_spawned} pedestrians")
            
            print(f"   ✓ Total obstacles: {vehicles_spawned + pedestrians_spawned}")
            
            # Give time for spawned actors to settle
            if vehicles_spawned + pedestrians_spawned > 0:
                print("   ⏳ Waiting for actors to settle...")
                for i in range(3):
                    if time.time() - spawn_start_time > timeout:
                        print("   ⚠️ Spawn timeout - proceeding anyway")
                        break
                    self.world.tick()
                    time.sleep(0.1)
                print("   ✓ Ready!")
            
        except Exception as e:
            print(f"   ⚠️ Error spawning obstacles: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"   ⏱️ Spawning took {time.time() - spawn_start_time:.1f}s")
    
    def cleanup_spawned_actors(self):
        """Destroy all spawned traffic actors - handles already-destroyed actors"""
        if len(self.spawned_actors) > 0:
            print(f"\n🧹 Cleaning up {len(self.spawned_actors)} spawned actors...")
            destroyed_count = 0
            already_destroyed_count = 0
            
            for actor in self.spawned_actors:
                try:
                    # Check if actor is still alive before destroying
                    if actor.is_alive:
                        actor.destroy()
                        destroyed_count += 1
                    else:
                        already_destroyed_count += 1
                except RuntimeError as e:
                    # Actor already destroyed by CARLA
                    if "destroyed actor" in str(e).lower():
                        already_destroyed_count += 1
                    else:
                        print(f"   ⚠️ Unexpected error: {e}")
                except Exception as e:
                    # Any other error - just skip
                    pass
            
            self.spawned_actors.clear()
            print(f"   ✓ Destroyed: {destroyed_count}, Already gone: {already_destroyed_count}")
            print("   ✓ Cleanup complete")
        
    def setup_control_parameters(self):
        self.target_speed = 10.0  # km/h
        self.max_steer = STEER_LIMIT
        self.lanes_detected_history = deque(maxlen=10)
        self.lane_center_history = deque(maxlen=3)
        self.steering_history = deque(maxlen=4)
        self.gradual_stop_active = False
        self.gradual_stop_rate = 0.1
        # fallback controller params (kept)
        self.steer_gain = 1.2
        self.steer_damping = 0.1
        self.min_lanes_for_driving = 1
        self.lane_loss_threshold = 5
        self.startup_frames = 5
        self.frame_count = 0
        # Medium-article control state
        self.pid = PID(PID_KP, PID_KI, PID_KD, i_limit=PID_I_LIM, rate_limit=STEER_RATE, out_limit=STEER_LIMIT)
        self.err_ema = None
        self.lanes_ok_window = deque(maxlen=MISS_WINDOW)
        self.last_coeff_left  = None
        self.last_coeff_right = None

    def setup_bev(self):
        self.bev_h, self.bev_w = 400, 300
        src_pts = np.float32([
            [self.img_w * 0.15, self.img_h * 0.9],
            [self.img_w * 0.85, self.img_h * 0.9],
            [self.img_w * 0.55, self.img_h * 0.6],
            [self.img_w * 0.45, self.img_h * 0.6]
        ])
        dst_pts = np.float32([
            [50, self.bev_h - 50],
            [self.bev_w - 50, self.bev_h - 50],
            [self.bev_w - 50, 50],
            [50, 50]
        ])
        self.M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.px_to_m_y = 25.0 / (self.bev_h - 100)
        self.px_to_m_x = LANE_WIDTH_M / max(20.0, 0.4*(self.bev_w-100))

    def setup_roi_initial(self):
        print("\n🔎 Checking for saved ROI points...")
        self.roi_selector.load_from_csv()
    
    def _camera_callback(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self.camera_data = array
    
    # ========= Ultra-Fast lane detection (unchanged) =========
    def detect_lanes(self, img):
        if img is None:
            return None, 0, None, []
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.img_transforms(img_pil).unsqueeze(0).cuda()
        with torch.no_grad():
            out = self.net(img_tensor)
        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.cfg.griding_num] = 0
        out_j = loc
        lanes = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                lane_points = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        x_coord = int(out_j[k, i] * col_sample_w * self.img_w / 800) - 1
                        y_coord = int(self.img_h * (self.row_anchor[self.cls_num_per_lane-1-k]/288)) - 1
                        x_coord = max(0, min(x_coord, self.img_w-1))
                        y_coord = max(0, min(y_coord, self.img_h-1))
                        lane_points.append([x_coord, y_coord])
                if len(lane_points) > 1:
                    lanes.append(lane_points)
        filtered_lanes = self.roi_selector.filter_lanes_by_roi(lanes, img.shape)
        lane_center = self.calculate_lane_center(filtered_lanes)  # HUD only
        return out_j, len(filtered_lanes), lane_center, filtered_lanes
    
    def calculate_lane_center(self, lanes):
        if len(lanes) == 0:
            return None
        center_x = self.img_w // 2
        if len(lanes) == 1:
            lane = lanes[0]
            bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:3]
            if len(bottom_points) > 0:
                avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                if avg_x < center_x * 0.3:  lane_offset = 250
                elif avg_x > center_x * 1.7: lane_offset = -250
                else:                        lane_offset = 200 if avg_x < center_x else -200
                estimated_center = avg_x + lane_offset
                estimated_center = max(center_x * 0.5, min(center_x * 1.5, estimated_center))
                return int(estimated_center)
        left_lanes, right_lanes = [], []
        for lane in lanes:
            bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:3]
            if bottom_points:
                avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                bottom_point = bottom_points[0]
                if avg_x < center_x: left_lanes.append(bottom_point)
                else:                right_lanes.append(bottom_point)
        if len(left_lanes) > 0 and len(right_lanes) > 0:
            closest_left = max(left_lanes, key=lambda p: p[0])
            closest_right = min(right_lanes, key=lambda p: p[0])
            return (closest_left[0] + closest_right[0]) // 2
        elif len(left_lanes) > 0:
            return max(left_lanes, key=lambda p: p[0])[0] + 180
        elif len(right_lanes) > 0:
            return min(right_lanes, key=lambda p: p[0])[0] - 180
        return None

    # ========= BEV + two-stage fits =========
    def warp_to_bev(self, lanes):
        bev_sets = []
        for lane in lanes:
            pts = np.array(lane, dtype=np.float32).reshape(-1,1,2)
            w = cv2.perspectiveTransform(pts, self.M_bev).reshape(-1,2)
            m = (w[:,0]>=0)&(w[:,0]<self.bev_w)&(w[:,1]>=0)&(w[:,1]<self.bev_h)
            w = w[m]
            if len(w) >= 6:
                bev_sets.append(w)
        return bev_sets

    @staticmethod
    def linear_prefit(x,y):
        A = np.stack([y, np.ones_like(y)], axis=1)
        try:
            (a,b), *_ = np.linalg.lstsq(A, x, rcond=None)
            return a,b
        except: return None

    @staticmethod
    def quad_fit_x_of_y(x,y):
        A = np.stack([y**2, y, np.ones_like(y)], axis=1)
        try:
            (Q2,Q1,Q0), *_ = np.linalg.lstsq(A, x, rcond=None)
            return Q2,Q1,Q0
        except: return None

    def fit_lane_curve(self, wpts):
        if wpts is None or len(wpts) < 12:
            return None
        y = wpts[:,1].astype(np.float32); x = wpts[:,0].astype(np.float32)
        ab = self.linear_prefit(x,y)
        if ab is None: return None
        a,b = ab
        x_lin = a*y + b
        resid = np.abs(x - x_lin)
        thr = max(4.0, 2.0*np.median(resid))
        inl = resid < thr
        if inl.sum() < 10:
            inl = resid < (thr*1.5)
        return self.quad_fit_x_of_y(x[inl], y[inl])

    def update_px_to_m_x(self, coeff_left, coeff_right, y_eval):
        def x_at(c): return c[0]*y_eval*y_eval + c[1]*y_eval + c[2]
        if coeff_left is not None and coeff_right is not None:
            xl = x_at(coeff_left); xr = x_at(coeff_right)
            gap_px = abs(xr - xl)
            if gap_px > 5:
                self.px_to_m_x = LANE_WIDTH_M / gap_px

    def lateral_error_m(self, coeff_left, coeff_right, y_eval):
        def x_at(c): return c[0]*y_eval*y_eval + c[1]*y_eval + c[2]
        car_cx = self.bev_w * 0.5
        if coeff_left is None and coeff_right is None:
            return None
        if coeff_left is not None and coeff_right is not None:
            lane_cx = 0.5*(x_at(coeff_left) + x_at(coeff_right))
        elif coeff_left is not None:
            lane_cx = x_at(coeff_left) + (LANE_WIDTH_M/self.px_to_m_x)*0.5
        else:
            lane_cx = x_at(coeff_right) - (LANE_WIDTH_M/self.px_to_m_x)*0.5
        offset_px = lane_cx - car_cx
        return float(offset_px * self.px_to_m_x)

    # ========= Visualization =========
    def create_birds_eye_view(self, img, lanes):
        bev_height, bev_width = self.bev_h, self.bev_w
        bev_img = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
        M = self.M_bev
        for lane in lanes:
            if len(lane) > 1:
                lane_pts = np.array(lane, dtype=np.float32).reshape(-1, 1, 2)
                transformed_pts = cv2.perspectiveTransform(lane_pts, M)
                transformed_pts = transformed_pts.reshape(-1, 2).astype(np.int32)
                for pt in transformed_pts:
                    if 0 <= pt[0] < bev_width and 0 <= pt[1] < bev_height:
                        cv2.circle(bev_img, tuple(pt), 3, (255, 255, 255), -1)
                for i in range(len(transformed_pts) - 1):
                    pt1 = tuple(transformed_pts[i]); pt2 = tuple(transformed_pts[i+1])
                    if (0 <= pt1[0] < bev_width and 0 <= pt1[1] < bev_height and
                        0 <= pt2[0] < bev_width and 0 <= pt2[1] < bev_height):
                        cv2.line(bev_img, pt1, pt2, (255, 255, 255), 2)
        vehicle_pos = (bev_width // 2, bev_height - 30)
        cv2.circle(bev_img, vehicle_pos, 8, (0, 255, 0), -1)
        cv2.putText(bev_img, "Bird's Eye View", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return bev_img
    
    # --- MODIFIED: Added 'yolo_detections' and 'lane_bounds' arguments ---
    def visualize_lanes(self, img, out_j, lanes_detected, lane_center, 
                        filtered_lanes, yolo_detections=None, lane_bounds=None,
                        show_lane_mask=False):
        if img is None:
            return img, None
        
        vis = img.copy()

        # --- NEW: Visualize lane mask if requested ---
        if show_lane_mask and self.yolo_lane_filter.lane_mask is not None:
            vis = self.yolo_lane_filter.visualize_lane_mask(vis, alpha=0.2)

        # --- Draw YOLO detections first (so lanes overlay on top) ---
        if yolo_detections is not None and self.yolo_detector:
            vis = self.yolo_detector.visualize_detections(vis, yolo_detections, lane_bounds)

        if len(self.roi_selector.roi_points) == 3:
            pts = np.array(self.roi_selector.roi_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, vis, 0.9, 0, vis)
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        for i, lane in enumerate(filtered_lanes):
            color = colors[i % len(colors)]
            for point in lane:
                cv2.circle(vis, tuple(point), 3, color, -1)
            if len(lane) > 1:
                points = np.array(lane, dtype=np.int32)
                cv2.polylines(vis, [points], False, color, 2)
        if lane_center is not None:
            cv2.circle(vis, (int(lane_center), self.img_h - 50), 10, (255, 0, 255), -1)
            cv2.line(vis, (int(lane_center), self.img_h - 100), 
                     (int(lane_center), self.img_h), (255, 0, 255), 3)
        center_x = self.img_w // 2
        cv2.line(vis, (center_x, 0), (center_x, self.img_h), (128, 128, 128), 2)

        # HUD
        speed = self.get_vehicle_speed()
        status_color = (0, 255, 0) if (self.mode=='manual' or not self.gradual_stop_active) else (0, 0, 255)
        status_text = "MANUAL" if self.mode=='manual' else ("DRIVING" if not self.gradual_stop_active else "STOPPING")
        cv2.putText(vis, f"Mode: {status_text}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis, f"Lanes: {lanes_detected}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(vis, f"Speed: {speed:.1f} km/h", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
        # --- Display obstacle count ---
        if yolo_detections is not None:
            obstacle_count = len(yolo_detections)
            dangerous_count = sum(1 for d in yolo_detections if d['is_dangerous'])
            cv2.putText(vis, f"Objects: {obstacle_count} (Danger: {dangerous_count})", (10, 114),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(vis, "ROI: Active" if len(self.roi_selector.roi_points) == 3 else "ROI: Inactive", 
                        (10, 142), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(vis, "ROI: Active" if len(self.roi_selector.roi_points) == 3 else "ROI: Inactive", 
                        (10, 114), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.putText(vis, "Keys: [M] Manual  [L] Auto  [W/S/A/D]  [Space] Brake  [Q] Quit",
                    (10, self.img_h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230,230,230), 2)

        bev_img = self.create_birds_eye_view(img, filtered_lanes)
        return vis, bev_img
    
    # ========= Safety =========
    def update_safety_state(self, lanes_detected):
        self.lanes_detected_history.append(lanes_detected)
        recent_detections = list(self.lanes_detected_history)[-self.lane_loss_threshold:]
        sufficient_lanes = [d >= self.min_lanes_for_driving for d in recent_detections]
        if len(recent_detections) >= self.lane_loss_threshold:
            if not any(sufficient_lanes):
                if not self.gradual_stop_active:
                    print("⚠️ WARNING: Insufficient lane detection. Initiating gradual stop.")
                    self.gradual_stop_active = True
            else:
                if self.gradual_stop_active:
                    print("✓ Lanes re-detected. Resuming normal driving.")
                    self.gradual_stop_active = False
    
    def get_vehicle_speed(self):
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)

    def get_signed_speed_kmh(self):
        """+forward, -reverse relative to car heading"""
        vel = self.vehicle.get_velocity()
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        speed_ms = vel.x*fwd.x + vel.y*fwd.y + vel.z*fwd.z
        return speed_ms * 3.6

    # ========= Fallback pixel-center =========
    def calculate_steering(self, lane_center):
        if lane_center is None:
            if len(self.steering_history) > 0:
                return self.steering_history[-1] * 0.8
            return 0.0
        image_center = self.img_w // 2
        pixel_error = lane_center - image_center
        normalized_error = pixel_error / image_center
        steering = -normalized_error * self.steer_gain * STEER_SIGN
        if self.frame_count < self.startup_frames:
            startup_factor = min(1.0, (self.frame_count + 1) / self.startup_frames)
            steering *= startup_factor
        if len(self.steering_history) > 0:
            steering_change = steering - self.steering_history[-1]
            steering = steering - (steering_change * self.steer_damping)
        steering = max(-self.max_steer, min(self.max_steer, steering))
        self.steering_history.append(steering)
        self.frame_count += 1
        return steering

    # ========= PID steering from meters =========
    def pid_steer_from_error(self, err_m):
        now = time.time()
        last = self.steering_history[-1] if len(self.steering_history) else None
        steer = self.pid.step(err_m, now, last)
        steer = max(-self.max_steer, min(self.max_steer, steer))
        self.steering_history.append(steer)
        return steer

    # ========= Manual control helpers (with launch & reverse) =========
    def process_manual_keys(self, key, signed_speed_kmh):
        # Steering
        if key == ord('a'):
            self.manual_steer = max(-STEER_LIMIT, self.manual_steer - MAN_STEER_STEP)
        elif key == ord('d'):
            self.manual_steer = min( STEER_LIMIT, self.manual_steer + MAN_STEER_STEP)
        else:
            self.manual_steer *= MAN_STEER_DECAY

        # Throttle / Brake / Reverse logic
        near_stop = abs(signed_speed_kmh) < 0.5

        if key == ord('w'):
            # Force forward; release brake; give launch if stopped
            self.manual_reverse = False
            self.manual_brake = 0.0
            if near_stop and self.manual_throttle < 0.25:
                self.manual_throttle = 0.25
            else:
                self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)

        elif key == ord('s'):
            if signed_speed_kmh > 1.0:
                # Moving forward: brake
                self.manual_throttle = 0.0
                self.manual_brake = min(MAN_MAX_BRAKE, self.manual_brake + MAN_BRAKE_STEP)
            else:
                # Stopped or rolling backward: engage reverse and throttle
                self.manual_reverse = True
                self.manual_brake = 0.0
                if near_stop and self.manual_throttle < 0.25:
                    self.manual_throttle = 0.25
                else:
                    self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)

        elif key == 32:  # Space = hard brake
            self.manual_throttle = 0.0
            self.manual_brake = 1.0

        else:
            # No throttle/brake keys: decay
            self.manual_throttle *= MAN_THR_DECAY
            self.manual_brake    *= MAN_BRAKE_DECAY

        # Brake wins over throttle if both > 0
        if self.manual_brake > 0.1:
            self.manual_throttle = 0.0

    def apply_manual_control(self):
        ctrl = carla.VehicleControl()
        ctrl.throttle   = float(self.manual_throttle)
        ctrl.brake      = float(self.manual_brake)
        ctrl.steer      = float(self.manual_steer)
        ctrl.hand_brake = False
        ctrl.reverse    = bool(self.manual_reverse)
        self.vehicle.apply_control(ctrl)
        return ctrl

    # ========= ROI choice on switching to auto =========
    def handle_roi_choice_when_auto(self, current_frame_bgr):
        has_existing = self.roi_selector.load_from_csv()
        
        # Auto-accept existing ROI if available
        if has_existing:
            print("→ Auto-using EXISTING ROI points (auto mode).")
            self.awaiting_roi_choice = False
            return
        
        prompt = current_frame_bgr.copy()
        cv2.rectangle(prompt, (20, 20), (self.img_w-20, 140), (0,0,0), -1)
        cv2.putText(prompt, "Autonomous mode: Choose ROI   [1]=Existing   [2]=Mark New   [Esc]=Skip",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.imshow('CARLA Lane Detection with ROI', prompt)
        self.awaiting_roi_choice = True
        self.roi_choice_deadline = time.time() + 5.0
        self.roi_choice_has_existing = has_existing

    def maybe_consume_roi_choice_key(self, key):
        if not self.awaiting_roi_choice:
            return
        if key == ord('1') and self.roi_choice_has_existing:
            print("→ Using EXISTING ROI points.")
            self.awaiting_roi_choice = False
        elif key == ord('2'):
            print("→ Mark NEW ROI points.")
            if self.camera_data is not None:
                ok = self.roi_selector.select_roi(self.camera_data)
                print("ROI updated." if ok else "ROI selection cancelled; previous ROI (if any) kept.")
            self.awaiting_roi_choice = False
        elif key == 27:  # ESC
            print("→ Skipping ROI selection; running with current ROI.")
            self.awaiting_roi_choice = False
        elif time.time() > getattr(self, 'roi_choice_deadline', 0):
            print("→ ROI choice timeout; continuing with current ROI.")
            self.awaiting_roi_choice = False

    # ========= Main loop =========
    def run(self, duration=300, save_video=True):
        print("\n" + "="*70)
        print("  CONTROLS:")
        print("  [L] = Autonomous Lane Keeping Mode (AUTO)")
        print("  [M] = Manual Driving Mode")
        print("  [W/A/S/D] = Manual throttle/brake/steering")
        print("  [Space] = Brake")
        print("  [V] = Toggle lane mask visualization")
        print("  [Q] = Quit")
        print("="*70)
        print("\n⚠️  Vehicle starts in MANUAL mode")
        print("⚠️  Press [L] to enable AUTONOMOUS lane keeping")
        print("⚠️  Press [W] to start driving manually\n")
        
        # Wait for camera to initialize
        print("⏳ Waiting for camera data...")
        wait_start = time.time()
        while self.camera_data is None:
            if time.time() - wait_start > 10:
                print("❌ Error: Camera timeout. Check CARLA server.")
                return
            time.sleep(0.1)
            self.world.tick()
        print("✓ Camera ready!")
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            video_writer = cv2.VideoWriter('carla_lane_driving.avi', fourcc, 20, (self.img_w, self.img_h))
            bev_writer   = cv2.VideoWriter('carla_bev.avi',       fourcc, 20, (self.bev_w, self.bev_h))
        
        start_time = time.time()
        frame_count = 0
        
        # --- Variables for YOLO detections ---
        yolo_detections = []
        lane_bounds = None
        show_lane_mask_debug = False  # Set to True to visualize lane mask

        try:
            while time.time() - start_time < duration:
                if self.camera_data is None:
                    time.sleep(0.01)
                    continue

                key = cv2.waitKey(1) & 0xFF

                # Mode switches
                if key == ord('m'):
                    self.mode = 'manual'
                    print("⚙️  Switched to MANUAL mode. Use W/S/A/D (Space=brake). Press L for Auto.")
                if key == ord('l'):
                    self.mode = 'auto'
                    print("🤖 Switched to AUTONOMOUS mode.")
                    self.handle_roi_choice_when_auto(self.camera_data)
                
                # --- NEW: Toggle lane mask visualization ---
                if key == ord('v'):
                    show_lane_mask_debug = not show_lane_mask_debug
                    print(f"Lane mask visualization: {'ON' if show_lane_mask_debug else 'OFF'}")

                # If awaiting ROI choice (after pressing L), process choice keys
                self.maybe_consume_roi_choice_key(key)

                # MANUAL MODE
                if self.mode == 'manual':
                    signed_speed = self.get_signed_speed_kmh()
                    self.process_manual_keys(key, signed_speed)
                    ctrl = self.apply_manual_control()

                    # For display only (still show detections)
                    lane_data, lanes_detected, lane_center, filtered_lanes = self.detect_lanes(self.camera_data)
                    
                    # --- NEW: Create lane mask and filter YOLO detections ---
                    if self.yolo_detector:
                        try:
                            # Create lane mask from detected lanes with FORWARD EXTENSION
                            self.yolo_lane_filter.create_lane_mask_from_lanes(
                                filtered_lanes, 
                                expansion_width=50,
                                forward_extension=300  # Extend 300 pixels forward to catch distant objects
                            )
                            
                            # Detect all objects
                            all_detections, _ = self.yolo_detector.detect_and_calculate_distance(
                                self.camera_data
                            )
                            
                            # Filter to only lane objects
                            yolo_detections = self.yolo_lane_filter.filter_detections_by_lane(
                                all_detections, overlap_threshold=0.3
                            )
                            
                            lane_bounds = self.yolo_lane_filter.get_lane_bounds()
                            
                            if frame_count % 30 == 0:  # Debug info
                                print(f"[MANUAL] All objects: {len(all_detections)}, In-lane: {len(yolo_detections)}")
                            
                        except Exception as e:
                            print(f"Error in YOLO detection (manual): {e}")
                            yolo_detections = []
                            lane_bounds = None
                    
                    vis_image, bev_image = self.visualize_lanes(
                        self.camera_data, lane_data, lanes_detected, lane_center, 
                        filtered_lanes, yolo_detections, lane_bounds, show_lane_mask_debug
                    )
                    if save_video:
                        if vis_image is not None: video_writer.write(vis_image)
                        if bev_image is not None: bev_writer.write(bev_image)
                    if vis_image is not None: cv2.imshow('CARLA Lane Detection with ROI', vis_image)
                    if bev_image is not None: cv2.imshow('Bird\'s Eye View', bev_image)

                # AUTONOMOUS MODE
                else:
                    # --- Detect lanes (original + ROI) ---
                    lane_data, lanes_detected, lane_center, filtered_lanes = self.detect_lanes(self.camera_data)

                    # --- BEV + two-stage fit (article logic) ---
                    bev_sets = self.warp_to_bev(filtered_lanes)
                    left_set = right_set = None
                    if len(bev_sets) == 1:
                        if bev_sets[0][:,0].mean() < self.bev_w*0.5: left_set = bev_sets[0]
                        else: right_set = bev_sets[0]
                    elif len(bev_sets) >= 2:
                        s = sorted(bev_sets, key=lambda a: a[:,0].mean())
                        left_set, right_set = s[0], s[-1]
                    coeff_left  = self.fit_lane_curve(left_set)  if left_set  is not None else None
                    coeff_right = self.fit_lane_curve(right_set) if right_set is not None else None
                    if coeff_left  is None: coeff_left  = self.last_coeff_left
                    if coeff_right is None: coeff_right = self.last_coeff_right

                    y_eval = self.bev_h - LOOK_Y_OFFSET
                    if coeff_left is not None or coeff_right is not None:
                        self.update_px_to_m_x(coeff_left, coeff_right, y_eval)
                        err_m_raw = self.lateral_error_m(coeff_left, coeff_right, y_eval)
                    else:
                        err_m_raw = None

                    if err_m_raw is not None:
                        if self.err_ema is None: self.err_ema = err_m_raw
                        else: self.err_ema = EMA_ALPHA*err_m_raw + (1-EMA_ALPHA)*self.err_ema

                    # --- NEW: Lane-filtered YOLO Detection ---
                    obstacle_detected = False
                    nearest_obstacle = None
                    
                    if self.yolo_detector:
                        try:
                            # Create lane mask with FORWARD EXTENSION
                            self.yolo_lane_filter.create_lane_mask_from_lanes(
                                filtered_lanes, 
                                expansion_width=50,
                                forward_extension=300  # Extend 300 pixels forward to catch distant objects
                            )
                            
                            # Detect all objects
                            all_detections, _ = self.yolo_detector.detect_and_calculate_distance(
                                self.camera_data
                            )
                            
                            # Filter to only in-lane objects
                            yolo_detections = self.yolo_lane_filter.filter_detections_by_lane(
                                all_detections, overlap_threshold=0.3
                            )
                            
                            # Check for obstacles using ONLY filtered detections
                            obstacle_detected, nearest_obstacle, lane_bounds = self.yolo_detector.should_stop(
                                yolo_detections, filtered_lanes, self.img_w
                            )
                            
                            if obstacle_detected and nearest_obstacle:
                                if frame_count % 10 == 0:
                                    in_lane_flag = "✓" if nearest_obstacle.get('in_lane', False) else "?"
                                    print(f"⚠️ {in_lane_flag} Obstacle: {nearest_obstacle['class']} at {nearest_obstacle['distance']}m")
                        
                            if frame_count % 30 == 0:
                                print(f"[AUTO] All: {len(all_detections)}, In-lane: {len(yolo_detections)}, Danger: {obstacle_detected}")
                        
                        except Exception as e:
                            print(f"Error in YOLO detection: {e}")
                            yolo_detections = []
                            lane_bounds = None
                    
                    # --- Stop Logic ---
                    lanes_ok = (coeff_left is not None) or (coeff_right is not None)
                    self.lanes_ok_window.append(1 if lanes_ok else 0)
                    
                    is_stopping_for_lanes = len(self.lanes_ok_window)==self.lanes_ok_window.maxlen and sum(self.lanes_ok_window) <= MISS_THRESH
                    
                    if is_stopping_for_lanes:
                        if not self.gradual_stop_active: print("⚠️ Stopping: Lane loss")
                        self.gradual_stop_active = True
                    elif obstacle_detected:
                        if not self.gradual_stop_active: 
                            dist_str = f"{nearest_obstacle['distance']}m" if nearest_obstacle['distance'] else "unknown"
                            print(f"⚠️ Stopping: {nearest_obstacle['class']} at {dist_str}")
                        self.gradual_stop_active = True
                    elif lanes_ok:
                        if self.gradual_stop_active: print("✓ Resuming: All clear")
                        self.gradual_stop_active = False

                    # ...existing control code...
                    if coeff_left  is not None: self.last_coeff_left  = coeff_left
                    if coeff_right is not None: self.last_coeff_right = coeff_right

                    control = carla.VehicleControl()
                    current_speed = self.get_vehicle_speed()
                    if self.gradual_stop_active:
                        if current_speed > 1.0:
                            control.throttle = 0.0
                            control.brake = min(1.0, self.gradual_stop_rate)
                            control.steer = self.steering_history[-1]*0.8 if len(self.steering_history) else 0.0
                        else:
                            control.throttle = 0.0; control.brake = 1.0; control.steer = 0.0
                    else:
                        if current_speed < self.target_speed - 5:   control.throttle, control.brake = 0.7, 0.0
                        elif current_speed < self.target_speed:     control.throttle, control.brake = 0.4, 0.0
                        elif current_speed > self.target_speed + 5: control.throttle, control.brake = 0.0, 0.3
                        else:                                        control.throttle, control.brake = 0.2, 0.0
                        
                        if self.err_ema is not None:
                            control.steer = self.pid_steer_from_error(self.err_ema)
                        else:
                            if lane_center is not None:
                                self.lane_center_history.append(lane_center)
                                if len(self.lane_center_history) >= 2:
                                    smoothed_center = sum(self.lane_center_history)/len(self.lane_center_history)
                                else:
                                    smoothed_center = lane_center
                                control.steer = self.calculate_steering(smoothed_center)
                            else:
                                control.steer = self.steering_history[-1]*0.9 if len(self.steering_history) else 0.0
                    self.vehicle.apply_control(control)

                    # --- Visualization with YOLO detections ---
                    vis_image, bev_image = self.visualize_lanes(
                        self.camera_data, lane_data, lanes_detected, lane_center, 
                        filtered_lanes, yolo_detections, lane_bounds, show_lane_mask_debug
                    )
                    if save_video:
                        if vis_image is not None: video_writer.write(vis_image)
                        if bev_image is not None: bev_writer.write(bev_image)
                    if vis_image is not None: cv2.imshow('CARLA Lane Detection with ROI', vis_image)
                    if bev_image is not None: cv2.imshow('Bird\'s Eye View', bev_image)

                if key == ord('q'):
                    break

                if frame_count % 100 == 0:
                    speed = self.get_vehicle_speed()
                    print(f"[{frame_count}] mode={self.mode} speed={speed:.1f} km/h")
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n⏹ Interrupted by user")
        
        finally:
            print("🧹 Cleaning up...")
            
            # Close video writers first
            if save_video:
                try:
                    if 'video_writer' in locals() and video_writer is not None:
                        video_writer.release()
                    if 'bev_writer' in locals() and bev_writer is not None:
                        bev_writer.release()
                except:
                    pass
            
            # Close OpenCV windows
            try:
                cv2.destroyAllWindows()
            except:
                pass
            
            # Stop vehicle safely
            try:
                if hasattr(self, 'vehicle') and self.vehicle is not None:
                    control = carla.VehicleControl()
                    control.throttle = 0.0
                    control.brake = 1.0
                    control.steer = 0.0
                    self.vehicle.apply_control(control)
            except:
                pass
            
            # Cleanup spawned obstacles (with error handling)
            try:
                self.cleanup_spawned_actors()
            except Exception as e:
                print(f"   ⚠️ Error during actor cleanup: {e}")
            
            # Destroy sensors and vehicle
            try:
                if hasattr(self, 'camera') and self.camera is not None:
                    self.camera.destroy()
            except:
                pass
            
            try:
                if hasattr(self, 'vehicle') and self.vehicle is not None:
                    self.vehicle.destroy()
            except:
                pass
            
            print("✓ Cleanup completed")


def main():
    controller = None
    try:
        print("\n" + "="*60)
        print("CARLA Lane Detection with YOLO Obstacle Detection")
        print("="*60)
        
        controller = LaneDetectionController(
            cfg_path="configs/tusimple.py",
            model_path="tusimple_18.pth"
        )
        
        print("\n✓ Initialization complete!")
        print("="*60)
        
        controller.run(duration=300, save_video=True)
        
    except KeyboardInterrupt:
        print("\n\n⏹ Interrupted by user (Ctrl+C)")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup even if error occurs during init
        if controller is not None:
            try:
                controller.cleanup_spawned_actors()
            except:
                pass

if __name__ == "__main__":
    main()

