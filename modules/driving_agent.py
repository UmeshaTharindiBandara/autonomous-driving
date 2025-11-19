"""
Driving Agent Module
Decision-making and control for autonomous driving with manual mode support
"""

import carla
import cv2
import time
from collections import deque
from typing import Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.lane_detector import LaneDetector
from modules.obstacle_detector import ObstacleDetector
from modules.traffic_light_detector import TrafficLightDetector
from core.pid_controller import PIDController
from core.carla_spawner import CarlaSpawner

# UPDATED: Import from detection module
from detection.yolo_lane_filter import YOLOLaneFilter

# Control parameters - Tuned for smooth steering at 15 km/h
PID_KP, PID_KI, PID_KD = 0.45, 0.015, 0.28  # Lower P, higher D for smoother response
STEER_LIMIT = 0.25
TARGET_SPEED = 10.0  # km/h

# Manual driving parameters
MAN_STEER_STEP = 0.04
MAN_STEER_DECAY = 0.90
MAN_THR_STEP = 0.05
MAN_THR_DECAY = 0.96
MAN_BRAKE_STEP = 0.08
MAN_BRAKE_DECAY = 0.90
MAN_MAX_THR = 0.85
MAN_MAX_BRAKE = 1.00


class DrivingAgent:
    """Autonomous driving agent with lane keeping and obstacle avoidance"""
    
    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        """Initialize driving agent"""
        self.world = world
        self.vehicle = vehicle
        
        # Initialize modules
        self.lane_detector = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        
        # NEW: Use the working YOLOLaneFilter with triangular ROI
        self.yolo_lane_filter = YOLOLaneFilter(
            img_width=self.lane_detector.img_w,
            img_height=self.lane_detector.img_h
        )
        # Enable triangular ROI to prevent detecting adjacent lane objects
        self.yolo_lane_filter.use_triangular_roi = True
        # Adjust ROI width (20% = conservative, 25% = moderate, 30% = wide)
        self.yolo_lane_filter.fixed_lane_width_ratio = 0.22
        
        # Traffic light detector
        self.traffic_light_detector = TrafficLightDetector(
            model_path="traffic_light.pt"
        )
        self.traffic_light_enabled = self.traffic_light_detector.is_available()
        
        # Optional: Customize traffic light ROI region if needed
        # Uncomment and adjust these values to change detection area:
        # self.traffic_light_detector.roi_top_ratio = 0.0    # Top: 0% from top
        # self.traffic_light_detector.roi_bottom_ratio = 0.4   # Bottom: 40% from top
        # self.traffic_light_detector.roi_left_ratio = 0.4     # Left: 40% from left
        # self.traffic_light_detector.roi_right_ratio = 0.8    # Right: 75% from left
        # self.traffic_light_detector.zoom_scale = 1.75        # Zoom factor

        # Calibrate obstacle detector
        self.obstacle_detector.calibrate_camera(
            self.lane_detector.img_w, 
            self.lane_detector.img_h, 
            fov_degrees=90
        )
        
        # Initialize PID controller for steering
        self.pid_controller = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            i_limit=0.5, rate_limit=0.025, out_limit=STEER_LIMIT, sign=-1.0  # Tighter rate limit
        )
        
        # State variables
        self.mode = 'manual'  # 'manual' or 'auto'
        # self.controller_type = 'curvature'  # 'curvature' or 'pid'
        self.controller_type = 'pid'
        self.target_speed = TARGET_SPEED
        self.gradual_stop_active = False
        self.gradual_stop_rate = 0.1
        self.steering_history = deque(maxlen=4)
        self.frame_count = 0
        
        # Steering smoothing - EMA filter for lateral error
        self.lateral_error_ema = None
        self.lateral_error_alpha = 0.3  # Lower = smoother (0.3 = 30% new, 70% old)
        
        # Manual control state
        self.manual_throttle = 0.0
        self.manual_brake = 0.0
        self.manual_steer = 0.0
        self.manual_reverse = False
        
        # ROI selection
        self.awaiting_roi_choice = False
        self.roi_choice_deadline = 0
        self.roi_choice_has_existing = False
        
        # Spawner for traffic
        self.spawner = None
        
        # Visualization flags
        self.show_lane_mask = False  # NEW: Toggle with V key
        
        # Load ROI initially
        print("🔎 Checking for saved ROI...")
        self.lane_detector.roi_selector.load_from_csv()
        
        print("✓ Driving Agent initialized")
        print(f"  Mode: {self.mode.upper()}")
        print(f"  Object Detection ROI: TRIANGULAR (prevents adjacent lane detection)")
        if self.traffic_light_enabled:
            print(f"  Traffic Light Detection: ENABLED")
        else:
            print(f"  Traffic Light Detection: DISABLED")
    
    def set_mode(self, mode: str):
        """Switch between manual and auto mode"""
        if mode in ['manual', 'auto']:
            self.mode = mode
            print(f"⚙️  Switched to {mode.upper()} mode")
            
            if mode == 'auto':
                # Reset manual controls
                self.manual_throttle = 0.0
                self.manual_brake = 0.0
                self.manual_steer = 0.0
                self.manual_reverse = False
        else:
            print(f"⚠️ Invalid mode: {mode}")
    
    def handle_roi_choice_when_auto(self, current_frame):
        """Handle ROI selection when switching to auto mode"""
        has_existing = self.lane_detector.roi_selector.load_from_csv()
        
        if has_existing:
            print("→ Auto-using EXISTING ROI points")
            self.awaiting_roi_choice = False
            return
        
        # Show prompt
        prompt = current_frame.copy()
        cv2.rectangle(prompt, (20, 20), (1260, 140), (0, 0, 0), -1)
        cv2.putText(prompt, "Autonomous mode: Choose ROI   [1]=Existing   [2]=Mark New   [Esc]=Skip",
                   (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.imshow('Autonomous Driving - Modular', prompt)
        
        self.awaiting_roi_choice = True
        self.roi_choice_deadline = time.time() + 5.0
        self.roi_choice_has_existing = has_existing
    
    def process_roi_choice_key(self, key):
        """Process ROI choice keys"""
        if not self.awaiting_roi_choice:
            return
        
        if key == ord('1') and self.roi_choice_has_existing:
            print("→ Using EXISTING ROI points")
            self.awaiting_roi_choice = False
        elif key == ord('2'):
            print("→ Mark NEW ROI points")
            # Get current camera frame
            # Note: Need to pass frame from main loop
            self.awaiting_roi_choice = False
        elif key == 27:  # ESC
            print("→ Skipping ROI selection")
            self.awaiting_roi_choice = False
        elif time.time() > self.roi_choice_deadline:
            print("→ ROI choice timeout")
            self.awaiting_roi_choice = False
    
    def spawn_traffic(self, num_vehicles=10, num_pedestrians=0, num_static=3):
        """Spawn traffic obstacles"""
        self.spawner = CarlaSpawner(self.world)
        self.spawner.spawn_traffic_obstacles(num_vehicles, num_pedestrians, num_static)
    
    def process_manual_keys(self, key):
        """Process manual control keys"""
        signed_speed = self._get_signed_speed_kmh()
        near_stop = abs(signed_speed) < 0.5
        
        # Steering
        if key == ord('a'):
            self.manual_steer = max(-STEER_LIMIT, self.manual_steer - MAN_STEER_STEP)
        elif key == ord('d'):
            self.manual_steer = min(STEER_LIMIT, self.manual_steer + MAN_STEER_STEP)
        else:
            self.manual_steer *= MAN_STEER_DECAY
        
        # Throttle/Brake
        if key == ord('w'):
            self.manual_reverse = False
            self.manual_brake = 0.0
            if near_stop and self.manual_throttle < 0.25:
                self.manual_throttle = 0.25
            else:
                self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)
        
        elif key == ord('s'):
            if signed_speed > 1.0:
                self.manual_throttle = 0.0
                self.manual_brake = min(MAN_MAX_BRAKE, self.manual_brake + MAN_BRAKE_STEP)
            else:
                self.manual_reverse = True
                self.manual_brake = 0.0
                if near_stop and self.manual_throttle < 0.25:
                    self.manual_throttle = 0.25
                else:
                    self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)
        
        elif key == 32:  # Space
            self.manual_throttle = 0.0
            self.manual_brake = 1.0
        
        else:
            self.manual_throttle *= MAN_THR_DECAY
            self.manual_brake *= MAN_BRAKE_DECAY
        
        # Brake wins
        if self.manual_brake > 0.1:
            self.manual_throttle = 0.0
    
    def apply_manual_control(self) -> carla.VehicleControl:
        """Apply manual control"""
        control = carla.VehicleControl()
        control.throttle = float(self.manual_throttle)
        control.brake = float(self.manual_brake)
        control.steer = float(self.manual_steer)
        control.hand_brake = False
        control.reverse = bool(self.manual_reverse)
        return control
    
    def process_frame(self, image):
        """Process single frame and return control decision"""
        
        # Manual mode
        if self.mode == 'manual':
            lane_result = self.lane_detector.detect(image)
            all_detections, _ = self.obstacle_detector.detect(image)
            
            if lane_result:
                # CREATE LANE MASK with TRIANGULAR ROI
                # When only 1 lane detected: Uses fixed-width trapezoid (20-22% of image width)
                #   that narrows toward vanishing point - prevents detecting adjacent lane objects
                # When 2 lanes detected: Uses actual lane boundaries (more accurate)
                self.yolo_lane_filter.create_lane_mask_from_lanes(
                    lane_result['filtered_lanes'],
                    expansion_width=50,
                    forward_extension=300
                    # Uses default: max_vertical_extent_single=0.8, max_vertical_extent_dual=0.9
                )
                
                # Filter detections using the proper lane filter
                lane_detections = self.yolo_lane_filter.filter_detections_by_lane(
                    all_detections,
                    overlap_threshold=0.3
                )
            else:
                lane_detections = []
            
            control = self.apply_manual_control()
            
            return {
                'control': control,
                'lane_data': lane_result,
                'obstacle_data': {
                    'all_detections': all_detections,
                    'lane_detections': lane_detections,
                    'nearest_obstacle': None,
                    'should_stop': False
                },
                'decision': 'MANUAL CONTROL'
            }
        
        # Auto mode
        lane_result = self.lane_detector.detect(image)
        if lane_result is None:
            return self._emergency_stop()
        
        lateral_error = self.lane_detector.compute_lateral_error(lane_result['filtered_lanes'])
        
        all_detections, _ = self.obstacle_detector.detect(image)
        
        # CREATE LANE MASK with TRIANGULAR ROI
        # Single lane: Fixed-width trapezoid prevents adjacent lane false positives
        # Dual lanes: Uses actual boundaries for accurate filtering
        self.yolo_lane_filter.create_lane_mask_from_lanes(
            lane_result['filtered_lanes'],
            expansion_width=50,
            forward_extension=300
            # Uses default: max_vertical_extent_single=0.8, max_vertical_extent_dual=0.9
        )
        
        # Filter using YOLOLaneFilter
        lane_detections = self.yolo_lane_filter.filter_detections_by_lane(
            all_detections,
            overlap_threshold=0.3
        )
        
        should_stop, nearest_obstacle = self.obstacle_detector.should_stop(lane_detections)
        
        lane_lost = self.lane_detector.is_lane_lost()
        
        control, decision = self._make_control_decision(
            lateral_error, should_stop, lane_lost, nearest_obstacle
        )
        
        return {
            'control': control,
            'lane_data': lane_result,
            'obstacle_data': {
                'all_detections': all_detections,
                'lane_detections': lane_detections,
                'nearest_obstacle': nearest_obstacle,
                'should_stop': should_stop
            },
            'decision': decision
        }
    
    def _make_control_decision(self, lateral_error: Optional[float], 
                               should_stop: bool, lane_lost: bool,
                               nearest_obstacle: Optional[Dict]) -> Tuple[carla.VehicleControl, str]:
        """Make control decision based on perception"""
        control = carla.VehicleControl()
        current_speed = self._get_vehicle_speed()
        
        # Determine if stopping
        if lane_lost:
            self.gradual_stop_active = True
            decision = "STOP: Lane loss"
        elif should_stop:
            self.gradual_stop_active = True
            if nearest_obstacle:
                dist = nearest_obstacle.get('distance', 'unknown')
                decision = f"STOP: {nearest_obstacle['class']} at {dist}m"
            else:
                decision = "STOP: Obstacle"
        else:
            self.gradual_stop_active = False
            decision = "DRIVE: Normal"
        
        # Apply control
        if self.gradual_stop_active:
            # Gradual stop
            if current_speed > 1.0:
                control.throttle = 0.0
                control.brake = min(1.0, self.gradual_stop_rate)
                control.steer = self.steering_history[-1] * 0.8 if self.steering_history else 0.0
            else:
                control.throttle = 0.0
                control.brake = 1.0
                control.steer = 0.0
        elif obstacle_action == 'slow':
            # Active slowdown: reduce speed significantly
            control.throttle = 0.0
            if current_speed > 20:
                control.brake = 0.5
            elif current_speed > 15:
                control.brake = 0.3
            else:
                control.brake = 0.15
            # Maintain steering with smoothed input
            if lateral_error is not None:
                # Apply EMA smoothing
                if self.lateral_error_ema is None:
                    self.lateral_error_ema = lateral_error
                else:
                    self.lateral_error_ema = (self.lateral_error_alpha * lateral_error + 
                                             (1 - self.lateral_error_alpha) * self.lateral_error_ema)
                smoothed_error = self.lateral_error_ema
                
                last_steer = self.steering_history[-1] if self.steering_history else None
                if self.controller_type == 'curvature':
                    control.steer = self.curv_controller.step(
                        lane_detector=self.lane_detector,
                        lateral_error_m=smoothed_error,
                        speed_kmh=current_speed,
                        last_out=last_steer
                    )
                else:
                    control.steer = self.pid_controller.step(smoothed_error, last_out=last_steer)
                self.steering_history.append(control.steer)
            else:
                control.steer = self.steering_history[-1] * 0.95 if self.steering_history else 0.0
        
        elif obstacle_action == 'cautious':
            # Cautious mode: gentle deceleration, reduce target speed
            reduced_target = min(self.target_speed * 0.6, 20.0)  # Max 20 km/h in cautious mode
            speed_err = reduced_target - current_speed
            
            if speed_err < -2:
                control.throttle = 0.0
                control.brake = 0.2
            elif speed_err < 0:
                control.throttle = 0.0
                control.brake = 0.0
            else:
                control.throttle = 0.2
                control.brake = 0.0
            
            # Maintain steering with smoothed input
            if lateral_error is not None:
                # Apply EMA smoothing
                if self.lateral_error_ema is None:
                    self.lateral_error_ema = lateral_error
                else:
                    self.lateral_error_ema = (self.lateral_error_alpha * lateral_error + 
                                             (1 - self.lateral_error_alpha) * self.lateral_error_ema)
                smoothed_error = self.lateral_error_ema
                
                last_steer = self.steering_history[-1] if self.steering_history else None
                if self.controller_type == 'curvature':
                    control.steer = self.curv_controller.step(
                        lane_detector=self.lane_detector,
                        lateral_error_m=smoothed_error,
                        speed_kmh=current_speed,
                        last_out=last_steer
                    )
                else:
                    control.steer = self.pid_controller.step(smoothed_error, last_out=last_steer)
                self.steering_history.append(control.steer)
            else:
                control.steer = self.steering_history[-1] * 0.95 if self.steering_history else 0.0
        else:
            # Normal driving
            # Adaptive target speed based on curvature (if available)
            kappa, kappa_cls = self.lane_detector.compute_centerline_curvature()
            # Updated speed policy:
            # straight: 30 km/h
            # gentle: 28 km/h
            # moderate: 26 km/h
            # sharp: 25 km/h
            # very_sharp: 18 km/h (tight bend safety)
            if kappa is not None and kappa_cls is not None:
                if kappa_cls == 'straight':
                    dyn_target = 15.0
                elif kappa_cls == 'gentle':
                    dyn_target = 15.0
                elif kappa_cls == 'moderate':
                    dyn_target = 15.0
                elif kappa_cls == 'sharp':
                    dyn_target = 15.0
                else:  # very_sharp
                    dyn_target = 15.0
            else:
                dyn_target = 15.0  # unknown curvature fallback

            # Clamp based on lane visibility
            if self.last_lanes_detected <= 0:       # no lanes
                dyn_target = min(dyn_target, 12.0)
            elif self.last_lanes_detected == 1:     # single lane
                dyn_target = min(dyn_target, 15.0)
            # (2+ lanes -> keep dyn_target)
            self.target_speed = dyn_target

            # Speed control toward dynamic target
            # Smoother speed control bands to reduce jerking
            speed_err = self.target_speed - current_speed
            if speed_err > 8:
                control.throttle, control.brake = 0.5, 0.0
            elif speed_err > 4:
                control.throttle, control.brake = 0.35, 0.0
            elif speed_err > 1:
                control.throttle, control.brake = 0.22, 0.0
            elif speed_err < -5:
                control.throttle, control.brake = 0.0, 0.25
            elif speed_err < -2:
                control.throttle, control.brake = 0.0, 0.12
            else:
                control.throttle, control.brake = 0.18, 0.0
            
            # Steering control (prefer map-based when lanes weak)
            use_map_fallback = (self.last_lanes_detected <= 1)
            map_steer = self._map_based_steer(lookahead_m=12.0) if use_map_fallback else None
            if map_steer is not None:
                control.steer = map_steer
            else:
                # Apply EMA smoothing to lateral error to reduce steering jerk
                if lateral_error is not None:
                    if self.lateral_error_ema is None:
                        self.lateral_error_ema = lateral_error
                    else:
                        self.lateral_error_ema = (self.lateral_error_alpha * lateral_error + 
                                                 (1 - self.lateral_error_alpha) * self.lateral_error_ema)
                    smoothed_error = self.lateral_error_ema
                else:
                    smoothed_error = None
                
                last_steer = self.steering_history[-1] if self.steering_history else None
                if self.controller_type == 'curvature':
                    control.steer = self.curv_controller.step(
                        lane_detector=self.lane_detector,
                        lateral_error_m=smoothed_error,
                        speed_kmh=current_speed,
                        last_out=last_steer
                    )
                else:
                    # PID with smoothed lateral error for reduced jerk
                    if smoothed_error is not None:
                        control.steer = self.pid_controller.step(smoothed_error, last_out=last_steer)
                    else:
                        control.steer = self.steering_history[-1] * 0.95 if self.steering_history else 0.0
            self.steering_history.append(control.steer)
        
        return control, decision
    
    def _emergency_stop(self) -> Dict:
        """Emergency stop"""
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        
        return {
            'control': control,
            'lane_data': None,
            'obstacle_data': None,
            'decision': "EMERGENCY: No data"
        }
    
    def _get_vehicle_speed(self) -> float:
        """Get vehicle speed in km/h"""
        import math
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
    
    def _get_signed_speed_kmh(self) -> float:
        """Get signed speed (forward/reverse)"""
        vel = self.vehicle.get_velocity()
        tf = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        speed_ms = vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z
        return speed_ms * 3.6
    
    def toggle_lane_mask_visualization(self):
        """Toggle lane mask visualization"""
        self.show_lane_mask = not self.show_lane_mask
        status = "ON" if self.show_lane_mask else "OFF"
        print(f"Lane mask visualization: {status}")
    
    def visualize(self, image, result: Dict) -> Tuple:
        """Create visualization"""
        vis = image.copy()
        
        # NEW: Draw lane mask using YOLOLaneFilter (the working one)
        if self.show_lane_mask:
            if result['lane_data'] and result['lane_data']['filtered_lanes']:
                # Use the YOLOLaneFilter's visualization method
                if self.yolo_lane_filter.lane_mask is not None:
                    vis = self.yolo_lane_filter.visualize_lane_mask(vis, alpha=0.3)
                    
                    # Add label
                    cv2.putText(vis, "LANE MASK ON", (vis.shape[1] - 200, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw ROI if active
        roi_points = self.lane_detector.roi_selector.roi_points
        if len(roi_points) == 3:
            import numpy as np
            pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, vis, 0.9, 0, vis)
        
        # Draw lanes
        if result['lane_data']:
            lane_data = result['lane_data']
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            
            for i, lane in enumerate(lane_data['filtered_lanes']):
                color = colors[i % len(colors)]
                for point in lane:
                    cv2.circle(vis, tuple(point), 3, color, -1)
                if len(lane) > 1:
                    import numpy as np
                    points = np.array(lane, dtype=np.int32)
                    cv2.polylines(vis, [points], False, color, 2)
        
        # Draw obstacles
        if result['obstacle_data']:
            obs_data = result['obstacle_data']
            vis = self.obstacle_detector.visualize(
                vis, 
                obs_data['lane_detections'],
                None
            )
        
        # Draw HUD
        speed = self._get_vehicle_speed()
        decision = result['decision']
        
        # Mode color
        if self.mode == 'manual':
            status_color = (255, 165, 0)  # Orange
        elif "DRIVE" in decision:
            status_color = (0, 255, 0)
        else:
            status_color = (0, 0, 255)
        
        cv2.putText(vis, f"Mode: {self.mode.upper()}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(vis, f"Decision: {decision}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(vis, f"Speed: {speed:.1f} km/h", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # if hasattr(self, 'last_curvature') and self.last_curvature is not None:
            # cv2.putText(vis, f"Curv: {self.last_curvature:.4f} ({self.last_curvature_class})", (10, 240),
                        # cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
        
        if result['lane_data']:
            lanes_detected = result['lane_data']['lanes_detected']
            cv2.putText(vis, f"Lanes: {lanes_detected}", (10, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if result['obstacle_data']:
            obs_count = len(result['obstacle_data']['lane_detections'])
            cv2.putText(vis, f"Lane Objects: {obs_count}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # ROI status
        roi_status = "Active" if len(roi_points) == 3 else "Inactive"
        cv2.putText(vis, f"ROI: {roi_status}", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Controls help
        cv2.putText(vis, "[M]=Manual [L]=Auto [V]=Lane Mask [W/S/A/D]=Drive [Q]=Quit", 
                   (10, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2)
        
        return vis, None
    
    def cleanup(self):
        """Cleanup resources"""
        if self.spawner:
            self.spawner.cleanup()
        
        # Stop vehicle
        control = carla.VehicleControl()
        control.throttle = 0.0
        control.brake = 1.0
        self.vehicle.apply_control(control)
