"""
Driving Agent Module
Decision-making and control for autonomous driving with manual mode support.

Control pipeline (auto mode, per frame):
    1. Lane detection  → lateral error and lane layout
    2. Obstacle detection → lane-filtered detections
    3. _make_control_decision() → VehicleControl + decision string

ACC zones (nearest_obstacle present):
    0 – 8 m   : emergency full stop
    8 – 25 m  : speed taper (proportional brake)
    25 – 40 m : follow mode  (match lead speed, hold gap)
    40 m+     : normal cruise

Overtake pre-conditions (all must pass):
    1. Lead 5+ km/h slower than ego
    2. Lead within 15 m
    3. Adjacent lane clear from lane layout
    4. Ego speed > 5 km/h
    5. Cooldown expired (8 s since last overtake)
"""

import carla
import cv2
import math
import time
from collections import deque
from typing import Dict, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from modules.lane_detector import LaneDetector
from modules.obstacle_detector import ObstacleDetector
from modules.speed_limit_detector import SpeedLimitDetector
from modules.traffic_light_detector import TrafficLightDetector
from core.pid_controller import PIDController
from core.carla_spawner import CarlaSpawner
from modules.lead_vehicle_controller import LeadVehicleController
from detection.yolo_lane_filter import YOLOLaneFilter

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
PID_KP, PID_KI, PID_KD = 0.55, 0.02, 0.22
STEER_LIMIT      = 0.25
TARGET_SPEED     = 10.0    # km/h — normal cruise

# ACC
ACC_MIN_DISTANCE    = 8.0    # m  — emergency stop threshold
ACC_TAPER_START     = 25.0   # m  — begin speed taper
ACC_FOLLOW_DISTANCE = 40.0   # m  — desired following gap (follow mode)
ACC_FOLLOW_KP       = 0.08   # gap-error → speed correction gain
ACC_MAX_DECEL_BRAKE = 0.6    # max brake in taper zone
ACC_STOP_BRAKE      = 1.0    # brake for emergency stop

# Overtake
OVERTAKE_MIN_EGO_SPEED     = 5.0   # km/h
OVERTAKE_MIN_LEAD_DISTANCE = 15.0   # m
OVERTAKE_MIN_SPEED_DIFF    = 5.0    # km/h ego faster than lead
OVERTAKE_COOLDOWN          = 8.0    # s between maneuvers

# MPC lane-change tuning
MPC_HORIZON_STEPS          = 10
MPC_STEP_TIME              = 0.20
MPC_STEER_CANDIDATES       = (-0.30, -0.24, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.24, 0.30)
MPC_WEIGHT_LATERAL         = 4.5
MPC_WEIGHT_HEADING         = 2.5
MPC_WEIGHT_STEER_CHANGE    = 0.25
MPC_WEIGHT_LANE_OFFSET     = 3.5

# Manual driving
MAN_STEER_STEP  = 0.04
MAN_STEER_DECAY = 0.90
MAN_THR_STEP    = 0.05
MAN_THR_DECAY   = 0.96
MAN_BRAKE_STEP  = 0.08
MAN_BRAKE_DECAY = 0.90
MAN_MAX_THR     = 0.85
MAN_MAX_BRAKE   = 1.00


class DrivingAgent:
    """Autonomous driving agent with lane keeping, ACC, and direct lane-change control."""

    def __init__(self, world: carla.World, vehicle: carla.Vehicle):
        self.world   = world
        self.vehicle = vehicle

        # Core modules
        self.lane_detector     = LaneDetector()
        self.obstacle_detector = ObstacleDetector()
        self.yolo_lane_filter  = YOLOLaneFilter(
            img_width=self.lane_detector.img_w,
            img_height=self.lane_detector.img_h,
        )
        self.speed_limit_detector = SpeedLimitDetector()
        self.traffic_light_detector = TrafficLightDetector()
        self.traffic_light_enabled = self.traffic_light_detector.is_available()
        self.obstacle_detector.calibrate_camera(
            self.lane_detector.img_w,
            self.lane_detector.img_h,
            fov_degrees=90,
        )
        self.pid_controller = PIDController(
            kp=PID_KP, ki=PID_KI, kd=PID_KD,
            i_limit=0.6, rate_limit=0.03, out_limit=STEER_LIMIT, sign=-1.0,
        )
        self.lead_vehicle = LeadVehicleController(world, vehicle)
        self.lead_vehicle.behavior_mode = "random"

        # Agent state
        self.mode             = "manual"
        self.target_speed     = TARGET_SPEED
        self.current_speed_limit = None
        self.default_speed_limit = 25
        self.steering_history = deque(maxlen=4)
        self.frame_count      = 0
        self.show_lane_mask   = False
        self.lane_change_active = False
        self.lane_change_direction = None
        self.lane_change_start_time = 0.0
        self.lane_change_duration = 4.0
        self.last_overtake_time = 0.0
        self.overtake_debug = None

        # Manual control state
        self.manual_throttle = 0.0
        self.manual_brake    = 0.0
        self.manual_steer    = 0.0
        self.manual_reverse  = False

        # ROI
        self.awaiting_roi_choice     = False
        self.roi_choice_deadline     = 0
        self.roi_choice_has_existing = False

        self.spawner = None

        print("🔎 Checking for saved ROI...")
        self.lane_detector.roi_selector.load_from_csv()

        # Spawn the lead vehicle so the obstacle/follow behavior has something to control.
        if not self.lead_vehicle.spawn_lead_vehicle():
            print("⚠️ Lead vehicle did not spawn; obstacle-follow behavior will stay idle")
        else:
            print("✓ Lead vehicle active")

        print("✓ Driving Agent initialized")
        print(f"  Mode: {self.mode.upper()}")

    # ------------------------------------------------------------------
    # Mode switching
    # ------------------------------------------------------------------

    def set_mode(self, mode: str):
        if mode not in ("manual", "auto"):
            print(f"⚠️ Invalid mode: {mode}")
            return
        self.mode = mode
        print(f"⚙️  Switched to {mode.upper()} mode")
        if mode == "auto":
            self.manual_throttle = 0.0
            self.manual_brake    = 0.0
            self.manual_steer    = 0.0
            self.manual_reverse  = False

    # ------------------------------------------------------------------
    # ROI helpers
    # ------------------------------------------------------------------

    def handle_roi_choice_when_auto(self, current_frame):
        has_existing = self.lane_detector.roi_selector.load_from_csv()
        if has_existing:
            print("→ Auto-using EXISTING ROI points")
            self.awaiting_roi_choice = False
            return
        prompt = current_frame.copy()
        cv2.rectangle(prompt, (20, 20), (1260, 140), (0, 0, 0), -1)
        cv2.putText(
            prompt,
            "Autonomous mode: Choose ROI   [1]=Existing   [2]=Mark New   [Esc]=Skip",
            (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )
        cv2.imshow("Autonomous Driving - Modular", prompt)
        self.awaiting_roi_choice     = True
        self.roi_choice_deadline     = time.time() + 5.0
        self.roi_choice_has_existing = has_existing

    def process_roi_choice_key(self, key):
        if not self.awaiting_roi_choice:
            return
        if key == ord("1") and self.roi_choice_has_existing:
            print("→ Using EXISTING ROI points")
            self.awaiting_roi_choice = False
        elif key == ord("2"):
            print("→ Mark NEW ROI points")
            self.awaiting_roi_choice = False
        elif key == 27:
            print("→ Skipping ROI selection")
            self.awaiting_roi_choice = False
        elif time.time() > self.roi_choice_deadline:
            print("→ ROI choice timeout")
            self.awaiting_roi_choice = False

    # ------------------------------------------------------------------
    # Traffic spawning
    # ------------------------------------------------------------------

    def spawn_traffic(self, num_vehicles=10, num_pedestrians=0, num_static=3):
        self.spawner = CarlaSpawner(self.world)
        self.spawner.spawn_traffic_obstacles(num_vehicles, num_pedestrians, num_static)

    # ------------------------------------------------------------------
    # Manual control
    # ------------------------------------------------------------------

    def process_manual_keys(self, key):
        signed_speed = self._get_signed_speed_kmh()
        near_stop    = abs(signed_speed) < 0.5

        if key == ord("a"):
            self.manual_steer = max(-STEER_LIMIT, self.manual_steer - MAN_STEER_STEP)
        elif key == ord("d"):
            self.manual_steer = min(STEER_LIMIT, self.manual_steer + MAN_STEER_STEP)
        else:
            self.manual_steer *= MAN_STEER_DECAY

        if key == ord("w"):
            self.manual_reverse = False
            self.manual_brake   = 0.0
            if near_stop and self.manual_throttle < 0.25:
                self.manual_throttle = 0.25
            else:
                self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)

        elif key == ord("s"):
            if signed_speed > 1.0:
                self.manual_throttle = 0.0
                self.manual_brake    = min(MAN_MAX_BRAKE, self.manual_brake + MAN_BRAKE_STEP)
            else:
                self.manual_reverse  = True
                self.manual_brake    = 0.0
                if near_stop and self.manual_throttle < 0.25:
                    self.manual_throttle = 0.25
                else:
                    self.manual_throttle = min(MAN_MAX_THR, self.manual_throttle + MAN_THR_STEP)

        elif key == 32:
            self.manual_throttle = 0.0
            self.manual_brake    = 1.0

        else:
            self.manual_throttle *= MAN_THR_DECAY
            self.manual_brake    *= MAN_BRAKE_DECAY

        if self.manual_brake > 0.1:
            self.manual_throttle = 0.0

    def apply_manual_control(self) -> carla.VehicleControl:
        ctrl            = carla.VehicleControl()
        ctrl.throttle   = float(self.manual_throttle)
        ctrl.brake      = float(self.manual_brake)
        ctrl.steer      = float(self.manual_steer)
        ctrl.hand_brake = False
        ctrl.reverse    = bool(self.manual_reverse)
        return ctrl

    # ------------------------------------------------------------------
    # Main frame processing
    # ------------------------------------------------------------------

    def process_frame(self, image) -> Dict:

        # Keep optional lead vehicle behavior running when enabled.
        self.lead_vehicle.update()

        # ── Manual mode ──────────────────────────────────────────────
        if self.mode == "manual":
            lane_result       = self.lane_detector.detect(image)
            all_detections, _ = self.obstacle_detector.detect(image)

            if lane_result:
                self.yolo_lane_filter.create_lane_mask_from_lanes(
                    lane_result["filtered_lanes"],
                    expansion_width=50,
                    forward_extension=300,
                )
                lane_detections = self.yolo_lane_filter.filter_detections_by_lane(
                    all_detections, overlap_threshold=0.3
                )
            else:
                lane_detections = []

            return {
                "control":        self.apply_manual_control(),
                "lane_data":      lane_result,
                "obstacle_data":  {
                    "all_detections":   all_detections,
                    "lane_detections":  lane_detections,
                    "nearest_obstacle": None,
                    "should_stop":      False,
                },
                "decision":       "MANUAL CONTROL",
                "overtake_debug": None,
            }

        # ── Auto mode ─────────────────────────────────────────────────
        lane_result = self.lane_detector.detect(image)
        if lane_result is None:
            return self._emergency_stop()

        lateral_error     = self.lane_detector.compute_lateral_error(lane_result["filtered_lanes"])
        all_detections, _ = self.obstacle_detector.detect(image)

        traffic_light_data = None
        if self.traffic_light_enabled:
            traffic_light_data = self.traffic_light_detector.detect(image, [])

        speed_limit_data = self.speed_limit_detector.detect(image, pre_computed_signs=[])
        self.current_speed_limit = speed_limit_data.get('current_limit', self.default_speed_limit)

        self.yolo_lane_filter.create_lane_mask_from_lanes(
            lane_result["filtered_lanes"],
            expansion_width=50,
            forward_extension=300,
        )
        lane_detections = self.yolo_lane_filter.filter_detections_by_lane(
            all_detections, overlap_threshold=0.3
        )

        should_stop, nearest_obstacle = self.obstacle_detector.should_stop(lane_detections)
        lane_lost                     = self.lane_detector.is_lane_lost()

        lane_layout = self.lane_detector.get_lane_layout(lane_result["filtered_lanes"])
        vehicle_speed = self._get_vehicle_speed()

        traffic_light_stop = False
        traffic_light_decision = None
        if traffic_light_data:
            tl_state = traffic_light_data.get('model_state')
            tl_decision_text, tl_control_action, tl_brake_force = self.traffic_light_detector.get_control_decision(
                tl_state, None, vehicle_speed
            )
            if tl_control_action in ('stop', 'slow'):
                traffic_light_stop = True
                traffic_light_decision = (tl_decision_text, tl_control_action, tl_brake_force)

        control, decision = self._make_control_decision(
            lateral_error,
            should_stop,
            lane_lost,
            nearest_obstacle,
            lane_layout,
            lane_detections,
            traffic_light_stop,
            traffic_light_decision,
        )

        return {
            "control":        control,
            "lane_data":      lane_result,
            "obstacle_data":  {
                "all_detections":   all_detections,
                "lane_detections":  lane_detections,
                "nearest_obstacle": nearest_obstacle,
                "should_stop":      should_stop,
            },
            "decision":       decision,
            "traffic_light_data": traffic_light_data,
            "speed_limit_data": speed_limit_data,
            "lane_layout": lane_layout,
            "overtake_debug": self.overtake_debug,
        }

    # ------------------------------------------------------------------
    # Control decision
    # ------------------------------------------------------------------

    def _make_control_decision(
        self,
        lateral_error:    Optional[float],
        should_stop:      bool,
        lane_lost:        bool,
        nearest_obstacle: Optional[Dict],
        lane_layout:      Optional[Dict],
        lane_detections:  list = None,
        traffic_light_stop: bool = False,
        traffic_light_decision: Optional[Tuple] = None,
    ) -> Tuple[carla.VehicleControl, str]:

        ctrl          = carla.VehicleControl()
        current_speed = self._get_vehicle_speed()
        self.overtake_debug = {
            "state": "idle",
            "lead_distance_m": None,
            "lead_speed_kmh": None,
            "ego_speed_kmh": current_speed,
            "direction": None,
            "abort_reason": "waiting_for_overtake_trigger",
        }

        if traffic_light_stop and traffic_light_decision:
            tl_decision_text, tl_control_action, tl_brake_force = traffic_light_decision
            if tl_control_action == 'stop':
                ctrl.throttle = 0.0
                ctrl.brake = tl_brake_force
                ctrl.steer = self.steering_history[-1] * 0.8 if self.steering_history else 0.0
                return ctrl, f"TL: {tl_decision_text}"
            if tl_control_action == 'slow':
                ctrl.throttle = 0.0
                ctrl.brake = tl_brake_force
                ctrl.steer = self.steering_history[-1] * 0.9 if self.steering_history else 0.0
                return ctrl, f"TL: {tl_decision_text}"

        if self.lane_change_active:
            self.overtake_debug = {
                "state": "lane_change",
                "direction": self.lane_change_direction,
                "ego_speed_kmh": current_speed,
                "abort_reason": None,
            }
            return self._continue_lane_change(ctrl, current_speed)

        # ══════════════════════════════════════════════════════════════
        # LANE LOST — gradual stop, hold last steer
        # ══════════════════════════════════════════════════════════════
        if lane_lost:
            self.overtake_debug.update({
                "state": "blocked",
                "abort_reason": "lane_lost",
            })
            if current_speed > 1.0:
                ctrl.throttle = 0.0
                ctrl.brake    = 0.1
                ctrl.steer    = (
                    self.steering_history[-1] * 0.8 if self.steering_history else 0.0
                )
            else:
                ctrl.throttle = 0.0
                ctrl.brake    = ACC_STOP_BRAKE
                ctrl.steer    = 0.0
            return ctrl, "STOP: Lane lost"

        # ══════════════════════════════════════════════════════════════
        # ACC — obstacle detected in lane
        # ══════════════════════════════════════════════════════════════
        if nearest_obstacle is not None:
            distance   = nearest_obstacle.get("distance", 0.0) or 0.0
            label      = nearest_obstacle.get("class", "obstacle")
            lead_status = self.lead_vehicle.get_status() if self.lead_vehicle else None
            lead_speed = lead_status.get('speed') if lead_status else None
            ctrl.steer = self._compute_steer(lateral_error)

            self.overtake_debug = {
                "state": "acc",
                "lead_distance_m": distance,
                "lead_speed_kmh": lead_speed,
                "ego_speed_kmh": current_speed,
                "direction": None,
                "abort_reason": None,
            }

            # Zone 1: Emergency stop
            if distance <= ACC_MIN_DISTANCE:
                ctrl.throttle = 0.0
                ctrl.brake    = ACC_STOP_BRAKE
                self.overtake_debug.update({"state": "blocked", "abort_reason": "too_close_for_overtake"})
                return ctrl, f"STOP: {label} at {distance:.1f}m"

            # Zone 2: Speed taper
            if distance <= ACC_TAPER_START:
                closeness = 1.0 - (
                    (distance - ACC_MIN_DISTANCE) / (ACC_TAPER_START - ACC_MIN_DISTANCE)
                )
                closeness     = max(0.0, min(1.0, closeness))
                desired_speed = TARGET_SPEED * (1.0 - closeness)

                if current_speed > desired_speed + 1.0:
                    ctrl.throttle = 0.0
                    ctrl.brake    = min(ACC_MAX_DECEL_BRAKE, closeness * 0.5)
                elif current_speed < desired_speed - 1.0:
                    ctrl.throttle = 0.3
                    ctrl.brake    = 0.0
                else:
                    ctrl.throttle = 0.1
                    ctrl.brake    = 0.0

                # If the lead car is stopped or nearly stopped, allow an overtaking attempt
                # from the taper zone instead of forcing a full stop.
                if lane_layout is not None and lead_speed is not None and lead_speed <= 1.0:
                    direction, abort_reason = self._should_start_overtake(
                        distance, current_speed, lead_speed, lane_layout
                    )
                    self.overtake_debug.update({
                        "state": "taper",
                        "direction": direction,
                        "abort_reason": abort_reason,
                    })
                    if direction is not None:
                        self._start_lane_change(direction)
                        self.overtake_debug["state"] = "overtaking"
                        self.overtake_debug["direction"] = direction
                        return self._continue_lane_change(ctrl, current_speed)

                self.overtake_debug.update({"state": "blocked", "abort_reason": "in_taper_zone"})
                return ctrl, f"ACC taper: {label} d={distance:.1f}m"

            # Zone 3: Follow mode — match lead speed, hold gap
            if distance <= ACC_FOLLOW_DISTANCE:
                target_follow  = lead_speed if lead_speed is not None else TARGET_SPEED
                target_follow  = min(target_follow, TARGET_SPEED)

                gap_error      = ACC_FOLLOW_DISTANCE - distance   # +ve = too close
                speed_corr     = gap_error * ACC_FOLLOW_KP        # km/h correction
                desired_speed  = max(0.0, min(TARGET_SPEED, target_follow - speed_corr))

                if current_speed < desired_speed - 1.0:
                    ctrl.throttle = 0.4
                    ctrl.brake    = 0.0
                elif current_speed > desired_speed + 1.0:
                    ctrl.throttle = 0.0
                    ctrl.brake    = 0.15
                else:
                    ctrl.throttle = 0.2
                    ctrl.brake    = 0.0

                lead_str = f"{lead_speed:.0f}km/h" if lead_speed is not None else "--"
                if lane_layout is not None:
                    direction, abort_reason = self._should_start_overtake(
                        distance, current_speed, lead_speed, lane_layout
                    )
                    self.overtake_debug.update({
                        "state": "follow",
                        "direction": direction,
                        "abort_reason": abort_reason,
                    })
                    if direction is not None:
                        self._start_lane_change(direction)
                        self.overtake_debug["state"] = "overtaking"
                        self.overtake_debug["direction"] = direction
                        return self._continue_lane_change(ctrl, current_speed)
                else:
                    self.overtake_debug.update({"abort_reason": "no_lane_layout"})

                return ctrl, f"ACC follow: {label} d={distance:.1f}m lead={lead_str}"

            if lane_layout:
                direction = self._choose_overtake_direction(lane_layout)
                if direction is not None:
                    self._start_lane_change(direction)
                    self.overtake_debug.update({"state": "overtaking", "direction": direction})
                    return self._continue_lane_change(ctrl, current_speed)
                self.overtake_debug.update({"state": "blocked", "abort_reason": "no_clear_adjacent_lane"})

        else:
            self.overtake_debug.update({
                "state": "idle",
                "abort_reason": "no_front_vehicle",
            })

        # ══════════════════════════════════════════════════════════════
        # NORMAL DRIVE
        # ══════════════════════════════════════════════════════════════
        speed_cap = min(self.target_speed, float(self.current_speed_limit or self.default_speed_limit))
        ctrl = self._normal_drive(ctrl, lateral_error, current_speed, speed_cap)
        return ctrl, "DRIVE: Normal"

    # ------------------------------------------------------------------
    # Normal drive helpers
    # ------------------------------------------------------------------

    def _normal_drive(
        self,
        ctrl: carla.VehicleControl,
        lateral_error: Optional[float],
        current_speed: float,
        speed_cap: float,
    ) -> carla.VehicleControl:
        target = min(self.target_speed, speed_cap)
        if current_speed < target - 5:
            ctrl.throttle, ctrl.brake = 0.7, 0.0
        elif current_speed < target:
            ctrl.throttle, ctrl.brake = 0.4, 0.0
        elif current_speed > target + 5:
            ctrl.throttle, ctrl.brake = 0.0, 0.3
        else:
            ctrl.throttle, ctrl.brake = 0.2, 0.0

        ctrl.steer = self._compute_steer(lateral_error)
        return ctrl

    def _compute_steer(self, lateral_error: Optional[float]) -> float:
        if lateral_error is not None:
            last_steer = self.steering_history[-1] if self.steering_history else None
            steer      = self.pid_controller.step(lateral_error, last_out=last_steer)
            self.steering_history.append(steer)
            return steer
        return self.steering_history[-1] * 0.9 if self.steering_history else 0.0

    def _choose_overtake_direction(self, lane_layout: Optional[Dict]) -> Optional[str]:
        if not lane_layout:
            return self._map_overtake_direction_fallback()
        if lane_layout.get('left_lane_ok'):
            return 'left'
        if lane_layout.get('right_lane_ok'):
            return 'right'
        return self._map_overtake_direction_fallback()

    def _should_start_overtake(
        self,
        lead_distance_m: float,
        ego_speed_kmh: float,
        lead_speed_kmh: Optional[float],
        lane_layout: Optional[Dict],
    ) -> Tuple[Optional[str], Optional[str]]:
        if self.lane_change_active:
            return None, "lane_change_active"

        now = time.time()
        if now - self.last_overtake_time < OVERTAKE_COOLDOWN:
            return None, "cooldown"

        if ego_speed_kmh < OVERTAKE_MIN_EGO_SPEED:
            return None, f"ego_speed<{OVERTAKE_MIN_EGO_SPEED:.0f}km/h"

        if lead_distance_m > OVERTAKE_MIN_LEAD_DISTANCE:
            return None, f"lead_distance>{OVERTAKE_MIN_LEAD_DISTANCE:.0f}m"

        if lead_speed_kmh is None:
            return None, "lead_speed_unknown"

        if (ego_speed_kmh - lead_speed_kmh) < OVERTAKE_MIN_SPEED_DIFF:
            return None, f"speed_gap<{OVERTAKE_MIN_SPEED_DIFF:.0f}km/h"

        direction = self._choose_overtake_direction(lane_layout)
        if direction is None:
            return None, "no_clear_adjacent_lane"

        return direction, None

    def _map_overtake_direction_fallback(self) -> Optional[str]:
        """Fallback to CARLA lane topology when vision lane layout is weak.

        This lets the agent overtake a stopped lead vehicle even when the lane detector
        cannot confidently mark adjacent lanes.
        """
        try:
            world_map = self.world.get_map()
            if world_map is None:
                return None

            veh_tf = self.vehicle.get_transform()
            curr_wp = world_map.get_waypoint(
                veh_tf.location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if curr_wp is None:
                return None

            left_wp = curr_wp.get_left_lane()
            right_wp = curr_wp.get_right_lane()

            if left_wp is not None and left_wp.lane_type == carla.LaneType.Driving:
                return 'left'
            if right_wp is not None and right_wp.lane_type == carla.LaneType.Driving:
                return 'right'
            return None
        except Exception:
            return None

    def _start_lane_change(self, direction: str):
        self.lane_change_active = True
        self.lane_change_direction = direction
        self.lane_change_start_time = time.time()
        print(f"Lane change started: {direction}")

    def _continue_lane_change(self, ctrl: carla.VehicleControl, current_speed: float) -> Tuple[carla.VehicleControl, str]:
        direction = self.lane_change_direction
        steer = self._lane_change_mpc_steer(direction, current_speed, lookahead_m=18.0)
        if steer is not None:
            # Keep a slightly stronger throttle during the maneuver so the ego vehicle clears the obstacle.
            ctrl.throttle = 0.30 if current_speed > 10.0 else 0.35
            ctrl.brake = 0.0
            ctrl.steer = steer
            if time.time() - self.lane_change_start_time >= self.lane_change_duration:
                self.lane_change_active = False
                self.last_overtake_time = time.time()
                completed = direction
                self.lane_change_direction = None
                self.lane_detector.reset_lane_lost_timer()
                self.lateral_error_ema = None
                self.last_lanes_detected = 0
                self.steering_history.clear()
                return ctrl, f"LANE CHANGE COMPLETE: {completed}"
            return ctrl, f"LANE CHANGE: {direction}"

        self.lane_change_active = False
        self.lane_change_direction = None
        return ctrl, "LANE CHANGE ABORTED"

    def _lane_change_mpc_steer(
        self,
        direction: Optional[str],
        current_speed_kmh: float,
        lookahead_m: float = 18.0,
    ) -> Optional[float]:
        """Choose a steering command by rolling out candidate controls over a short horizon.

        This is a lightweight MPC approximation: each candidate steer is simulated forward
        with a kinematic bicycle model and scored against lane-centre/heading error.
        """
        ref_waypoints = self._get_lane_reference_waypoints(direction, lookahead_m=lookahead_m)
        if not ref_waypoints:
            return None

        veh_tf = self.vehicle.get_transform()
        veh_loc = veh_tf.location
        yaw = math.radians(veh_tf.rotation.yaw)
        speed_mps = max(0.0, current_speed_kmh / 3.6)
        wheelbase_m = 2.9
        dt = MPC_STEP_TIME
        prev_steer = self.steering_history[-1] if self.steering_history else 0.0
        target_speed_mps = max(2.0, min(speed_mps, 8.0))
        prediction_speed_mps = 0.7 * speed_mps + 0.3 * target_speed_mps

        best_steer = None
        best_cost = float("inf")
        best_target_offset = 0.0

        for candidate_steer in MPC_STEER_CANDIDATES:
            x = float(veh_loc.x)
            y = float(veh_loc.y)
            theta = float(yaw)
            cost = 0.0
            offset_penalty_accum = 0.0

            for step_index in range(MPC_HORIZON_STEPS):
                x += prediction_speed_mps * math.cos(theta) * dt
                y += prediction_speed_mps * math.sin(theta) * dt
                theta += (prediction_speed_mps / wheelbase_m) * math.tan(candidate_steer) * dt

                ref_x, ref_y, ref_heading, ref_offset = self._sample_reference_state(
                    ref_waypoints, step_index
                )
                lateral_error = math.hypot(x - ref_x, y - ref_y)
                heading_error = self._wrap_angle(theta - ref_heading)
                offset_penalty_accum += abs(ref_offset)

                cost += (
                    MPC_WEIGHT_LATERAL * lateral_error
                    + MPC_WEIGHT_HEADING * abs(heading_error)
                )

            cost += MPC_WEIGHT_LANE_OFFSET * (offset_penalty_accum / max(1, MPC_HORIZON_STEPS))
            cost += MPC_WEIGHT_STEER_CHANGE * abs(candidate_steer - prev_steer)

            if cost < best_cost:
                best_cost = cost
                best_steer = candidate_steer
                best_target_offset = offset_penalty_accum

        if best_steer is None:
            return None

        # Bias the final command slightly toward the chosen lane center so the maneuver
        # settles more accurately on the target lane instead of drifting around it.
        settle_bias = max(-0.04, min(0.04, -0.002 * best_target_offset))
        best_steer += settle_bias

        return float(max(-STEER_LIMIT, min(STEER_LIMIT, best_steer)))

    def _get_lane_reference_waypoints(self, direction: Optional[str], lookahead_m: float = 18.0):
        try:
            world_map = self.world.get_map()
            if world_map is None:
                return None

            veh_tf = self.vehicle.get_transform()
            veh_loc = veh_tf.location
            curr_wp = world_map.get_waypoint(
                veh_loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if curr_wp is None:
                return None

            target_wp = curr_wp
            if direction == 'left':
                target_wp = curr_wp.get_left_lane() or curr_wp
            elif direction == 'right':
                target_wp = curr_wp.get_right_lane() or curr_wp

            waypoints = [target_wp]
            total_distance = 0.0
            cursor = target_wp
            while total_distance < lookahead_m:
                next_waypoints = cursor.next(2.0)
                if not next_waypoints:
                    break
                cursor = next_waypoints[0]
                waypoints.append(cursor)
                total_distance += 2.0

            return waypoints
        except Exception:
            return None

    def _sample_reference_state(self, ref_waypoints, step_index: int):
        idx = min(step_index, len(ref_waypoints) - 1)
        ref_wp = ref_waypoints[idx]
        ref_loc = ref_wp.transform.location
        ref_yaw = math.radians(ref_wp.transform.rotation.yaw)

        veh_tf = self.vehicle.get_transform()
        veh_loc = veh_tf.location
        dx = ref_loc.x - veh_loc.x
        dy = ref_loc.y - veh_loc.y
        cross = math.cos(ref_yaw) * dy - math.sin(ref_yaw) * dx
        lane_offset = cross

        return ref_loc.x, ref_loc.y, ref_yaw, lane_offset

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def _get_lane_target_waypoint(self, direction: Optional[str], lookahead_m: float = 12.0):
        try:
            world_map = self.world.get_map()
            if world_map is None:
                return None
            veh_tf = self.vehicle.get_transform()
            veh_loc = veh_tf.location
            curr_wp = world_map.get_waypoint(veh_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
            if curr_wp is None:
                return None
            target_wp = curr_wp
            if direction == 'left':
                target_wp = curr_wp.get_left_lane() or curr_wp
            elif direction == 'right':
                target_wp = curr_wp.get_right_lane() or curr_wp
            next_wps = target_wp.next(lookahead_m)
            if not next_wps:
                next_wps = target_wp.next(5.0)
                if not next_wps:
                    return None
            return next_wps[0]
        except Exception:
            return None

    def _steer_to_waypoint(self, target_wp) -> Optional[float]:
        try:
            if target_wp is None:
                return None
            import math
            veh_tf = self.vehicle.get_transform()
            veh_loc = veh_tf.location
            tgt = target_wp.transform.location
            yaw = math.radians(veh_tf.rotation.yaw)
            fwd_x, fwd_y = math.cos(yaw), math.sin(yaw)
            dx, dy = (tgt.x - veh_loc.x), (tgt.y - veh_loc.y)
            dist = math.hypot(dx, dy)
            if dist < 1e-3:
                return 0.0
            tx, ty = dx / dist, dy / dist
            cross_z = fwd_x * ty - fwd_y * tx
            dot = fwd_x * tx + fwd_y * ty
            yaw_err = math.atan2(cross_z, dot)
            steer = max(-STEER_LIMIT, min(STEER_LIMIT, 0.8 * yaw_err))
            return float(steer)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Emergency stop
    # ------------------------------------------------------------------

    def _emergency_stop(self) -> Dict:
        ctrl          = carla.VehicleControl()
        ctrl.throttle = 0.0
        ctrl.brake    = 1.0
        ctrl.steer    = 0.0
        return {
            "control":        ctrl,
            "lane_data":      None,
            "obstacle_data":  None,
            "decision":       "EMERGENCY: No data",
            "traffic_light_data": None,
            "speed_limit_data": None,
            "lane_layout": None,
        }

    # ------------------------------------------------------------------
    # Vehicle state
    # ------------------------------------------------------------------

    def _get_vehicle_speed(self) -> float:
        import math
        v = self.vehicle.get_velocity()
        return 3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)

    def _get_signed_speed_kmh(self) -> float:
        vel = self.vehicle.get_velocity()
        tf  = self.vehicle.get_transform()
        fwd = tf.get_forward_vector()
        return (vel.x * fwd.x + vel.y * fwd.y + vel.z * fwd.z) * 3.6

    def toggle_lane_mask_visualization(self):
        self.show_lane_mask = not self.show_lane_mask
        print(f"Lane mask visualization: {'ON' if self.show_lane_mask else 'OFF'}")

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize(self, image, result: Dict) -> Tuple:
        vis = image.copy()

        # Lane mask overlay
        if self.show_lane_mask:
            if result["lane_data"] and result["lane_data"]["filtered_lanes"]:
                if self.yolo_lane_filter.lane_mask is not None:
                    vis = self.yolo_lane_filter.visualize_lane_mask(vis, alpha=0.3)
                    cv2.putText(vis, "LANE MASK ON", (vis.shape[1] - 200, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # ROI polygon
        roi_points = self.lane_detector.roi_selector.roi_points
        if len(roi_points) == 3:
            import numpy as np
            pts = np.array(roi_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(vis, [pts], True, (255, 255, 0), 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 255))
            cv2.addWeighted(overlay, 0.1, vis, 0.9, 0, vis)

        # Lane lines
        if result["lane_data"]:
            import numpy as np
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, lane in enumerate(result["lane_data"]["filtered_lanes"]):
                color = colors[i % len(colors)]
                for pt in lane:
                    cv2.circle(vis, tuple(pt), 3, color, -1)
                if len(lane) > 1:
                    cv2.polylines(vis, [np.array(lane, dtype=np.int32)], False, color, 2)

        # Obstacle boxes
        if result["obstacle_data"]:
            vis = self.obstacle_detector.visualize(
                vis, result["obstacle_data"]["lane_detections"], None
            )

        # ── HUD ──────────────────────────────────────────────────────
        speed    = self._get_vehicle_speed()
        decision = result["decision"]
        mode     = self.mode

        if mode == "manual":
            status_color = (255, 165, 0)       # orange
        elif "DRIVE" in decision:
            status_color = (0, 255, 0)         # green
        elif "ACC follow" in decision:
            status_color = (0, 220, 255)       # cyan
        elif "ACC taper" in decision:
            status_color = (0, 140, 255)       # light orange
        elif "STOP" in decision:
            status_color = (0, 0, 255)         # red
        elif "OVERTAKING" in decision or "RETURNING" in decision:
            status_color = (255, 200, 0)       # yellow
        else:
            status_color = (200, 200, 200)

        hud_lines = [
            (f"Mode: {mode.upper()}",        status_color),
            (f"Decision: {decision}",         status_color),
            (f"Speed: {speed:.1f} km/h",     (255, 255, 255)),
        ]

        if result["lane_data"]:
            hud_lines.append(
                (f"Lanes: {result['lane_data']['lanes_detected']}", (255, 255, 255))
            )

        if result["obstacle_data"]:
            n = len(result["obstacle_data"]["lane_detections"])
            hud_lines.append((f"Lane objects: {n}", (255, 255, 255)))

        # Overtake debug panel
        od = result.get("overtake_debug")
        if od:
            lead_d  = od.get("lead_distance_m")
            lead_v  = od.get("lead_speed_kmh")
            abort   = od.get("abort_reason", "")
            d_str   = f"{lead_d:.1f}m"   if lead_d is not None else "--"
            v_str   = f"{lead_v:.1f}km/h" if lead_v is not None else "--"
            hud_lines.append((f"Lead: dist={d_str}  speed={v_str}", (200, 200, 0)))
            hud_lines.append((f"OT state: {od.get('state', '--')}", (200, 200, 0)))
            if abort:
                hud_lines.append((f"OT blocked: {abort}", (100, 100, 255)))

        lead_status = self.lead_vehicle.get_status()
        if lead_status:
            hud_lines.append((
                f"Lead vehicle: {lead_status['distance']:.1f}m @ {lead_status['speed']:.1f}km/h",
                (255, 120, 255),
            ))

        roi_status = "Active" if len(roi_points) == 3 else "Inactive"
        hud_lines.append((f"ROI: {roi_status}", (255, 255, 0)))

        y = 30
        for text, color in hud_lines:
            cv2.putText(vis, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            y += 28

        cv2.putText(
            vis,
            "[M]=Manual [L]=Auto [V]=Lane Mask [W/S/A/D]=Drive [Q]=Quit",
            (10, vis.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2,
        )

        return vis, None

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self):
        self.lead_vehicle.destroy()
        if self.spawner:
            self.spawner.cleanup()
        ctrl          = carla.VehicleControl()
        ctrl.throttle = 0.0
        ctrl.brake    = 1.0
        self.vehicle.apply_control(ctrl)
