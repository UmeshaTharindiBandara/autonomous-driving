"""
Driving Agent Module
Autonomous Driving + Lane Following + Intelligent Overtaking
"""

import carla
import cv2
import time
import math
import numpy as np

from collections import deque
from typing import Dict, Tuple, Optional

from modules.lane_detector import LaneDetector
from modules.obstacle_detector import ObstacleDetector
from modules.overtake_manager import OvertakeManager

from core.pid_controller import PIDController
from core.carla_spawner import CarlaSpawner

from detection.yolo_lane_filter import YOLOLaneFilter


# ==========================================================
# CONTROL PARAMETERS
# ==========================================================

PID_KP = 0.55
PID_KI = 0.02
PID_KD = 0.22

STEER_LIMIT = 0.35

TARGET_SPEED = 20.0
OVERTAKE_SPEED = 35.0

SAFE_OVERTAKE_DISTANCE = 18.0


class DrivingAgent:

    def __init__(self, world, vehicle):

        self.world = world
        self.vehicle = vehicle

        # ======================================================
        # MODULES
        # ======================================================

        self.lane_detector = LaneDetector()

        self.obstacle_detector = ObstacleDetector()

        self.overtake_manager = OvertakeManager(world, vehicle)

        self.yolo_lane_filter = YOLOLaneFilter(
            img_width=self.lane_detector.img_w,
            img_height=self.lane_detector.img_h
        )

        self.obstacle_detector.calibrate_camera(
            self.lane_detector.img_w,
            self.lane_detector.img_h,
            90
        )

        self.pid_controller = PIDController(
            kp=PID_KP,
            ki=PID_KI,
            kd=PID_KD,
            i_limit=0.6,
            rate_limit=0.03,
            out_limit=STEER_LIMIT,
            sign=-1.0
        )

        # ======================================================
        # STATES
        # ======================================================

        self.mode = "manual"

        self.target_speed = TARGET_SPEED

        self.frame_count = 0

        self.gradual_stop_active = False

        self.steering_history = deque(maxlen=5)

        self.show_lane_mask = False

        self.spawner = None

        print("✓ Driving Agent initialized")

    # ==========================================================
    # MAIN FRAME PROCESSING
    # ==========================================================

    def process_frame(self, image):

        if self.mode == "manual":
            return self._manual_mode(image)

        # ======================================================
        # PERCEPTION
        # ======================================================

        lane_result = self.lane_detector.detect(image)

        if lane_result is None:
            return self._emergency_stop()

        lateral_error = self.lane_detector.compute_lateral_error(
            lane_result['filtered_lanes']
        )

        all_detections, _ = self.obstacle_detector.detect(
            image,
            self._get_vehicle_speed()
        )

        # ======================================================
        # LANE FILTERING
        # ======================================================

        self.yolo_lane_filter.create_lane_mask_from_lanes(
            lane_result['filtered_lanes'],
            expansion_width=50,
            forward_extension=300
        )

        lane_detections = self.yolo_lane_filter.filter_detections_by_lane(
            all_detections,
            overlap_threshold=0.3
        )

        # ======================================================
        # OVERTAKE SYSTEM
        # ======================================================

        vehicle_speed = self._get_vehicle_speed()

        overtake_state = self.overtake_manager.update(
            lane_detections,
            vehicle_speed
        )

        # ======================================================
        # DECISION MAKING
        # ======================================================

        control, decision = self._make_control_decision(
            lateral_error,
            lane_detections,
            overtake_state
        )

        return {
            'control': control,
            'lane_data': lane_result,
            'obstacle_data': {
                'all_detections': all_detections,
                'lane_detections': lane_detections
            },
            'decision': decision,
            'overtake_state': overtake_state
        }

    # ==========================================================
    # CONTROL DECISION
    # ==========================================================

    def _make_control_decision(
            self,
            lateral_error,
            lane_detections,
            overtake_state):

        control = carla.VehicleControl()

        current_speed = self._get_vehicle_speed()

        # ======================================================
        # EMERGENCY STOP
        # ======================================================

        emergency_obstacle = self._check_emergency_obstacle(
            lane_detections
        )

        if emergency_obstacle:

            control.throttle = 0.0
            control.brake = 1.0
            control.steer = 0.0

            return control, "EMERGENCY STOP"

        # ======================================================
        # OVERTAKING MODE
        # ======================================================

        if overtake_state == "OVERTAKING":

            target_speed = OVERTAKE_SPEED

            # steer LEFT slightly
            control.steer = -0.22

            if current_speed < target_speed:
                control.throttle = 0.65
                control.brake = 0.0
            else:
                control.throttle = 0.25
                control.brake = 0.0

            return control, "OVERTAKING"

        # ======================================================
        # RETURNING TO LANE
        # ======================================================

        elif overtake_state == "RETURNING":

            control.steer = 0.20

            if current_speed < TARGET_SPEED:
                control.throttle = 0.4
            else:
                control.throttle = 0.2

            control.brake = 0.0

            return control, "RETURNING TO LANE"

        # ======================================================
        # NORMAL LANE FOLLOWING
        # ======================================================

        if current_speed < TARGET_SPEED - 5:
            control.throttle = 0.6
            control.brake = 0.0

        elif current_speed < TARGET_SPEED:
            control.throttle = 0.35
            control.brake = 0.0

        else:
            control.throttle = 0.2
            control.brake = 0.0

        # PID Steering
        if lateral_error is not None:

            last_steer = self.steering_history[-1] \
                if self.steering_history else 0.0

            steer = self.pid_controller.step(
                lateral_error,
                last_out=last_steer
            )

            steer = np.clip(steer, -STEER_LIMIT, STEER_LIMIT)

            control.steer = float(steer)

            self.steering_history.append(control.steer)

        else:
            control.steer = 0.0

        return control, "LANE FOLLOWING"

    # ==========================================================
    # EMERGENCY CHECK
    # ==========================================================

    def _check_emergency_obstacle(self, lane_detections):

        for det in lane_detections:

            distance = det.get("distance", 999)

            if distance < 6.0:
                return True

        return False

    # ==========================================================
    # MANUAL MODE
    # ==========================================================

    def _manual_mode(self, image):

        control = carla.VehicleControl()

        return {
            'control': control,
            'lane_data': None,
            'obstacle_data': None,
            'decision': "MANUAL MODE"
        }

    # ==========================================================
    # EMERGENCY STOP
    # ==========================================================

    def _emergency_stop(self):

        control = carla.VehicleControl()

        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0

        return {
            'control': control,
            'lane_data': None,
            'obstacle_data': None,
            'decision': "EMERGENCY STOP"
        }

    # ==========================================================
    # VEHICLE SPEED
    # ==========================================================

    def _get_vehicle_speed(self):

        vel = self.vehicle.get_velocity()

        speed = math.sqrt(
            vel.x ** 2 +
            vel.y ** 2 +
            vel.z ** 2
        )

        return speed * 3.6

    # ==========================================================
    # MODE SWITCH
    # ==========================================================

    def set_mode(self, mode):

        self.mode = mode

        print(f"Mode changed to {mode}")

    # ==========================================================
    # VISUALIZATION
    # ==========================================================

    def visualize(self, image, result):

        vis = image.copy()

        decision = result['decision']

        speed = self._get_vehicle_speed()

        cv2.putText(
            vis,
            f"Decision: {decision}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.putText(
            vis,
            f"Speed: {speed:.1f} km/h",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        if 'overtake_state' in result:

            cv2.putText(
                vis,
                f"Overtake: {result['overtake_state']}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 255),
                2
            )

        return vis, None

    # ==========================================================
    # CLEANUP
    # ==========================================================

    def cleanup(self):

        if self.spawner:
            self.spawner.cleanup()

        control = carla.VehicleControl()

        control.throttle = 0.0
        control.brake = 1.0

        self.vehicle.apply_control(control)