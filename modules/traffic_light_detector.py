"""
Traffic Light Detection Module
Integrates pre-computed traffic light detections with vehicle control logic.
"""

import time
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict
import cv2
import carla


class TrafficLightDetector:
    """Traffic light detector logic in pipeline mode."""

    def __init__(self):
        self.model = None

        # State smoothing
        self.smooth_n = 3
        self.recent_model_states = deque(maxlen=self.smooth_n)
        self.had_detection_this_frame = False
        self.last_raw_detection_time = 0.0

        # Control state
        self.stopped_for_red = False
        self.green_detected_time = 0.0
        self.resume_start_time = 0.0
        self.last_red_time = 0.0
        self.last_effective_state = None
        self.last_danger_state_time = 0.0
        self.last_detection_time = 0.0
        self.provisional_red_start = 0.0

        # Control parameters
        self.red_light_brake_force = 0.9
        self.yellow_light_brake_force = 0.5
        self.green_light_delay = 0.5
        self.no_detection_resume_time = 1.0
        self.resume_throttle = 0.5
        self.resume_duration = 2.0
        self.min_stop_speed = 0.5
        self.red_confirm_time = 0.3

    def is_available(self) -> bool:
        return True

    def detect(self, image, pre_computed_lights: List[Dict]) -> Dict:
        img_vis = image.copy()
        now = time.time()

        self.had_detection_this_frame = len(pre_computed_lights) > 0
        if self.had_detection_this_frame:
            self.last_raw_detection_time = now

        best_conf = -1.0
        best_cls_name = None
        unique_boxes = []

        for light in pre_computed_lights:
            conf = light.get('confidence', 0.0)
            state = light.get('state')
            bbox = light.get('bbox')
            if not state or not bbox:
                continue
            unique_boxes.append((bbox, state, conf, 'Pipeline'))
            if conf > best_conf:
                best_conf = conf
                best_cls_name = state

        model_state_frame = best_cls_name if best_cls_name else None

        if model_state_frame:
            self.recent_model_states.append(model_state_frame)
        model_state = self._majority_state(self.recent_model_states)

        if not self.had_detection_this_frame:
            gap = now - self.last_raw_detection_time if self.last_raw_detection_time > 0 else 999
            if gap > self.no_detection_resume_time:
                self.recent_model_states.clear()
                model_state = None

        for box_coords, cls_name, conf, source in unique_boxes:
            self._draw_detection(img_vis, box_coords, cls_name, conf, source)

        return {
            'detected_boxes': unique_boxes,
            'model_state': model_state,
            'raw_model_state': model_state_frame,
            'visualization': img_vis,
        }

    def get_control_decision(self, model_state: Optional[str], carla_state: Optional[str], vehicle_speed: float) -> Tuple[str, str, float]:
        effective = self._decide_effective_state(model_state, carla_state)

        is_stopped = vehicle_speed < self.min_stop_speed
        now = time.time()

        if self.had_detection_this_frame:
            self.last_detection_time = now

        if effective == 'red':
            if self.provisional_red_start == 0.0:
                self.provisional_red_start = now
            elapsed_red = now - self.provisional_red_start
            if self.had_detection_this_frame:
                self.last_red_time = now
                self.last_danger_state_time = now
            self.green_detected_time = 0.0

            if elapsed_red < self.red_confirm_time:
                brake_force = min(0.4, self._calculate_progressive_brake(vehicle_speed))
                return f"RED (confirming {self.red_confirm_time - elapsed_red:.2f}s)", 'stop', brake_force

            if not self.stopped_for_red:
                self.stopped_for_red = True
                brake_force = self._calculate_progressive_brake(vehicle_speed)
                return 'RED CONFIRMED - STOPPING', 'stop', brake_force

            if is_stopped:
                return 'STOPPED AT RED', 'stop', 1.0

            brake_force = self._calculate_progressive_brake(vehicle_speed)
            return 'STOPPING FOR RED', 'stop', brake_force

        if effective in ('green', 'yellow'):
            if self.stopped_for_red and effective == 'yellow' and is_stopped:
                return 'YELLOW (STOPPED - HOLDING)', 'stop', 1.0

            if effective == 'green' and self.stopped_for_red:
                if self.green_detected_time == 0.0:
                    self.green_detected_time = now
                time_since_green = now - self.green_detected_time
                if time_since_green < self.green_light_delay:
                    return f"GREEN - WAITING ({self.green_light_delay - time_since_green:.1f}s)", 'stop', 1.0

                if self.resume_start_time == 0.0:
                    self.resume_start_time = now
                time_since_resume = now - self.resume_start_time
                if time_since_resume < self.resume_duration:
                    return 'GREEN - RESUMING', 'resume', 0.0

                self.stopped_for_red = False
                self.green_detected_time = 0.0
                self.resume_start_time = 0.0
                self.provisional_red_start = 0.0
                return 'GREEN - DRIVING', 'drive', 0.0

            if effective == 'yellow':
                if self.had_detection_this_frame:
                    self.last_danger_state_time = now
                return 'YELLOW - SLOWING DOWN', 'slow', self.yellow_light_brake_force

            self.provisional_red_start = 0.0
            self.resume_start_time = 0.0
            return 'DRIVING', 'drive', 0.0

        time_since_last_danger = now - self.last_danger_state_time if self.last_danger_state_time > 0 else 999
        if self.stopped_for_red and is_stopped:
            if time_since_last_danger < self.no_detection_resume_time:
                return 'STOPPED (holding - no TL seen)', 'stop', 1.0

            self.stopped_for_red = False
            self.green_detected_time = 0.0
            self.resume_start_time = 0.0
            self.provisional_red_start = 0.0
            return 'NO DETECTION - CAUTIOUS RESUME', 'drive', 0.0

        if self.stopped_for_red:
            self.stopped_for_red = False
            self.green_detected_time = 0.0
            self.resume_start_time = 0.0
            self.provisional_red_start = 0.0

        return 'NO DETECTION', 'drive', 0.0

    def reset_state(self):
        self.stopped_for_red = False
        self.green_detected_time = 0.0
        self.resume_start_time = 0.0
        self.last_red_time = 0.0
        self.last_danger_state_time = 0.0
        self.last_detection_time = 0.0
        self.provisional_red_start = 0.0
        self.recent_model_states.clear()

    def _majority_state(self, history: deque) -> Optional[str]:
        if not history:
            return None
        c = Counter(history)
        pref = ['red', 'green', 'yellow']
        best_count = max(c.values())
        candidates = [s for s, k in c.items() if k == best_count]
        for p in pref:
            if p in candidates:
                return p
        return candidates[0]

    def _decide_effective_state(self, model_state: Optional[str], carla_state: Optional[str]) -> Optional[str]:
        if carla_state and model_state and carla_state != model_state:
            return carla_state
        return model_state or carla_state

    def _calculate_progressive_brake(self, vehicle_speed: float) -> float:
        if vehicle_speed > 30:
            return 0.8
        return min(1.0, self.red_light_brake_force + (1.0 - self.red_light_brake_force) * (1.0 - vehicle_speed / 30))

    def _draw_detection(self, image, box_coords: Tuple, cls_name: str, conf: float, source: str):
        x1, y1, x2, y2 = box_coords
        if cls_name == 'red':
            color = (0, 0, 255)
        elif cls_name == 'green':
            color = (0, 255, 0)
        elif cls_name == 'yellow':
            color = (0, 255, 255)
        else:
            color = (255, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f"{cls_name.upper()} {conf:.2f} [{source}]"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    @staticmethod
    def carla_tl_to_str(tl_state: carla.TrafficLightState) -> Optional[str]:
        if tl_state is None:
            return None
        name = tl_state.name.lower()
        if 'red' in name:
            return 'red'
        if 'green' in name:
            return 'green'
        if 'yellow' in name or 'amber' in name:
            return 'yellow'
        return None
