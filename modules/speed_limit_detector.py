"""
Speed Limit Detector Module
Detects speed limit signs on or beside the driving road
"""

import cv2
from typing import Dict, Tuple, Optional, List
from collections import deque
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Ultralytics YOLO not available. Speed limit detection disabled.")


class SpeedLimitDetector:
    """
    Detects speed limit signs and returns current speed limit recommendation
    Supports signs: 30, 50, 90 km/h
    """

    SPEED_LIMITS = {
        '30': 20,
        '50': 30,
        '90': 30,
        'no_limit': 30,
    }

    def __init__(self, model_path: str = 'yolo11n.pt', conf_threshold: float = 0.5):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None

        self.current_detected_speed_limit: Optional[int] = None
        self.default_speed_limit = 25
        self.max_speed_limit = 30

        self.detection_history = deque(maxlen=5)
        self.sign_detection_confidence_threshold = 0.4

        self.last_confirmed_limit = None
        self.frames_until_update = 0
        self.update_interval = 10

        print("Speed Limit Detector initialized")

    def detect(self, image, pre_computed_signs: List[Dict] = None) -> Dict:
        img_vis = image.copy()

        detected_speeds = []
        detected_boxes = []

        if pre_computed_signs:
            for sign in pre_computed_signs:
                class_name = sign.get('class', 'unknown')
                confidence = sign.get('confidence', 0.0)
                bbox = sign.get('bbox', (0, 0, 0, 0))

                speed_limit = self._extract_speed_limit(class_name)

                if speed_limit and confidence >= self.sign_detection_confidence_threshold:
                    detected_speeds.append({
                        'limit': speed_limit,
                        'confidence': confidence,
                        'class': class_name,
                        'bbox': bbox,
                    })
                    detected_boxes.append((bbox, str(speed_limit)))

        self.current_detected_speed_limit = self._update_speed_limit(detected_speeds)
        effective_limit = self._get_effective_limit()

        for bbox, text in detected_boxes:
            img_vis = self._draw_detection(img_vis, bbox, text)

        return {
            'detected_signs': detected_speeds,
            'current_limit': effective_limit,
            'detected_limit': self.current_detected_speed_limit,
            'visualization': img_vis,
            'has_detection': len(detected_speeds) > 0,
        }

    def _extract_speed_limit(self, class_name: str) -> Optional[int]:
        class_lower = str(class_name).lower().strip()

        for sign_key, limit_value in self.SPEED_LIMITS.items():
            if sign_key.lower() in class_lower or class_lower in sign_key.lower():
                return limit_value

        try:
            import re
            numbers = re.findall(r'\d+', class_lower)
            if numbers:
                speed = int(numbers[0])
                if speed in [25, 30, 50, 90]:
                    return speed
        except Exception:
            pass

        return None

    def _update_speed_limit(self, detected_speeds: List[Dict]) -> Optional[int]:
        if not detected_speeds:
            return None

        best_detection = max(detected_speeds, key=lambda x: x['confidence'])
        detected_limit = best_detection['limit']
        confidence = best_detection['confidence']

        if confidence >= 0.6 and detected_limit is not None:
            return detected_limit

        return None

    def _get_effective_limit(self) -> int:
        if self.current_detected_speed_limit is not None:
            return min(self.current_detected_speed_limit, self.max_speed_limit)
        return min(self.default_speed_limit, self.max_speed_limit)

    def _draw_detection(self, image, bbox: Tuple, speed_text: str):
        if not bbox or len(bbox) < 4:
            return image

        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        x1 = max(0, min(x1, image.shape[1] - 1))
        y1 = max(0, min(y1, image.shape[0] - 1))
        x2 = max(0, min(x2, image.shape[1] - 1))
        y2 = max(0, min(y2, image.shape[0] - 1))

        color = (0, 165, 255)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)

        label = f"Speed: {speed_text} km/h"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.rectangle(image, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def reset_state(self):
        self.current_detected_speed_limit = None
        self.detection_history.clear()
        self.frames_until_update = 0
