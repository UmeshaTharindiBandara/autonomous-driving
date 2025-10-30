"""
YOLO-based obstacle detection with distance estimation for CARLA
Uses YOLOv11 with pinhole camera model for distance calculation
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available. Install with: pip install ultralytics")


class YOLODistanceDetector:
    """YOLO-based object detection with distance estimation using monocular camera"""
    
    # Known object heights (in meters) - real-world measurements
    OBJECT_HEIGHTS = {
        'person': 1.7,
        'bicycle': 1.1,
        'car': 1.5,
        'motorcycle': 1.2,
        'bus': 3.2,
        'truck': 3.5,
        'default': 1.6
    }
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.25):
        """Initialize YOLO detector with distance estimation"""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Camera parameters
        self.focal_length = None
        self.image_height = None
        
        # Distance thresholds (in meters) - UPDATED FOR 15M STOP DISTANCE
        self.stop_distance = 15.0
        self.danger_distance = 10.0
        self.warning_distance = 20.0
        self.safe_distance_threshold = 25.0
        
        # Detection history
        self.detection_history = []
        self.history_size = 3
        
        print(f"✓ YOLO model loaded: {model_path}")
        print(f"  Stop distance: {self.stop_distance}m")
        print(f"  Danger distance: {self.danger_distance}m")
        print(f"  Warning distance: {self.warning_distance}m")
    
    def calibrate_focal_length(self, image_width: int, image_height: int, fov_degrees: float = 90):
        """Calculate focal length from camera FOV"""
        self.image_height = image_height
        fov_radians = np.deg2rad(fov_degrees)
        self.focal_length = (image_width / 2.0) / np.tan(fov_radians / 2.0)
        print(f"✓ Camera calibrated: focal_length={self.focal_length:.1f}px, FOV={fov_degrees}°")
    
    def estimate_distance(self, bbox_height: float, object_class: str) -> Optional[float]:
        """Estimate distance using pinhole camera model"""
        if self.focal_length is None or bbox_height <= 0:
            return None
        
        real_height = self.OBJECT_HEIGHTS.get(object_class, self.OBJECT_HEIGHTS['default'])
        distance = (real_height * self.focal_length) / bbox_height
        
        if 0.5 <= distance <= 200:
            return distance
        return None
    
    def detect_and_calculate_distance(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
        """Run YOLO detection and calculate distances"""
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                bbox_height = y2 - y1
                distance = self.estimate_distance(bbox_height, class_name)
                
                is_dangerous = False
                if distance is not None:
                    is_dangerous = distance <= self.stop_distance
                
                detection = {
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'class': class_name,
                    'class_id': class_id,
                    'distance': distance,
                    'is_dangerous': is_dangerous,
                    'bbox_center': (int((x1 + x2) / 2), int((y1 + y2) / 2))
                }
                
                detections.append(detection)
        
        annotated = results[0].plot() if len(results) > 0 else image.copy()
        
        return detections, annotated
    
    def should_stop(self, detections: List[Dict], lanes: List = None, 
                   image_width: int = 1280) -> Tuple[bool, Optional[Dict], Optional[Tuple[int, int]]]:
        """Determine if vehicle should stop"""
        if not detections:
            return False, None, None
        
        dangerous_objects = [d for d in detections if d.get('is_dangerous', False)]
        
        if not dangerous_objects:
            return False, None, None
        
        dangerous_objects.sort(key=lambda x: x.get('distance', float('inf')))
        nearest = dangerous_objects[0]
        
        if lanes and len(lanes) > 0:
            all_x = [pt[0] for lane in lanes for pt in lane]
            lane_left = min(all_x) if all_x else int(image_width * 0.2)
            lane_right = max(all_x) if all_x else int(image_width * 0.8)
        else:
            lane_left = int(image_width * 0.2)
            lane_right = int(image_width * 0.8)
        
        lane_bounds = (lane_left, lane_right)
        
        return True, nearest, lane_bounds
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], 
                            lane_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Draw detections on image"""
        vis = image.copy()
        
        if lane_bounds:
            left_x, right_x = lane_bounds
            cv2.line(vis, (left_x, 0), (left_x, vis.shape[0]), (255, 255, 0), 2)
            cv2.line(vis, (right_x, 0), (right_x, vis.shape[0]), (255, 255, 0), 2)
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            distance = det.get('distance')
            class_name = det['class']
            confidence = det['confidence']
            is_dangerous = det.get('is_dangerous', False)
            
            if is_dangerous and distance is not None:
                if distance < self.danger_distance:
                    color = (0, 0, 255)
                elif distance < self.stop_distance:
                    color = (0, 140, 255)
                elif distance < self.warning_distance:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
            else:
                color = (0, 255, 0)
            
            thickness = 3 if is_dangerous else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            if distance is not None:
                label = f"{class_name} {distance:.1f}m ({confidence:.2f})"
            else:
                label = f"{class_name} ({confidence:.2f})"
            
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            center_x, center_y = det['bbox_center']
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
        
        return vis