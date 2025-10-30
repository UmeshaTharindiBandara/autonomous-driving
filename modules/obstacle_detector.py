"""
Obstacle Detection Module
YOLO-based obstacle detection with distance estimation and lane filtering
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# UPDATED: Import from detection module
from detection.yolo_distance_detector import YOLO_AVAILABLE

if YOLO_AVAILABLE:
    from detection.yolo_distance_detector import YOLO


class ObstacleDetector:
    """YOLO-based obstacle detection with distance estimation"""
    
    OBJECT_HEIGHTS = {
        'person': 1.7,
        'bicycle': 1.1,
        'car': 1.5,
        'motorcycle': 1.2,
        'bus': 3.2,
        'truck': 3.5,
        'default': 1.6
    }
    
    def __init__(self, model_path='yolo11n.pt', conf_threshold=0.5):
        """Initialize YOLO detector"""
        if not YOLO_AVAILABLE:
            raise ImportError("Ultralytics YOLO not installed. Run: pip install ultralytics")
        
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # Camera parameters
        self.focal_length = None
        self.image_height = None
        
        # Distance thresholds (meters)
        self.stop_distance = 15.0
        self.danger_distance = 10.0
        self.warning_distance = 20.0
        
        # Lane filtering
        self.lane_mask = None
        self.lane_polygon = None
        
        print(f"✓ Obstacle Detector initialized")
        print(f"  Model: {model_path}")
        print(f"  Stop distance: {self.stop_distance}m")
    
    def calibrate_camera(self, image_width: int, image_height: int, fov_degrees: float = 90):
        """Calibrate camera focal length"""
        self.image_height = image_height
        fov_radians = np.deg2rad(fov_degrees)
        self.focal_length = (image_width / 2.0) / np.tan(fov_radians / 2.0)
        print(f"  ✓ Camera calibrated: f={self.focal_length:.1f}px")
    
    def detect(self, image: np.ndarray) -> Tuple[List[Dict], np.ndarray]:
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
                distance = self._estimate_distance(bbox_height, class_name)
                
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
                    'bbox_center': (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                    'in_lane': False
                }
                
                detections.append(detection)
        
        annotated = results[0].plot() if len(results) > 0 else image.copy()
        
        return detections, annotated
    
    def filter_by_lane(self, detections: List[Dict], filtered_lanes: List, 
                       img_shape: Tuple, forward_extension: int = 300) -> List[Dict]:
        """Filter detections to only include objects in lane"""
        if not filtered_lanes:
            return detections
        
        # Create lane mask
        self._create_lane_mask(filtered_lanes, img_shape, forward_extension)
        
        # Filter detections
        filtered = []
        overlap_threshold = 0.3
        
        for detection in detections:
            bbox = detection['bbox']
            x1, y1, x2, y2 = bbox
            
            # Clamp to image bounds
            img_h, img_w = img_shape[:2]
            x1 = max(0, min(int(x1), img_w - 1))
            y1 = max(0, min(int(y1), img_h - 1))
            x2 = max(0, min(int(x2), img_w - 1))
            y2 = max(0, min(int(y2), img_h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0:
                continue
            
            # Calculate overlap with lane mask
            mask_region = self.lane_mask[y1:y2, x1:x2]
            lane_pixels = np.sum(mask_region > 0)
            overlap_ratio = lane_pixels / bbox_area
            
            detection['lane_overlap'] = overlap_ratio
            detection['in_lane'] = overlap_ratio >= overlap_threshold
            
            if overlap_ratio >= overlap_threshold:
                filtered.append(detection)
        
        return filtered
    
    def should_stop(self, lane_detections: List[Dict]) -> Tuple[bool, Optional[Dict]]:
        """Determine if vehicle should stop"""
        if not lane_detections:
            return False, None
        
        dangerous_objects = [d for d in lane_detections if d.get('is_dangerous', False)]
        
        if not dangerous_objects:
            return False, None
        
        # Sort by distance
        dangerous_objects.sort(key=lambda x: x.get('distance', float('inf')))
        nearest = dangerous_objects[0]
        
        return True, nearest
    
    def _estimate_distance(self, bbox_height: float, object_class: str) -> Optional[float]:
        """Estimate distance using pinhole camera model"""
        if self.focal_length is None or bbox_height <= 0:
            return None
        
        real_height = self.OBJECT_HEIGHTS.get(object_class, self.OBJECT_HEIGHTS['default'])
        distance = (real_height * self.focal_length) / bbox_height
        
        if 0.5 <= distance <= 200:
            return distance
        return None
    
    def _create_lane_mask(self, filtered_lanes: List, img_shape: Tuple, forward_extension: int):
        """Create lane mask for filtering"""
        img_h, img_w = img_shape[:2]
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        
        if not filtered_lanes:
            self.lane_mask = mask
            self.lane_polygon = None
            return
        
        # Separate left and right lanes
        center_x = img_w // 2
        left_lanes = []
        right_lanes = []
        
        for lane in filtered_lanes:
            if len(lane) > 0:
                avg_x = np.mean([pt[0] for pt in lane])
                if avg_x < center_x:
                    left_lanes.append(lane)
                else:
                    right_lanes.append(lane)
        
        # Get boundaries
        left_boundary = None
        right_boundary = None
        
        if len(left_lanes) > 0:
            left_boundary = min(left_lanes, key=lambda lane: np.mean([pt[0] for pt in lane]))
        
        if len(right_lanes) > 0:
            right_boundary = max(right_lanes, key=lambda lane: np.mean([pt[0] for pt in lane]))
        
        # Create polygon
        all_points = []
        
        if left_boundary is not None:
            left_sorted = sorted(left_boundary, key=lambda p: p[1])
            all_points.extend(left_sorted)
            
            # Extend forward
            if len(left_sorted) >= 2:
                top_pt = np.array(left_sorted[0])
                direction = np.array(left_sorted[0]) - np.array(left_sorted[1])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    extended_pt = top_pt + direction * forward_extension
                    extended_pt = np.clip(extended_pt, [0, 0], [img_w - 1, 0]).astype(int)
                    all_points.insert(0, list(extended_pt))
        
        if right_boundary is not None:
            right_sorted = sorted(right_boundary, key=lambda p: p[1], reverse=True)
            
            # Extend forward
            if len(right_sorted) >= 2:
                top_pt = np.array(right_sorted[-1])
                direction = np.array(right_sorted[-1]) - np.array(right_sorted[-2])
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction = direction / direction_norm
                    extended_pt = top_pt + direction * forward_extension
                    extended_pt = np.clip(extended_pt, [0, 0], [img_w - 1, 0]).astype(int)
                    all_points.append(list(extended_pt))
            
            all_points.extend(right_sorted)
        
        # Create convex hull
        if len(all_points) > 2:
            hull = cv2.convexHull(np.array(all_points, dtype=np.int32))
            cv2.fillPoly(mask, [hull], 255)
            self.lane_polygon = hull  # STORE polygon
            print(f"   DEBUG: Lane mask created with {len(hull)} points")  # DEBUG
        else:
            self.lane_polygon = None
            print("   DEBUG: Not enough points for lane mask")  # DEBUG
        
        self.lane_mask = mask  # ALWAYS store mask
    
    def visualize(self, image: np.ndarray, detections: List[Dict], 
                  lane_bounds: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Visualize detections on image"""
        vis = image.copy()
        
        # Draw lane bounds
        if lane_bounds:
            left_x, right_x = lane_bounds
            cv2.line(vis, (left_x, 0), (left_x, vis.shape[0]), (255, 255, 0), 2)
            cv2.line(vis, (right_x, 0), (right_x, vis.shape[0]), (255, 255, 0), 2)
        
        # Draw detections
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            distance = det.get('distance')
            class_name = det['class']
            confidence = det['confidence']
            is_dangerous = det.get('is_dangerous', False)
            in_lane = det.get('in_lane', False)
            
            # Color based on distance and lane position
            if in_lane and is_dangerous:
                if distance is not None:
                    if distance < self.danger_distance:
                        color = (0, 0, 255)  # Red
                    elif distance < self.stop_distance:
                        color = (0, 140, 255)  # Orange
                    else:
                        color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
            else:
                color = (0, 255, 0)  # Green (safe)
            
            thickness = 3 if (in_lane and is_dangerous) else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            
            # Label
            if distance is not None:
                label = f"{class_name} {distance:.1f}m"
            else:
                label = f"{class_name}"
            
            if in_lane:
                label += " [LANE]"
            
            # Draw label
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(vis, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 255), 2)
            
            # Draw center point
            center_x, center_y = det['bbox_center']
            cv2.circle(vis, (center_x, center_y), 5, color, -1)
        
        return vis