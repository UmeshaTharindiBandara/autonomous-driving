"""
Traffic Light Detection Module
Integrates YOLO-based traffic light detection with vehicle control
"""

import numpy as np
import cv2
import time
from collections import deque, Counter
from typing import Optional, Tuple, List, Dict
import carla

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ Ultralytics YOLO not available. Traffic light detection disabled.")


class TrafficLightDetector:
    """
    Traffic light detector using YOLO model with enhanced stopping behavior
    """
    
    def __init__(self, model_path: str = "traffic_light.pt", 
                 class_names: List[str] = None):
        """
        Initialize traffic light detector
        
        Args:
            model_path: Path to YOLO model weights
            class_names: List of class names (default: ["green", "red", "yellow"])
        """
        self.model = None
        self.model_path = model_path
        self.class_names = class_names or ["green", "red", "yellow"]
        
        # Detection settings
        self.conf_thresh = 0.5
        self.iou_thresh = 0.45
        self.min_box_pixels = 16 * 16
        
        # ROI/Zoom settings
        self.use_roi_zoom = True
        self.use_dual_detection = True
        self.use_multiple_rois = True  # Enable multiple ROI regions
        
        # Multiple ROI definitions (top, bottom, left, right ratios)
        # You can define 2 or more ROIs to cover different areas
        self.roi_definitions = [
            # ROI 1: Upper center (main traffic lights ahead)
            {
                'name': 'Center',
                'top': 0.05,
                'bottom': 0.25,
                'left': 0.30,
                'right': 0.60,
                'zoom': 1.75
            },
            # ROI 2: Upper right (traffic lights on right side)
            {
                'name': 'Right',
                'top': 0.10,
                'bottom': 0.30,
                'left': 0.55,
                'right': 0.70,
                'zoom': 1.25
            },
        ]
        
        # Legacy single ROI settings (used when use_multiple_rois = False)
        self.roi_top_ratio = 0.30
        self.roi_bottom_ratio = 0.6
        self.roi_left_ratio = 0.3
        self.roi_right_ratio = 0.8
        self.zoom_scale = 1.5
        
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
        self.last_danger_state_time = 0.0  # last time we saw RED or YELLOW
        self.last_detection_time = 0.0     # last time we saw any TL state
        self.provisional_red_start = 0.0   # start time of a provisional red (for confirmation)
        
        # Control parameters
        self.red_light_brake_force = 0.9
        self.yellow_light_brake_force = 0.5
        self.green_light_delay = 0.5
        # Hold time after last RED/YELLOW before resuming when no detections
        self.no_detection_resume_time = 1.0  # seconds (requested behavior)
        self.resume_throttle = 0.5
        self.resume_duration = 2.0
        self.min_stop_speed = 0.5  # km/h
        self.red_confirm_time = 0.3  # seconds of continuous red needed before full stop
        self.ignore_low_y_ratio = 0.15  # ignore detections whose bottom is below this ratio of ROI height (vehicles)
        
        # Load model
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_path)
                print(f"✓ Traffic light detector loaded: {model_path}")
            except Exception as e:
                print(f"⚠️ Failed to load traffic light model: {e}")
                self.model = None
        else:
            print("⚠️ YOLO not available - traffic light detection disabled")
    
    def is_available(self) -> bool:
        """Check if detector is available"""
        return self.model is not None
    
    def detect(self, image: np.ndarray) -> Dict:
        """
        Detect traffic lights in image
        
        Returns:
            Dict with:
                - detected_boxes: List of (bbox, class_name, confidence, source)
                - model_state: Detected state ('red'/'green'/'yellow' or None)
                - visualization: Image with detections drawn
        """
        if not self.is_available():
            return {
                'detected_boxes': [],
                'model_state': None,
                'visualization': image.copy()
            }
        
        img_vis = image.copy()
        now = time.time()
        
        # Prepare detection images
        detection_images = self._prepare_detection_images(image)
        
        # Run detection on all images
        all_detected_boxes = []
        for detection_img, offset, scale, source in detection_images:
            boxes = self._run_detection(detection_img, offset, scale, source)
            all_detected_boxes.extend(boxes)
        
        # Remove duplicates
        unique_boxes = self._remove_duplicate_detections(all_detected_boxes)
        # Track whether we had any raw detections this frame
        self.had_detection_this_frame = len(unique_boxes) > 0
        if self.had_detection_this_frame:
            self.last_raw_detection_time = now
        
        # Find best detection
        best_conf = -1.0
        best_cls_name = None
        for box_coords, cls_name, conf, source in unique_boxes:
            if conf > best_conf:
                best_conf = conf
                best_cls_name = cls_name
        
        model_state_frame = best_cls_name if best_cls_name else None
        
        # Update smoothing
        if model_state_frame:
            self.recent_model_states.append(model_state_frame)
        
        model_state = self._majority_state(self.recent_model_states)
        
        # If no detections have been seen for longer than resume time, flush smoothed state
        if not self.had_detection_this_frame:
            gap = now - self.last_raw_detection_time if self.last_raw_detection_time > 0 else 999
            if gap > self.no_detection_resume_time:
                self.recent_model_states.clear()
                model_state = None
        
        # Draw detections
        for box_coords, cls_name, conf, source in unique_boxes:
            img_vis = self._draw_detection(img_vis, box_coords, cls_name, conf, source)
        
        # Draw ROI if enabled
        if self.use_roi_zoom:
            img_vis = self._draw_roi_box(img_vis)
        
        return {
            'detected_boxes': unique_boxes,
            'model_state': model_state,
            'raw_model_state': model_state_frame,
            'visualization': img_vis
        }
    
    def get_control_decision(self, model_state: Optional[str], 
                             carla_state: Optional[str],
                             vehicle_speed: float) -> Tuple[str, str, float]:
        """
        Get control decision based on traffic light state
        
        Args:
            model_state: State from model ('red'/'green'/'yellow' or None)
            carla_state: State from CARLA API (if available)
            vehicle_speed: Current vehicle speed in km/h
            
        Returns:
            Tuple of (decision_text, control_action, brake_force)
            control_action: 'stop', 'resume', 'slow', 'drive'
        """
        # Decide effective state (prefer CARLA if both present and differ)
        effective = self._decide_effective_state(model_state, carla_state)
        
        is_stopped = vehicle_speed < self.min_stop_speed
        now = time.time()

        # Track last raw detection timestamps only when we actually saw detections
        if self.had_detection_this_frame:
            self.last_detection_time = now

        # RED logic
        if effective == "red":
            if self.provisional_red_start == 0.0:
                self.provisional_red_start = now
            elapsed_red = now - self.provisional_red_start
            # Only treat as danger if we saw a detection this frame
            if self.had_detection_this_frame:
                self.last_red_time = now
                self.last_danger_state_time = now
            self.green_detected_time = 0.0

            if elapsed_red < self.red_confirm_time:
                brake_force = min(0.4, self._calculate_progressive_brake(vehicle_speed))
                decision = f"RED (confirming {self.red_confirm_time - elapsed_red:.2f}s)"
                return decision, "stop", brake_force
            # Confirmed
            if not self.stopped_for_red:
                self.stopped_for_red = True
                decision = "RED CONFIRMED - STOPPING"
                brake_force = self._calculate_progressive_brake(vehicle_speed)
                return decision, "stop", brake_force
            # Already stopping or stopped
            if is_stopped:
                return "STOPPED AT RED", "stop", 1.0
            brake_force = self._calculate_progressive_brake(vehicle_speed)
            return "STOPPING FOR RED", "stop", brake_force

        # GREEN / YELLOW logic
        if effective in ("green", "yellow"):
            # Yellow while stopped at previous red
            if self.stopped_for_red and effective == "yellow" and is_stopped:
                return "YELLOW (STOPPED - HOLDING)", "stop", 1.0

            if effective == "green" and self.stopped_for_red:
                if self.green_detected_time == 0.0:
                    self.green_detected_time = now
                time_since_green = now - self.green_detected_time
                if time_since_green < self.green_light_delay:
                    return f"GREEN - WAITING ({self.green_light_delay - time_since_green:.1f}s)", "stop", 1.0
                # Resume phase
                if self.resume_start_time == 0.0:
                    self.resume_start_time = now
                time_since_resume = now - self.resume_start_time
                if time_since_resume < self.resume_duration:
                    return "GREEN - RESUMING", "resume", 0.0
                # Fully resumed
                self.stopped_for_red = False
                self.green_detected_time = 0.0
                self.resume_start_time = 0.0
                self.provisional_red_start = 0.0
                return "GREEN - DRIVING", "drive", 0.0

            if effective == "yellow":
                if self.had_detection_this_frame:
                    self.last_danger_state_time = now
                return "YELLOW - SLOWING DOWN", "slow", self.yellow_light_brake_force

            # Green or yellow while not stopped
            self.provisional_red_start = 0.0
            self.resume_start_time = 0.0
            return "DRIVING", "drive", 0.0

        # UNKNOWN / NONE logic
        time_since_last_danger = now - self.last_danger_state_time if self.last_danger_state_time > 0 else 999
        if self.stopped_for_red and is_stopped:
            if time_since_last_danger < self.no_detection_resume_time:
                return "STOPPED (holding - no TL seen)", "stop", 1.0
            # Resume after gap
            self.stopped_for_red = False
            self.green_detected_time = 0.0
            self.resume_start_time = 0.0
            self.provisional_red_start = 0.0
            return "NO DETECTION - CAUTIOUS RESUME", "drive", 0.0

        # No detection, not stopped
        if self.stopped_for_red:
            self.stopped_for_red = False
            self.green_detected_time = 0.0
            self.resume_start_time = 0.0
            self.provisional_red_start = 0.0
        return "NO DETECTION", "drive", 0.0
    
    def reset_state(self):
        """Reset detector state"""
        self.stopped_for_red = False
        self.green_detected_time = 0.0
        self.resume_start_time = 0.0
        self.last_red_time = 0.0
        self.last_danger_state_time = 0.0
        self.last_detection_time = 0.0
        self.provisional_red_start = 0.0
        self.recent_model_states.clear()
    
    # Private helper methods
    
    def _prepare_detection_images(self, image: np.ndarray) -> List[Tuple]:
        """Prepare images for detection (ROI, zoom, etc.)"""
        detection_images = []
        
        if self.use_roi_zoom:
            if self.use_multiple_rois and self.roi_definitions:
                # Use multiple ROI definitions
                for roi_def in self.roi_definitions:
                    roi_name = roi_def['name']
                    roi_img, roi_offset = self._extract_roi_custom(
                        image,
                        roi_def['top'],
                        roi_def['bottom'],
                        roi_def['left'],
                        roi_def['right']
                    )
                    
                    # Add non-zoomed ROI detection
                    if self.use_dual_detection:
                        detection_images.append((roi_img, roi_offset, 1.0, f"ROI-{roi_name}"))
                    
                    # Add zoomed ROI detection if zoom is specified
                    zoom_scale = roi_def.get('zoom', 1.0)
                    if zoom_scale > 1.0:
                        zoomed_img, zoom_offset, scale = self._zoom_image(roi_img, zoom_scale)
                        combined_offset = (roi_offset[0] + zoom_offset[0], roi_offset[1] + zoom_offset[1])
                        detection_images.append((zoomed_img, combined_offset, scale, f"Zoom-{roi_name}"))
                    elif not self.use_dual_detection:
                        # If no zoom and no dual, still add the ROI
                        detection_images.append((roi_img, roi_offset, 1.0, f"ROI-{roi_name}"))
            else:
                # Use legacy single ROI settings
                roi_img, roi_offset = self._extract_roi(image)
                
                # Add non-zoomed ROI
                if self.use_dual_detection:
                    detection_images.append((roi_img, roi_offset, 1.0, "ROI"))
                
                # Add zoomed ROI
                if self.zoom_scale > 1.0:
                    zoomed_img, zoom_offset, scale = self._zoom_image(roi_img, self.zoom_scale)
                    combined_offset = (roi_offset[0] + zoom_offset[0], 
                                     roi_offset[1] + zoom_offset[1])
                    detection_images.append((zoomed_img, combined_offset, scale, "Zoomed"))
                else:
                    detection_images.append((roi_img, roi_offset, 1.0, "ROI"))
        else:
            detection_images.append((image, (0, 0), 1.0, "Full"))
        
        return detection_images
    
    def _extract_roi(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract region of interest using default settings"""
        return self._extract_roi_custom(
            image, 
            self.roi_top_ratio, 
            self.roi_bottom_ratio,
            self.roi_left_ratio, 
            self.roi_right_ratio
        )
    
    def _extract_roi_custom(self, image: np.ndarray, 
                           top_ratio: float, bottom_ratio: float,
                           left_ratio: float, right_ratio: float) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Extract region of interest with custom ratios"""
        h, w = image.shape[:2]
        y1 = int(h * top_ratio)
        y2 = int(h * bottom_ratio)
        x1 = int(w * left_ratio)
        x2 = int(w * right_ratio)
        
        roi = image[y1:y2, x1:x2].copy()
        return roi, (x1, y1)
    
    def _zoom_image(self, image: np.ndarray, zoom_scale: float = None) -> Tuple[np.ndarray, Tuple[int, int], float]:
        """Zoom into center of image"""
        if zoom_scale is None:
            zoom_scale = self.zoom_scale
            
        h, w = image.shape[:2]
        new_h, new_w = int(h / zoom_scale), int(w / zoom_scale)
        
        y1 = (h - new_h) // 2
        y2 = y1 + new_h
        x1 = (w - new_w) // 2
        x2 = x1 + new_w
        
        cropped = image[y1:y2, x1:x2]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return zoomed, (x1, y1), zoom_scale
    
    def _run_detection(self, image: np.ndarray, offset: Tuple[int, int], 
                       scale: float, source: str) -> List[Tuple]:
        """Run YOLO detection on image"""
        boxes = []
        
        results = self.model(
            image,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            verbose=False,
            stream=False
        )
        
        if results:
            r = results[0]
            if hasattr(r, "boxes") and r.boxes is not None:
                for box in r.boxes:
                    if box.cls is None:
                        continue
                    cls_id = int(box.cls[0])
                    if not (0 <= cls_id < len(self.class_names)):
                        continue
                    cls_name = self.class_names[cls_id]
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = box.xyxy[0]
                    w, h = float(x2 - x1), float(y2 - y1)
                    
                    if w * h < self.min_box_pixels:
                        continue
                    # Filter boxes that are too low within the detection image (likely vehicles)
                    det_h = image.shape[0]
                    bottom_ratio = float(y2) / float(det_h)
                    if bottom_ratio > (1.0 - self.ignore_low_y_ratio):
                        continue
                    
                    # Map back to original coordinates
                    orig_box = self._map_bbox_to_original([x1, y1, x2, y2], offset, scale)
                    boxes.append((orig_box, cls_name, conf, source))
        
        return boxes
    
    def _map_bbox_to_original(self, bbox: List, offset: Tuple[int, int], 
                              scale: float) -> Tuple[int, int, int, int]:
        """Map bounding box back to original coordinates"""
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
        x_off, y_off = offset
        
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
    
    def _remove_duplicate_detections(self, all_boxes: List[Tuple]) -> List[Tuple]:
        """Remove duplicate detections using IoU"""
        unique_boxes = []
        
        for box_data in all_boxes:
            box_coords, cls_name, conf, source = box_data
            
            is_duplicate = False
            for i, (existing_box, existing_cls, existing_conf, existing_source) in enumerate(unique_boxes):
                iou = self._calculate_iou(box_coords, existing_box)
                if iou > 0.5:
                    # Keep higher confidence
                    if conf > existing_conf:
                        unique_boxes[i] = (box_coords, cls_name, conf, source)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_boxes.append((box_coords, cls_name, conf, source))
        
        return unique_boxes
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _majority_state(self, history: deque) -> Optional[str]:
        """Get majority state from history"""
        if not history:
            return None
        c = Counter(history)
        pref = ["red", "green", "yellow"]
        best_count = max(c.values())
        candidates = [s for s, k in c.items() if k == best_count]
        for p in pref:
            if p in candidates:
                return p
        return candidates[0]
    
    def _decide_effective_state(self, model_state: Optional[str], 
                               carla_state: Optional[str]) -> Optional[str]:
        """Decide effective state (CARLA overrides model on mismatch)"""
        if carla_state and model_state and carla_state != model_state:
            return carla_state
        return model_state or carla_state
    
    def _calculate_progressive_brake(self, vehicle_speed: float) -> float:
        """Calculate progressive brake force based on speed"""
        if vehicle_speed > 30:
            return 0.8
        else:
            return min(1.0, self.red_light_brake_force + 
                      (1.0 - self.red_light_brake_force) * (1.0 - vehicle_speed / 30))
    
    def _draw_detection(self, image: np.ndarray, box_coords: Tuple, 
                       cls_name: str, conf: float, source: str) -> np.ndarray:
        """Draw detection box on image"""
        x1, y1, x2, y2 = box_coords
        
        # Color based on state
        if cls_name == "red":
            color = (0, 0, 255)
        elif cls_name == "green":
            color = (0, 255, 0)
        elif cls_name == "yellow":
            color = (0, 255, 255)
        else:
            color = (255, 0, 255)
        
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        
        # Label
        label = f"{cls_name.upper()} {conf:.2f} [{source}]"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(image, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        cv2.putText(image, label, (x1, y1 - 6),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        
        return image
    
    def _draw_roi_box(self, image: np.ndarray) -> np.ndarray:
        """Draw ROI box on image"""
        h, w = image.shape[:2]
        
        if self.use_multiple_rois and self.roi_definitions:
            # Draw all ROI boxes with different colors
            roi_colors = [(255, 255, 0), (0, 255, 255), (255, 0, 255), (255, 128, 0)]
            for idx, roi_def in enumerate(self.roi_definitions):
                roi_x1 = int(w * roi_def['left'])
                roi_y1 = int(h * roi_def['top'])
                roi_x2 = int(w * roi_def['right'])
                roi_y2 = int(h * roi_def['bottom'])
                color = roi_colors[idx % len(roi_colors)]
                cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), color, 2)
                label = f"ROI-{roi_def['name']}"
                if roi_def.get('zoom', 1.0) > 1.0:
                    label += f" ({roi_def['zoom']}x)"
                cv2.putText(image, label, (roi_x1 + 5, roi_y1 + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        else:
            # Draw single ROI box
            roi_x1 = int(w * self.roi_left_ratio)
            roi_y1 = int(h * self.roi_top_ratio)
            roi_x2 = int(w * self.roi_right_ratio)
            roi_y2 = int(h * self.roi_bottom_ratio)
            
            cv2.rectangle(image, (roi_x1, roi_y1), (roi_x2, roi_y2), 
                         (255, 255, 0), 2)
            cv2.putText(image, "ROI", (roi_x1 + 5, roi_y1 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        return image
    
    @staticmethod
    def carla_tl_to_str(tl_state: carla.TrafficLightState) -> Optional[str]:
        """Convert CARLA traffic light state to string"""
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
