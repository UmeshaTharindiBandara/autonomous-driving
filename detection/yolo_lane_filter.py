"""
Filter YOLO detections to only include objects within the driving lane
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict

class YOLOLaneFilter:
    """Filter YOLO detections to only include objects within the driving lane"""
    
    def __init__(self, img_width: int = 1280, img_height: int = 720):
        self.img_w = img_width
        self.img_h = img_height
        self.lane_mask = None
        self.lane_polygon = None
        
        # Triangular ROI settings (prevents detecting adjacent lane objects)
        self.use_triangular_roi = True  # Set False to use old parallel estimation
        self.fixed_lane_width_ratio = 0.20  # Lane width as ratio of image width (20%)
        self.vanishing_point_y_ratio = 0.35  # Vanishing point height (35% from top)
        self.vanishing_width_ratio = 0.08  # Width at vanishing point (8% of image)
        
    def create_lane_mask_from_lanes(self, filtered_lanes: List, 
                                     expansion_width: int = 10,
                                     forward_extension: int = 250,
                                     max_vertical_extent_single: float = 0.8,
                                     max_vertical_extent_dual: float = 0.9) -> np.ndarray:
        """
        Create lane mask with forward extension
        
        Args:
            max_vertical_extent_single: Max vertical extent for single lane (0.8 = bottom 80%)
            max_vertical_extent_dual: Max vertical extent for dual lanes (0.9 = bottom 90%)
        """
        mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        
        if len(filtered_lanes) == 0:
            self.lane_mask = mask
            self.lane_polygon = None
            return mask
        
        center_x = self.img_w // 2
        left_lanes = []
        right_lanes = []
        
        for lane in filtered_lanes:
            if len(lane) > 0:
                avg_x = np.mean([pt[0] for pt in lane])
                if avg_x < center_x:
                    left_lanes.append(lane)
                else:
                    right_lanes.append(lane)
        
        left_boundary = None
        right_boundary = None
        
        if len(left_lanes) > 0:
            left_boundary = min(left_lanes, key=lambda lane: np.mean([pt[0] for pt in lane]))
        
        if len(right_lanes) > 0:
            right_boundary = max(right_lanes, key=lambda lane: np.mean([pt[0] for pt in lane]))
        
        # Different handling for single lane vs two lanes
        if left_boundary is not None and right_boundary is not None:
            # Both lanes detected - use 90% vertical extent
            polygon_points = self._create_polygon_from_boundaries_limited(
                left_boundary, right_boundary, expansion_width, forward_extension,
                max_y=int(self.img_h * (1.0 - max_vertical_extent_dual)),  # 90% = top 10% excluded
                use_full_extension=True  # Full forward extension for dual lanes
            )
        elif left_boundary is not None:
            # Only left lane detected
            if self.use_triangular_roi:
                # Use fixed-width triangular ROI (prevents detecting adjacent lane)
                polygon_points = self._create_triangular_roi_from_lane(
                    left_boundary, is_left=True, expansion_width=expansion_width,
                    max_y=int(self.img_h * (1.0 - max_vertical_extent_single))
                )
            else:
                # Old method: estimate parallel lane (may detect adjacent lanes)
                estimated_right = self._estimate_parallel_lane(
                    left_boundary, offset=int(self.img_w * 0.25)
                )
                polygon_points = self._create_polygon_from_boundaries_limited(
                    left_boundary, estimated_right, expansion_width, forward_extension,
                    max_y=int(self.img_h * (1.0 - max_vertical_extent_single)),
                    use_full_extension=False
                )
        elif right_boundary is not None:
            # Only right lane detected
            if self.use_triangular_roi:
                # Use fixed-width triangular ROI (prevents detecting adjacent lane)
                polygon_points = self._create_triangular_roi_from_lane(
                    right_boundary, is_left=False, expansion_width=expansion_width,
                    max_y=int(self.img_h * (1.0 - max_vertical_extent_single))
                )
            else:
                # Old method: estimate parallel lane (may detect adjacent lanes)
                estimated_left = self._estimate_parallel_lane(
                    right_boundary, offset=-int(self.img_w * 0.25)
                )
                polygon_points = self._create_polygon_from_boundaries_limited(
                    estimated_left, right_boundary, expansion_width, forward_extension,
                    max_y=int(self.img_h * (1.0 - max_vertical_extent_single)),
                    use_full_extension=False
                )
        else:
            # No lanes detected - use vehicle-centered triangular ROI
            polygon_points = self._create_default_triangular_roi()
        
        if polygon_points is not None and len(polygon_points) > 0:
            cv2.fillPoly(mask, [polygon_points], 255)
            self.lane_polygon = polygon_points
        
        self.lane_mask = mask
        return mask
    
    def _estimate_parallel_lane(self, reference_lane: List, offset: int) -> List:
        """OLD METHOD: Estimate parallel lane (may detect adjacent lanes)"""
        estimated = []
        for pt in reference_lane:
            estimated.append([pt[0] + offset, pt[1]])
        return estimated
    
    def _create_triangular_roi_from_lane(self, lane: List, is_left: bool, 
                                          expansion_width: int, max_y: int) -> np.ndarray:
        """
        Create triangular/trapezoid ROI with fixed width that narrows toward vanishing point.
        Prevents detecting objects in adjacent lanes when only one lane is visible.
        
        Args:
            lane: Detected lane points
            is_left: True if this is left lane, False if right lane
            expansion_width: Additional expansion for the detected lane side
            max_y: Maximum Y coordinate (top limit)
        """
        # Sort lane points by Y coordinate
        lane_sorted = sorted(lane, key=lambda p: p[1])
        
        # Filter points above max_y
        lane_filtered = [pt for pt in lane_sorted if pt[1] >= max_y]
        if len(lane_filtered) == 0:
            lane_filtered = lane_sorted
        
        # Get bottom and top points of the lane
        bottom_pt = lane_filtered[-1]  # Closest to vehicle
        top_pt = lane_filtered[0]      # Farthest from vehicle
        
        # Calculate fixed lane width at bottom (near vehicle)
        lane_width_bottom = int(self.img_w * self.fixed_lane_width_ratio)
        
        # Calculate width at top (near vanishing point)
        lane_width_top = int(self.img_w * self.vanishing_width_ratio)
        
        # Create trapezoid points
        if is_left:
            # Left lane detected - extend rightward with fixed width
            # Bottom points (near vehicle)
            left_bottom = [max(0, bottom_pt[0] - expansion_width), bottom_pt[1]]
            right_bottom = [min(self.img_w - 1, bottom_pt[0] + lane_width_bottom), bottom_pt[1]]
            
            # Top points (near vanishing point)
            left_top = [max(0, top_pt[0] - expansion_width), top_pt[1]]
            right_top = [min(self.img_w - 1, top_pt[0] + lane_width_top), top_pt[1]]
        else:
            # Right lane detected - extend leftward with fixed width
            # Bottom points (near vehicle)
            left_bottom = [max(0, bottom_pt[0] - lane_width_bottom), bottom_pt[1]]
            right_bottom = [min(self.img_w - 1, bottom_pt[0] + expansion_width), bottom_pt[1]]
            
            # Top points (near vanishing point)
            left_top = [max(0, top_pt[0] - lane_width_top), top_pt[1]]
            right_top = [min(self.img_w - 1, top_pt[0] + expansion_width), top_pt[1]]
        
        # Create trapezoid polygon (counter-clockwise)
        polygon = np.array([
            left_bottom,
            left_top,
            right_top,
            right_bottom
        ], dtype=np.int32)
        
        return polygon
    
    def _create_default_triangular_roi(self) -> np.ndarray:
        """
        Create default triangular ROI centered on vehicle when no lanes detected.
        """
        center_x = self.img_w // 2
        bottom_y = self.img_h - 1
        top_y = int(self.img_h * self.vanishing_point_y_ratio)
        
        # Bottom width (near vehicle)
        bottom_half_width = int(self.img_w * self.fixed_lane_width_ratio / 2)
        
        # Top width (near vanishing point)
        top_half_width = int(self.img_w * self.vanishing_width_ratio / 2)
        
        # Create trapezoid centered on vehicle
        polygon = np.array([
            [max(0, center_x - bottom_half_width), bottom_y],
            [max(0, center_x - top_half_width), top_y],
            [min(self.img_w - 1, center_x + top_half_width), top_y],
            [min(self.img_w - 1, center_x + bottom_half_width), bottom_y]
        ], dtype=np.int32)
        
        return polygon
    
    def _create_polygon_from_boundaries_limited(self, left_lane: List, right_lane: List, 
                                                 expansion: int, forward_extension: int,
                                                 max_y: int = None,
                                                 use_full_extension: bool = True) -> np.ndarray:
        """
        Create polygon with LIMITED vertical extent
        
        Args:
            max_y: Maximum Y coordinate (top limit). Points above this are filtered.
            use_full_extension: If True, use full forward_extension. If False, use 60%.
        """
        left_sorted = sorted(left_lane, key=lambda p: p[1])
        right_sorted = sorted(right_lane, key=lambda p: p[1])
        
        # FILTER points: only keep those below max_y threshold
        if max_y is not None:
            left_sorted = [pt for pt in left_sorted if pt[1] >= max_y]
            right_sorted = [pt for pt in right_sorted if pt[1] >= max_y]
        
        # If no points left after filtering, use original
        if len(left_sorted) == 0:
            left_sorted = sorted(left_lane, key=lambda p: p[1])
        if len(right_sorted) == 0:
            right_sorted = sorted(right_lane, key=lambda p: p[1])
        
        left_top = left_sorted[0]
        right_top = right_sorted[0]
        
        # Calculate direction vectors
        if len(left_sorted) >= 2:
            left_direction_x = left_sorted[0][0] - left_sorted[1][0]
            left_direction_y = left_sorted[0][1] - left_sorted[1][1]
        else:
            left_direction_x = 0
            left_direction_y = -1
        
        if len(right_sorted) >= 2:
            right_direction_x = right_sorted[0][0] - right_sorted[1][0]
            right_direction_y = right_sorted[0][1] - right_sorted[1][1]
        else:
            right_direction_x = 0
            right_direction_y = -1
        
        left_length = max(1.0, np.sqrt(left_direction_x**2 + left_direction_y**2))
        right_length = max(1.0, np.sqrt(right_direction_x**2 + right_direction_y**2))
        
        # Adjust forward extension based on mode
        actual_extension = forward_extension if use_full_extension else forward_extension * 0.6
        
        left_extended = [
            int(left_top[0] + (left_direction_x / left_length) * actual_extension),
            max(max_y if max_y else 0, int(left_top[1] + (left_direction_y / left_length) * actual_extension))
        ]
        
        right_extended = [
            int(right_top[0] + (right_direction_x / right_length) * actual_extension),
            max(max_y if max_y else 0, int(right_top[1] + (right_direction_y / right_length) * actual_extension))
        ]
        
        # Create expanded boundaries
        left_expanded = [[max(0, pt[0] - expansion), pt[1]] for pt in left_sorted]
        right_expanded = [[min(self.img_w - 1, pt[0] + expansion), pt[1]] for pt in right_sorted]
        
        # Add extended points
        left_expanded.insert(0, [max(0, left_extended[0] - expansion), left_extended[1]])
        right_expanded.insert(0, [min(self.img_w - 1, right_extended[0] + expansion), right_extended[1]])
        
        polygon = left_expanded + right_expanded[::-1]
        
        return np.array(polygon, dtype=np.int32)
    
    def filter_detections_by_lane(self, yolo_detections: List[Dict], 
                                   overlap_threshold: float = 0.3) -> List[Dict]:
        if self.lane_mask is None or len(yolo_detections) == 0:
            return yolo_detections
        
        filtered = []
        
        for detection in yolo_detections:
            bbox = detection.get('bbox')
            if bbox is None:
                continue
            
            x1, y1, x2, y2 = bbox
            
            x1 = max(0, min(int(x1), self.img_w - 1))
            y1 = max(0, min(int(y1), self.img_h - 1))
            x2 = max(0, min(int(x2), self.img_w - 1))
            y2 = max(0, min(int(y2), self.img_h - 1))
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            bbox_area = (x2 - x1) * (y2 - y1)
            if bbox_area == 0:
                continue
            
            mask_region = self.lane_mask[y1:y2, x1:x2]
            lane_pixels = np.sum(mask_region > 0)
            
            overlap_ratio = lane_pixels / bbox_area
            
            detection['lane_overlap'] = overlap_ratio
            detection['in_lane'] = overlap_ratio >= overlap_threshold
            
            if overlap_ratio >= overlap_threshold:
                filtered.append(detection)
        
        return filtered
    
    def get_lane_center_bottom(self) -> Optional[int]:
        if self.lane_polygon is None or len(self.lane_polygon) == 0:
            return None
        
        bottom_points = [pt for pt in self.lane_polygon if pt[1] > self.img_h * 0.7]
        
        if len(bottom_points) == 0:
            return None
        
        avg_x = int(np.mean([pt[0] for pt in bottom_points]))
        return avg_x
    
    def visualize_lane_mask(self, img: np.ndarray, alpha: float = 0.3) -> np.ndarray:
        if self.lane_mask is None:
            return img
        
        overlay = img.copy()
        
        mask_colored = np.zeros_like(img)
        mask_colored[self.lane_mask > 0] = [0, 255, 0]
        
        cv2.addWeighted(overlay, 1 - alpha, mask_colored, alpha, 0, overlay)
        
        if self.lane_polygon is not None and len(self.lane_polygon) > 0:
            cv2.polylines(overlay, [self.lane_polygon], True, (0, 255, 255), 2)
        
        return overlay
    
    def get_lane_bounds(self) -> Optional[Tuple[int, int]]:
        if self.lane_polygon is None or len(self.lane_polygon) == 0:
            return None
        
        bottom_points = [pt for pt in self.lane_polygon if pt[1] > self.img_h * 0.8]
        
        if len(bottom_points) < 2:
            return None
        
        x_coords = [pt[0] for pt in bottom_points]
        return (min(x_coords), max(x_coords))