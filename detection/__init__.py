"""
Detection module for YOLO-based obstacle and lane detection
"""

from .yolo_distance_detector import YOLODistanceDetector
from .yolo_lane_filter import YOLOLaneFilter

__all__ = ['YOLODistanceDetector', 'YOLOLaneFilter']
