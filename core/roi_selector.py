"""
ROI (Region of Interest) Selector
Allows user to define and manage lane detection region
"""

import cv2
import numpy as np
import csv
import os


class ROISelector:
    """Class for selecting and managing ROI points"""
    
    def __init__(self, csv_file="roi_points.csv"):
        self.csv_file = csv_file
        self.roi_points = []
        self.temp_points = []
        self.selecting = False
        self.window_name = "Select ROI Points"
    
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.temp_points) < 3:
            self.temp_points.append((x, y))
            print(f"Point {len(self.temp_points)}: ({x}, {y})")
            img_copy = param.copy()
            for i, point in enumerate(self.temp_points):
                cv2.circle(img_copy, point, 5, (0, 255, 0), -1)
                label = "Left" if i == 0 else ("Right" if i == 1 else "Top")
                cv2.putText(img_copy, label, (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if i > 0:
                    cv2.line(img_copy, self.temp_points[i - 1], point, (0, 255, 0), 2)
                if len(self.temp_points) == 3:
                    cv2.line(img_copy, self.temp_points[2], self.temp_points[0], (0, 255, 0), 2)
            cv2.imshow(self.window_name, img_copy)
    
    def load_from_csv(self):
        if os.path.exists(self.csv_file):
            try:
                with open(self.csv_file, 'r') as file:
                    reader = csv.reader(file)
                    _ = next(reader)
                    row = next(reader)
                    self.roi_points = [
                        (int(row[0]), int(row[1])),
                        (int(row[2]), int(row[3])),
                        (int(row[4]), int(row[5]))
                    ]
                print(f"✓ ROI points loaded from {self.csv_file}")
                print(f"   Left: {self.roi_points[0]}, Right: {self.roi_points[1]}, Top: {self.roi_points[2]}")
                return True
            except Exception as e:
                print(f"⚠️ Error loading ROI points: {e}")
                return False
        return False
    
    def save_to_csv(self):
        try:
            with open(self.csv_file, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['left_x', 'left_y', 'right_x', 'right_y', 'top_x', 'top_y'])
                writer.writerow([
                    self.roi_points[0][0], self.roi_points[0][1],
                    self.roi_points[1][0], self.roi_points[1][1],
                    self.roi_points[2][0], self.roi_points[2][1]
                ])
            print(f"✓ ROI points saved to {self.csv_file}")
            return True
        except Exception as e:
            print(f"⚠️ Error saving ROI points: {e}")
            return False
    
    def select_roi(self, sample_image):
        print("\n🔧 ROI Selection Mode")
        print("1) Left-bottom  2) Right-bottom  3) Top-center")
        print("[r] reset  [s] save  [q] cancel")
        img_copy = sample_image.copy()
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback, img_copy)
        cv2.imshow(self.window_name, img_copy)
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.temp_points = []
                cv2.imshow(self.window_name, sample_image)
                print("↺ Selection reset")
            elif key == ord('s') and len(self.temp_points) == 3:
                self.roi_points = self.temp_points.copy()
                self.save_to_csv()
                cv2.destroyWindow(self.window_name)
                return True
            elif key == ord('q'):
                cv2.destroyWindow(self.window_name)
                return False
    
    def create_roi_mask(self, img_shape):
        mask = np.zeros(img_shape[:2], dtype=np.uint8)
        if len(self.roi_points) == 3:
            pts = np.array(self.roi_points, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask
    
    def filter_lanes_by_roi(self, lanes, img_shape):
        if len(self.roi_points) != 3 or len(lanes) == 0:
            return lanes
        
        mask = self.create_roi_mask(img_shape)
        filtered_lanes = []
        
        for lane in lanes:
            points_in_roi = 0
            total_points = len(lane)
            bottom_half = sorted(lane, key=lambda p: p[1], reverse=True)[:total_points // 2 + 1]
            
            for point in bottom_half:
                if 0 <= point[0] < img_shape[1] and 0 <= point[1] < img_shape[0]:
                    if mask[point[1], point[0]] > 0:
                        points_in_roi += 1
            
            threshold = 0.3 if len(lanes) == 1 else 0.4
            if points_in_roi > len(bottom_half) * threshold:
                filtered_lanes.append(lane)
        
        if len(filtered_lanes) > 2:
            center_x = img_shape[1] // 2
            lane_scores = []
            for lane in filtered_lanes:
                bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:5]
                if len(bottom_points) > 0:
                    avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                    distance_to_center = abs(avg_x - center_x)
                    score = distance_to_center - len(lane) * 2
                    lane_scores.append((lane, score))
            lane_scores.sort(key=lambda x: x[1])
            filtered_lanes = [ls[0] for ls in lane_scores[:2]]
        
        return filtered_lanes