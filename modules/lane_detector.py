"""
Lane Detection Module
Ultra-Fast lane detection with ROI filtering and BEV transformation
"""

import torch
import cv2
import numpy as np
import scipy.special
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.model import parsingNet
from utils.common import merge_config
from data.constant import culane_row_anchor, tusimple_row_anchor
from core.roi_selector import ROISelector  # CHANGED: core instead of utils

# Constants
LANE_WIDTH_M = 3.7
LOOK_Y_OFFSET = 60
EMA_ALPHA = 0.30
MISS_WINDOW = 20
MISS_THRESH = 2


class LaneDetector:
    """Lane detection with BEV transformation and curve fitting"""
    
    def __init__(self, cfg_path="configs/tusimple.py", model_path="tusimple_18.pth"):
        self.setup_model(cfg_path, model_path)
        self.setup_bev()
        self.roi_selector = ROISelector()
        
        # State variables
        self.err_ema = None
        self.lanes_ok_window = deque(maxlen=MISS_WINDOW)
        self.last_coeff_left = None
        self.last_coeff_right = None
        self.lane_center_history = deque(maxlen=3)
        
        print("✓ Lane Detector initialized")
    
    def setup_model(self, cfg_path, model_path):
        """Load lane detection model"""
        sys.argv = ['script_name', cfg_path, '--test_model', model_path]
        args, self.cfg = merge_config()
        
        if self.cfg.dataset == 'CULane':
            self.cls_num_per_lane = 18
            self.img_w, self.img_h = 1640, 590
            self.row_anchor = culane_row_anchor
        elif self.cfg.dataset == 'Tusimple':
            self.cls_num_per_lane = 56
            self.img_w, self.img_h = 1280, 720
            self.row_anchor = tusimple_row_anchor
        else:
            raise NotImplementedError
        
        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        self.net = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            cls_dim=(self.cfg.griding_num + 1, self.cls_num_per_lane, 4),
            use_aux=False
        ).cuda()
        
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        compatible_state_dict = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state_dict.items()
        }
        
        self.net.load_state_dict(compatible_state_dict, strict=False)
        self.net.eval()
        print(f"  ✓ Loaded model from {model_path}")
    
    def setup_bev(self):
        """Setup Bird's Eye View transformation"""
        self.bev_h, self.bev_w = 400, 300
        
        src_pts = np.float32([
            [self.img_w * 0.15, self.img_h * 0.9],
            [self.img_w * 0.85, self.img_h * 0.9],
            [self.img_w * 0.55, self.img_h * 0.6],
            [self.img_w * 0.45, self.img_h * 0.6]
        ])
        
        dst_pts = np.float32([
            [50, self.bev_h - 50],
            [self.bev_w - 50, self.bev_h - 50],
            [self.bev_w - 50, 50],
            [50, 50]
        ])
        
        self.M_bev = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)
        
        self.px_to_m_y = 25.0 / (self.bev_h - 100)
        self.px_to_m_x = LANE_WIDTH_M / max(20.0, 0.4 * (self.bev_w - 100))
    
    def detect(self, image):
        """Detect lanes in image"""
        if image is None:
            return None
        
        # Convert and preprocess
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tensor = self.img_transforms(img_pil).unsqueeze(0).cuda()
        
        # Inference
        with torch.no_grad():
            out = self.net(img_tensor)
        
        # Post-process
        col_sample = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]
        
        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]
        
        prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx = np.arange(self.cfg.griding_num) + 1
        idx = idx.reshape(-1, 1, 1)
        loc = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.cfg.griding_num] = 0
        out_j = loc
        
        # Extract lane points
        lanes = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                lane_points = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        x_coord = int(out_j[k, i] * col_sample_w * self.img_w / 800) - 1
                        y_coord = int(self.img_h * (self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
                        x_coord = max(0, min(x_coord, self.img_w - 1))
                        y_coord = max(0, min(y_coord, self.img_h - 1))
                        lane_points.append([x_coord, y_coord])
                if len(lane_points) > 1:
                    lanes.append(lane_points)
        
        # Filter by ROI
        filtered_lanes = self.roi_selector.filter_lanes_by_roi(lanes, image.shape)
        
        # Calculate lane center
        lane_center = self._calculate_lane_center(filtered_lanes)
        
        return {
            'filtered_lanes': filtered_lanes,
            'lane_center': lane_center,
            'lanes_detected': len(filtered_lanes),
            'raw_output': out_j
        }
    
    def compute_lateral_error(self, filtered_lanes):
        """Compute lateral error from lane center using BEV"""
        # Transform to BEV
        bev_sets = self._warp_to_bev(filtered_lanes)
        
        # Fit curves
        left_set = right_set = None
        if len(bev_sets) == 1:
            if bev_sets[0][:, 0].mean() < self.bev_w * 0.5:
                left_set = bev_sets[0]
            else:
                right_set = bev_sets[0]
        elif len(bev_sets) >= 2:
            s = sorted(bev_sets, key=lambda a: a[:, 0].mean())
            left_set, right_set = s[0], s[-1]
        coeff_left = self._fit_lane_curve(left_set) if left_set is not None else None
        coeff_right = self._fit_lane_curve(right_set) if right_set is not None else None

        # Use last good coefficients if current fit fails
        if coeff_left is None:
            coeff_left = self.last_coeff_left
        if coeff_right is None:
            coeff_right = self.last_coeff_right

        y_eval = self.bev_h - LOOK_Y_OFFSET
        if coeff_left is not None or coeff_right is not None:
            self._update_px_to_m_x(coeff_left, coeff_right, y_eval)
            err_m_raw = self._lateral_error_m(coeff_left, coeff_right, y_eval)
        else:
            err_m_raw = None

        # Apply EMA smoothing
        if err_m_raw is not None:
            if self.err_ema is None: self.err_ema = err_m_raw
            else: self.err_ema = EMA_ALPHA*err_m_raw + (1-EMA_ALPHA)*self.err_ema

        # Update state
        if coeff_left is not None: self.last_coeff_left = coeff_left
        if coeff_right is not None: self.last_coeff_right = coeff_right

        # Track lane health
        lanes_ok = (coeff_left is not None) or (coeff_right is not None)
        self.lanes_ok_window.append(1 if lanes_ok else 0)
        
        return self.err_ema
    
    def is_lane_lost(self):
        """Check if lanes are lost"""
        if len(self.lanes_ok_window) == self.lanes_ok_window.maxlen:
            return sum(self.lanes_ok_window) <= MISS_THRESH
        return False
    
    # Helper methods (copy from ard_man_copy.py)
    def _calculate_lane_center(self, lanes):
        """Calculate lane center point"""
        if len(lanes) == 0:
            return None
        
        center_x = self.img_w // 2
        
        if len(lanes) == 1:
            lane = lanes[0]
            bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:3]
            if len(bottom_points) > 0:
                avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                if avg_x < center_x * 0.3:
                    lane_offset = 250
                elif avg_x > center_x * 1.7:
                    lane_offset = -250
                else:
                    lane_offset = 200 if avg_x < center_x else -200
                estimated_center = avg_x + lane_offset
                estimated_center = max(center_x * 0.5, min(center_x * 1.5, estimated_center))
                return int(estimated_center)
        
        left_lanes, right_lanes = [], []
        for lane in lanes:
            bottom_points = sorted(lane, key=lambda p: p[1], reverse=True)[:3]
            if bottom_points:
                avg_x = sum([p[0] for p in bottom_points]) / len(bottom_points)
                bottom_point = bottom_points[0]
                if avg_x < center_x:
                    left_lanes.append(bottom_point)
                else:
                    right_lanes.append(bottom_point)
        
        if len(left_lanes) > 0 and len(right_lanes) > 0:
            closest_left = max(left_lanes, key=lambda p: p[0])
            closest_right = min(right_lanes, key=lambda p: p[0])
            return (closest_left[0] + closest_right[0]) // 2
        elif len(left_lanes) > 0:
            return max(left_lanes, key=lambda p: p[0])[0] + 180
        elif len(right_lanes) > 0:
            return min(right_lanes, key=lambda p: p[0])[0] - 180
        return None
    
    def _warp_to_bev(self, lanes):
        """Warp lanes to bird's eye view"""
        bev_sets = []
        for lane in lanes:
            pts = np.array(lane, dtype=np.float32).reshape(-1, 1, 2)
            w = cv2.perspectiveTransform(pts, self.M_bev).reshape(-1, 2)
            m = (w[:, 0] >= 0) & (w[:, 0] < self.bev_w) & (w[:, 1] >= 0) & (w[:, 1] < self.bev_h)
            w = w[m]
            if len(w) >= 6:
                bev_sets.append(w)
        return bev_sets
    
    @staticmethod
    def _linear_prefit(x, y):
        """Linear pre-fit for outlier removal"""
        A = np.stack([y, np.ones_like(y)], axis=1)
        try:
            (a, b), *_ = np.linalg.lstsq(A, x, rcond=None)
            return a, b
        except:
            return None
    
    @staticmethod
    def _quad_fit_x_of_y(x, y):
        """Quadratic fit x(y)"""
        A = np.stack([y ** 2, y, np.ones_like(y)], axis=1)
        try:
            (Q2, Q1, Q0), *_ = np.linalg.lstsq(A, x, rcond=None)
            return Q2, Q1, Q0
        except:
            return None
    
    def _fit_lane_curve(self, wpts):
        """Fit lane curve with outlier rejection"""
        if wpts is None or len(wpts) < 12:
            return None
        
        y = wpts[:, 1].astype(np.float32)
        x = wpts[:, 0].astype(np.float32)
        
        ab = self._linear_prefit(x, y)
        if ab is None:
            return None
        
        a, b = ab
        x_lin = a * y + b
        resid = np.abs(x - x_lin)
        thr = max(4.0, 2.0 * np.median(resid))
        inl = resid < thr
        
        if inl.sum() < 10:
            inl = resid < (thr * 1.5)
        
        return self._quad_fit_x_of_y(x[inl], y[inl])
    
    def _update_px_to_m_x(self, coeff_left, coeff_right, y_eval):
        """Update pixel to meter conversion"""
        def x_at(c):
            return c[0] * y_eval * y_eval + c[1] * y_eval + c[2]
        
        if coeff_left is not None and coeff_right is not None:
            xl = x_at(coeff_left)
            xr = x_at(coeff_right)
            gap_px = abs(xr - xl)
            if gap_px > 5:
                self.px_to_m_x = LANE_WIDTH_M / gap_px
    
    def _lateral_error_m(self, coeff_left, coeff_right, y_eval):
        """Calculate lateral error in meters"""
        def x_at(c):
            return c[0] * y_eval * y_eval + c[1] * y_eval + c[2]
        
        car_cx = self.bev_w * 0.5
        
        if coeff_left is None and coeff_right is None:
            return None
        
        if coeff_left is not None and coeff_right is not None:
            lane_cx = 0.5 * (x_at(coeff_left) + x_at(coeff_right))
        elif coeff_left is not None:
            lane_cx = x_at(coeff_left) + (LANE_WIDTH_M / self.px_to_m_x) * 0.5
        else:
            lane_cx = x_at(coeff_right) - (LANE_WIDTH_M / self.px_to_m_x) * 0.5
        
        offset_px = lane_cx - car_cx
        return float(offset_px * self.px_to_m_x)