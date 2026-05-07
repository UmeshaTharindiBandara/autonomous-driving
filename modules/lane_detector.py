"""
Lane Detection Module
Ultra-Fast lane detection with ROI filtering and BEV transformation

Improvements over original:
  - get_lane_layout()       : adjacent lane detection for OvertakeManager
  - get_debug_overlay()     : draws all lane info onto a copy of the frame
  - get_lane_confidence()   : per-lane quality score 0-1
  - compute_centerline_curvature() : now returns direction (left/right/straight)
  - reset()                 : clean state reset without re-loading the model
  - _bottom_avg_x()         : extracted as module-level helper (DRY)
  - All constants at top    : easy to tune without touching the class
"""

import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import scipy.special
import torch
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.model import parsingNet
from utils.common import merge_config
from data.constant import culane_row_anchor, tusimple_row_anchor
from core.roi_selector import ROISelector

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------
LANE_WIDTH_M          = 3.7    # standard lane width (metres)
LOOK_Y_OFFSET         = 60     # BEV look-ahead from bottom (pixels)
EMA_ALPHA             = 0.30   # lateral-error EMA weight
MISS_WINDOW           = 20     # rolling window size for lane-health check
MISS_THRESH           = 2      # frames with lanes below this → lane lost
LANE_MEMORY_DURATION  = 2.0    # seconds — keep last good coefficients this long

# Adjacent-lane layout (OvertakeManager)
ADJ_LANE_MIN_WIDTH_PX = 80     # px — narrower gaps ignored as noise
ADJ_LANE_CLEAR_MARGIN = 0.5    # m  — extra safety margin beyond one lane width

# Lane confidence scoring
MIN_LANE_POINTS       = 6      # fewer points → confidence = 0
CONF_FULL_POINTS      = 20     # this many points → max count-score

# Debug overlay colours (BGR)
COLOR_LEFT_LANE   = (0,   255,   0)   # green
COLOR_RIGHT_LANE  = (0,   200, 255)   # yellow
COLOR_ADJ_LANE    = (255, 128,   0)   # orange
COLOR_CENTER_LINE = (0,   0,   255)   # red
COLOR_EGO_CENTER  = (128, 128, 128)   # grey
COLOR_TEXT        = (255, 255, 255)   # white


# ---------------------------------------------------------------------------
# Module-level helper (shared by class methods)
# ---------------------------------------------------------------------------
def _bottom_avg_x(lane: list, n: int = 3) -> float:
    """Average x of the n lowest (highest-y) points in a lane."""
    pts = sorted(lane, key=lambda p: p[1], reverse=True)[:n]
    return sum(p[0] for p in pts) / len(pts) if pts else 0.0


# ---------------------------------------------------------------------------
# LaneDetector
# ---------------------------------------------------------------------------
class LaneDetector:
    """
    Lane detection with BEV transformation, curve fitting, and overtake support.

    Public API
    ----------
    detect(image)                      → dict
    compute_lateral_error(lanes)       → float | None
    compute_centerline_curvature()     → (kappa, direction, classification)
    get_lane_layout(lanes)             → dict   ← NEW (for OvertakeManager)
    get_lane_confidence(lane)          → float  ← NEW
    get_debug_overlay(image, result)   → np.ndarray  ← NEW
    is_lane_lost()                     → bool
    get_lane_lost_duration()           → float
    reset_lane_lost_timer()            → None
    reset()                            → None   ← NEW
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def __init__(self,
                 cfg_path:   str = "configs/tusimple.py",
                 model_path: str = "tusimple_18.pth"):
        self._setup_model(cfg_path, model_path)
        self._setup_bev()
        self.roi_selector = ROISelector()
        self._init_state()
        print("✓ Lane Detector initialized")

    def _init_state(self):
        """Initialise / zero all runtime state."""
        self.err_ema                  = None
        self.lanes_ok_window          = deque(maxlen=MISS_WINDOW)
        self.last_coeff_left          = None
        self.last_coeff_right         = None
        self.lane_center_history      = deque(maxlen=3)
        self.last_good_detection_time = 0.0
        self.lane_lost_start_time     = 0.0

    def reset(self):
        """Full state reset — use between runs without reloading weights."""
        self._init_state()
        print("✓ Lane Detector state reset")

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------
    def _setup_model(self, cfg_path: str, model_path: str):
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
            raise NotImplementedError(f"Unknown dataset: {self.cfg.dataset}")

        self.img_transforms = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                 (0.229, 0.224, 0.225)),
        ])

        self.net = parsingNet(
            pretrained=False,
            backbone=self.cfg.backbone,
            cls_dim=(self.cfg.griding_num + 1, self.cls_num_per_lane, 4),
            use_aux=False,
        ).cuda()

        state_dict = torch.load(model_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']

        compat = {(k[7:] if k.startswith('module.') else k): v
                  for k, v in state_dict.items()}
        self.net.load_state_dict(compat, strict=False)
        self.net.eval()
        print(f"  ✓ Model loaded: {model_path}")

    # ------------------------------------------------------------------
    # BEV setup
    # ------------------------------------------------------------------
    def _setup_bev(self):
        self.bev_h, self.bev_w = 400, 300

        src_pts = np.float32([
            [self.img_w * 0.15, self.img_h * 0.9],
            [self.img_w * 0.85, self.img_h * 0.9],
            [self.img_w * 0.55, self.img_h * 0.6],
            [self.img_w * 0.45, self.img_h * 0.6],
        ])
        dst_pts = np.float32([
            [50,              self.bev_h - 50],
            [self.bev_w - 50, self.bev_h - 50],
            [self.bev_w - 50, 50],
            [50,              50],
        ])

        self.M_bev     = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.M_inv     = cv2.getPerspectiveTransform(dst_pts, src_pts)
        self.px_to_m_y = 25.0 / (self.bev_h - 100)
        self.px_to_m_x = LANE_WIDTH_M / max(20.0, 0.4 * (self.bev_w - 100))

    # ==================================================================
    # PUBLIC API
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. Core detection
    # ------------------------------------------------------------------
    def detect(self, image: np.ndarray) -> dict | None:
        """
        Run lane detection on a BGR image.

        Returns dict:
            filtered_lanes   : list[list[[x,y]]]
            lane_center      : int | None
            lanes_detected   : int
            lane_confidences : list[float]   one per lane
            raw_output       : np.ndarray
        """
        if image is None:
            return None

        img_rgb    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = (self.img_transforms(Image.fromarray(img_rgb))
                      .unsqueeze(0).cuda())

        with torch.no_grad():
            out = self.net(img_tensor)

        col_sample   = np.linspace(0, 800 - 1, self.cfg.griding_num)
        col_sample_w = col_sample[1] - col_sample[0]

        out_j = out[0].data.cpu().numpy()
        out_j = out_j[:, ::-1, :]

        prob  = scipy.special.softmax(out_j[:-1, :, :], axis=0)
        idx   = (np.arange(self.cfg.griding_num) + 1).reshape(-1, 1, 1)
        loc   = np.sum(prob * idx, axis=0)
        out_j = np.argmax(out_j, axis=0)
        loc[out_j == self.cfg.griding_num] = 0
        out_j = loc

        lanes = []
        for i in range(out_j.shape[1]):
            if np.sum(out_j[:, i] != 0) > 2:
                pts = []
                for k in range(out_j.shape[0]):
                    if out_j[k, i] > 0:
                        x = int(out_j[k, i] * col_sample_w * self.img_w / 800) - 1
                        y = int(self.img_h * (
                            self.row_anchor[self.cls_num_per_lane - 1 - k] / 288)) - 1
                        x = max(0, min(x, self.img_w - 1))
                        y = max(0, min(y, self.img_h - 1))
                        pts.append([x, y])
                if len(pts) > 1:
                    lanes.append(pts)

        filtered_lanes = self.roi_selector.filter_lanes_by_roi(lanes, image.shape)
        lane_center    = self._calculate_lane_center(filtered_lanes)
        confidences    = [self.get_lane_confidence(l) for l in filtered_lanes]

        return {
            'filtered_lanes':   filtered_lanes,
            'lane_center':      lane_center,
            'lanes_detected':   len(filtered_lanes),
            'lane_confidences': confidences,
            'raw_output':       out_j,
        }

    # ------------------------------------------------------------------
    # 2. Lateral error
    # ------------------------------------------------------------------
    def compute_lateral_error(self, filtered_lanes: list) -> float | None:
        """
        EMA-smoothed lateral offset of ego from lane centre (metres).
        Positive = ego is right of centre; negative = left of centre.
        """
        bev_sets = self._warp_to_bev(filtered_lanes)

        left_set = right_set = None
        if len(bev_sets) == 1:
            if bev_sets[0][:, 0].mean() < self.bev_w * 0.5:
                left_set = bev_sets[0]
            else:
                right_set = bev_sets[0]
        elif len(bev_sets) >= 2:
            s = sorted(bev_sets, key=lambda a: a[:, 0].mean())
            left_set, right_set = s[0], s[-1]

        coeff_left  = self._fit_lane_curve(left_set)  if left_set  is not None else None
        coeff_right = self._fit_lane_curve(right_set) if right_set is not None else None

        # Fall back to last good coefficients
        if coeff_left  is None: coeff_left  = self.last_coeff_left
        if coeff_right is None: coeff_right = self.last_coeff_right

        y_eval    = self.bev_h - LOOK_Y_OFFSET
        err_m_raw = None
        if coeff_left is not None or coeff_right is not None:
            self._update_px_to_m_x(coeff_left, coeff_right, y_eval)
            err_m_raw = self._lateral_error_m(coeff_left, coeff_right, y_eval)

        # EMA smoothing
        if err_m_raw is not None:
            self.err_ema = (err_m_raw if self.err_ema is None
                            else EMA_ALPHA * err_m_raw + (1 - EMA_ALPHA) * self.err_ema)

        # Persist coefficients
        if coeff_left  is not None: self.last_coeff_left  = coeff_left
        if coeff_right is not None: self.last_coeff_right = coeff_right

        # Health tracking
        lanes_ok = (coeff_left is not None) or (coeff_right is not None)
        self.lanes_ok_window.append(1 if lanes_ok else 0)
        if lanes_ok:
            self.last_good_detection_time = time.time()
            self.lane_lost_start_time     = 0.0
        elif self.lane_lost_start_time == 0.0:
            self.lane_lost_start_time = time.time()

        return self.err_ema

    # ------------------------------------------------------------------
    # 3. Curvature  (now returns direction)
    # ------------------------------------------------------------------
    def compute_centerline_curvature(self) -> tuple:
        """
        Approximate centreline curvature from BEV-fitted lane coefficients.

        Returns
        -------
        (kappa, direction, classification)
          kappa          : float | None  — magnitude in 1/m
          direction      : str  | None  — 'left' | 'right' | 'straight'
          classification : str  | None  — 'straight' | 'gentle' | 'moderate'
                                          | 'sharp' | 'very_sharp'

        Returns (None, None, None) if no coefficients are available.
        """
        cL, cR = self.last_coeff_left, self.last_coeff_right
        if cL is None and cR is None:
            return None, None, None

        px_to_m_x = getattr(self, 'px_to_m_x', None)
        px_to_m_y = getattr(self, 'px_to_m_y', None)
        if not px_to_m_x or not px_to_m_y:
            return None, None, None

        y_eval = self.bev_h - LOOK_Y_OFFSET

        def _kappa_signed(coeff):
            Q2, Q1, _ = coeff
            dx_m  = (2.0 * Q2 * y_eval + Q1) * (px_to_m_x / px_to_m_y)
            d2x_m = (2.0 * Q2) * (px_to_m_x / px_to_m_y ** 2)
            denom = (1.0 + dx_m ** 2) ** 1.5
            return d2x_m / denom if denom > 1e-6 else 0.0

        vals         = [_kappa_signed(c) for c in (cL, cR) if c is not None]
        kappa_signed = sum(vals) / len(vals)
        kappa        = abs(kappa_signed)

        if   kappa < 0.005:       direction = 'straight'
        elif kappa_signed > 0:    direction = 'right'
        else:                     direction = 'left'

        if   kappa < 0.010: cls = 'straight'
        elif kappa < 0.015: cls = 'gentle'
        elif kappa < 0.020: cls = 'moderate'
        elif kappa < 0.025: cls = 'sharp'
        else:               cls = 'very_sharp'

        return kappa, direction, cls

    # ------------------------------------------------------------------
    # 4. Adjacent lane layout  (NEW — for OvertakeManager)
    # ------------------------------------------------------------------
    def get_lane_layout(self, filtered_lanes: list) -> dict:
        """
        Classify all lane markings relative to the ego vehicle and return
        usable clearance info for overtake decisions.

        Returns dict:
            ego_left_x    : int | None   pixel x — inner left boundary
            ego_right_x   : int | None   pixel x — inner right boundary
            adj_left_x    : int | None   pixel x — outer left boundary
            adj_right_x   : int | None   pixel x — outer right boundary
            ego_width_px  : int | None
            ego_width_m   : float | None
            left_clear_m  : float | None  clearance in left adjacent lane
            right_clear_m : float | None  clearance in right adjacent lane
            left_lane_ok  : bool  True if safe to enter left lane
            right_lane_ok : bool  True if safe to enter right lane
            total_lanes   : int

        Example usage in OvertakeManager
        ---------------------------------
            layout = self.lane_detector.get_lane_layout(lane_data['filtered_lanes'])
            if layout['left_lane_ok']:
                self.state = OvertakeState.CHANGE_L
        """
        cx = self.img_w // 2

        left_lanes  = sorted(
            [l for l in filtered_lanes if _bottom_avg_x(l) < cx],
            key=_bottom_avg_x, reverse=True    # closest-to-centre first
        )
        right_lanes = sorted(
            [l for l in filtered_lanes if _bottom_avg_x(l) >= cx],
            key=_bottom_avg_x                  # closest-to-centre first
        )

        ego_left_x  = int(_bottom_avg_x(left_lanes[0]))  if left_lanes           else None
        ego_right_x = int(_bottom_avg_x(right_lanes[0])) if right_lanes          else None
        adj_left_x  = int(_bottom_avg_x(left_lanes[1]))  if len(left_lanes)  > 1 else None
        adj_right_x = int(_bottom_avg_x(right_lanes[1])) if len(right_lanes) > 1 else None

        # Ego lane width
        ego_width_px = ego_width_m = None
        if ego_left_x is not None and ego_right_x is not None:
            ego_width_px = ego_right_x - ego_left_x
            if ego_width_px > ADJ_LANE_MIN_WIDTH_PX:
                ego_width_m = ego_width_px * self.px_to_m_x

        # Left adjacent clearance
        left_clear_m = None
        left_lane_ok = False
        if ego_left_x is not None and adj_left_x is not None:
            gap_px = ego_left_x - adj_left_x
            if gap_px >= ADJ_LANE_MIN_WIDTH_PX:
                left_clear_m = gap_px * self.px_to_m_x
                left_lane_ok = left_clear_m >= (LANE_WIDTH_M + ADJ_LANE_CLEAR_MARGIN)

        # Right adjacent clearance
        right_clear_m = None
        right_lane_ok = False
        if ego_right_x is not None and adj_right_x is not None:
            gap_px = adj_right_x - ego_right_x
            if gap_px >= ADJ_LANE_MIN_WIDTH_PX:
                right_clear_m = gap_px * self.px_to_m_x
                right_lane_ok = right_clear_m >= (LANE_WIDTH_M + ADJ_LANE_CLEAR_MARGIN)

        return {
            'ego_left_x':    ego_left_x,
            'ego_right_x':   ego_right_x,
            'adj_left_x':    adj_left_x,
            'adj_right_x':   adj_right_x,
            'ego_width_px':  ego_width_px,
            'ego_width_m':   ego_width_m,
            'left_clear_m':  left_clear_m,
            'right_clear_m': right_clear_m,
            'left_lane_ok':  left_lane_ok,
            'right_lane_ok': right_lane_ok,
            'total_lanes':   len(filtered_lanes),
        }

    # ------------------------------------------------------------------
    # 5. Lane confidence score  (NEW)
    # ------------------------------------------------------------------
    def get_lane_confidence(self, lane: list) -> float:
        """
        Quality score in [0, 1] for a single detected lane.

        Based on:
          - point count  (more → higher)
          - vertical span (longer lane → higher)
          - straightness  (less lateral scatter → higher)
        """
        if not lane or len(lane) < MIN_LANE_POINTS:
            return 0.0

        n           = len(lane)
        count_score = min(1.0, (n - MIN_LANE_POINTS) /
                               max(1, CONF_FULL_POINTS - MIN_LANE_POINTS))

        ys          = [p[1] for p in lane]
        span_score  = min(1.0, (max(ys) - min(ys)) / (self.img_h * 0.5))

        xs = np.array([p[0] for p in lane], dtype=np.float32)
        ya = np.array(ys,                   dtype=np.float32)
        try:
            coeff       = np.polyfit(ya, xs, 1)
            residuals   = xs - np.polyval(coeff, ya)
            straightness = 1.0 - min(1.0, float(np.std(residuals)) / 30.0)
        except Exception:
            straightness = 0.5

        score = (0.5 * count_score +
                 0.3 * span_score  +
                 0.2 * straightness)
        return round(float(np.clip(score, 0.0, 1.0)), 3)

    # ------------------------------------------------------------------
    # 6. Debug overlay  (NEW)
    # ------------------------------------------------------------------
    def get_debug_overlay(self, image: np.ndarray, result: dict) -> np.ndarray:
        """
        Draw all lane-detection info onto a copy of the BGR frame.

        Draws:
          - Detected lanes (green = left side, yellow = right side)
          - Per-lane confidence score
          - Adjacent lane boundaries (orange dashed lines)
          - Ego lane centre (red) vs image centre (grey)
          - HUD: lateral error, curvature, road class, overtake availability

        Parameters
        ----------
        image  : original BGR frame (not modified)
        result : dict returned by detect()

        Returns
        -------
        Annotated BGR image — safe to pass directly to cv2.imshow()
        """
        vis = image.copy()
        if result is None:
            return vis

        cx               = self.img_w // 2
        filtered_lanes   = result.get('filtered_lanes',   [])
        lane_confidences = result.get('lane_confidences', [])
        lane_center      = result.get('lane_center')

        # --- Lanes ---
        for i, lane in enumerate(filtered_lanes):
            avg_x = _bottom_avg_x(lane)
            color = COLOR_LEFT_LANE if avg_x < cx else COLOR_RIGHT_LANE
            conf  = lane_confidences[i] if i < len(lane_confidences) else 0.0
            pts   = np.array(lane, dtype=np.int32)

            for j in range(len(pts) - 1):
                cv2.line(vis, tuple(pts[j]), tuple(pts[j + 1]), color, 3)

            # Confidence label at topmost point
            top = pts[np.argmin(pts[:, 1])]
            cv2.putText(vis, f"{conf:.2f}", tuple(top),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        # --- Adjacent lane boundaries ---
        layout = self.get_lane_layout(filtered_lanes)
        for x_val, label in [(layout['adj_left_x'],  'adj-L'),
                              (layout['adj_right_x'], 'adj-R')]:
            if x_val is not None:
                cv2.line(vis, (x_val, 0), (x_val, self.img_h),
                         COLOR_ADJ_LANE, 1)
                cv2.putText(vis, label, (x_val + 4, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            COLOR_ADJ_LANE, 1, cv2.LINE_AA)

        # --- Lane centre vs image centre ---
        if lane_center is not None:
            cv2.line(vis, (lane_center, self.img_h - 70),
                     (lane_center, self.img_h), COLOR_CENTER_LINE, 2)
        cv2.line(vis, (cx, self.img_h - 70),
                 (cx, self.img_h), COLOR_EGO_CENTER, 1)

        # --- HUD ---
        lat_err              = self.err_ema
        kappa, direction, cls = self.compute_centerline_curvature()

        def _fmt_clear(side, m_val, ok):
            status = 'OK ✓' if ok else 'BLOCKED'
            dist   = f"  {m_val:.1f}m" if m_val is not None else ''
            return f"{side} adj: {status}{dist}"

        hud = [
            f"Lanes detected : {result.get('lanes_detected', 0)}",
            (f"Lateral error  : {lat_err:+.3f} m"
             if lat_err is not None else "Lateral error  : --"),
            (f"Curvature      : {kappa:.4f} 1/m  [{direction}]"
             if kappa is not None else "Curvature      : --"),
            f"Road class     : {cls or '--'}",
            _fmt_clear('Left',  layout['left_clear_m'],  layout['left_lane_ok']),
            _fmt_clear('Right', layout['right_clear_m'], layout['right_lane_ok']),
            f"Lane lost      : {'YES ⚠' if self.is_lane_lost() else 'no'}",
        ]
        y0 = 18
        for line in hud:
            cv2.putText(vis, line, (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        COLOR_TEXT, 1, cv2.LINE_AA)
            y0 += 22

        return vis

    # ------------------------------------------------------------------
    # 7. Lane health
    # ------------------------------------------------------------------
    def is_lane_lost(self) -> bool:
        """True if lanes have been absent beyond the memory window."""
        if self.last_good_detection_time > 0:
            if time.time() - self.last_good_detection_time < LANE_MEMORY_DURATION:
                return False
        if len(self.lanes_ok_window) == self.lanes_ok_window.maxlen:
            return sum(self.lanes_ok_window) <= MISS_THRESH
        return False

    def get_lane_lost_duration(self) -> float:
        """Seconds since lanes were last reliably detected."""
        return (0.0 if self.lane_lost_start_time == 0.0
                else time.time() - self.lane_lost_start_time)

    def reset_lane_lost_timer(self):
        """Call when stopped at a red light so the timer doesn't fire."""
        self.lane_lost_start_time = 0.0

    # ==================================================================
    # PRIVATE HELPERS
    # ==================================================================

    def _calculate_lane_center(self, lanes: list) -> int | None:
        """Estimate pixel-x of lane centre from detected markings."""
        if not lanes:
            return None
        cx = self.img_w // 2

        if len(lanes) == 1:
            ax = _bottom_avg_x(lanes[0])
            if   ax < cx * 0.3: offset =  250
            elif ax > cx * 1.7: offset = -250
            else:               offset =  200 if ax < cx else -200
            return int(max(cx * 0.5, min(cx * 1.5, ax + offset)))

        left_pts, right_pts = [], []
        for lane in lanes:
            bp = sorted(lane, key=lambda p: p[1], reverse=True)[:3]
            if not bp:
                continue
            ax = sum(p[0] for p in bp) / len(bp)
            (left_pts if ax < cx else right_pts).append(bp[0])

        if left_pts and right_pts:
            lx = max(left_pts,  key=lambda p: p[0])[0]
            rx = min(right_pts, key=lambda p: p[0])[0]
            return (lx + rx) // 2
        if left_pts:
            return max(left_pts,  key=lambda p: p[0])[0] + 180
        if right_pts:
            return min(right_pts, key=lambda p: p[0])[0] - 180
        return None

    def _warp_to_bev(self, lanes: list) -> list:
        """Transform image-space lane point-lists to BEV coordinates."""
        bev_sets = []
        for lane in lanes:
            pts = np.array(lane, dtype=np.float32).reshape(-1, 1, 2)
            w   = cv2.perspectiveTransform(pts, self.M_bev).reshape(-1, 2)
            m   = ((w[:, 0] >= 0) & (w[:, 0] < self.bev_w) &
                   (w[:, 1] >= 0) & (w[:, 1] < self.bev_h))
            w   = w[m]
            if len(w) >= 6:
                bev_sets.append(w)
        return bev_sets

    @staticmethod
    def _linear_prefit(x: np.ndarray, y: np.ndarray):
        A = np.stack([y, np.ones_like(y)], axis=1)
        try:
            (a, b), *_ = np.linalg.lstsq(A, x, rcond=None)
            return a, b
        except Exception:
            return None

    @staticmethod
    def _quad_fit_x_of_y(x: np.ndarray, y: np.ndarray):
        A = np.stack([y ** 2, y, np.ones_like(y)], axis=1)
        try:
            (Q2, Q1, Q0), *_ = np.linalg.lstsq(A, x, rcond=None)
            return Q2, Q1, Q0
        except Exception:
            return None

    def _fit_lane_curve(self, wpts) -> tuple | None:
        """Quadratic x(y) fit with linear-residual outlier rejection."""
        if wpts is None or len(wpts) < 12:
            return None
        y    = wpts[:, 1].astype(np.float32)
        x    = wpts[:, 0].astype(np.float32)
        ab   = self._linear_prefit(x, y)
        if ab is None:
            return None
        a, b  = ab
        resid = np.abs(x - (a * y + b))
        thr   = max(4.0, 2.0 * np.median(resid))
        inl   = resid < thr
        if inl.sum() < 10:
            inl = resid < (thr * 1.5)
        return self._quad_fit_x_of_y(x[inl], y[inl])

    def _update_px_to_m_x(self, coeff_left, coeff_right, y_eval: float):
        """Recalibrate horizontal px→m scale from observed lane width."""
        def x_at(c): return c[0] * y_eval ** 2 + c[1] * y_eval + c[2]
        if coeff_left is not None and coeff_right is not None:
            gap = abs(x_at(coeff_right) - x_at(coeff_left))
            if gap > 5:
                self.px_to_m_x = LANE_WIDTH_M / gap

    def _lateral_error_m(self, coeff_left, coeff_right, y_eval: float) -> float | None:
        """Signed lateral offset of ego from lane centre (metres)."""
        def x_at(c): return c[0] * y_eval ** 2 + c[1] * y_eval + c[2]
        car_cx = self.bev_w * 0.5
        if coeff_left is None and coeff_right is None:
            return None
        if coeff_left is not None and coeff_right is not None:
            lane_cx = 0.5 * (x_at(coeff_left) + x_at(coeff_right))
        elif coeff_left is not None:
            lane_cx = x_at(coeff_left)  + (LANE_WIDTH_M / self.px_to_m_x) * 0.5
        else:
            lane_cx = x_at(coeff_right) - (LANE_WIDTH_M / self.px_to_m_x) * 0.5
        return float((lane_cx - car_cx) * self.px_to_m_x)
