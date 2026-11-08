"""Cameras: Intel RealSense (color + aligned depth), any OpenCV webcam (color only), or a mock."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

from .sim import WORLD

log = logging.getLogger(__name__)


@dataclass
class Frame:
    color: np.ndarray                 # H x W x 3, BGR, uint8
    depth: np.ndarray | None = None   # H x W float32 metres, aligned to color, NaN = no data
    fx: float | None = None           # color intrinsics in pixels
    fy: float | None = None
    t: float = 0.0


class RealSenseCamera:
    """Streams continuously; capture() drains frames so exposure settles after the LED strip turns on."""

    def __init__(self, cfg: dict):
        import pyrealsense2 as rs  # noqa: F401  (fails here if not installed)
        self.cfg = cfg
        self._lock = threading.Lock()
        self.pipe = None

    def start(self):
        import pyrealsense2 as rs
        w, h = self.cfg["color_size"]
        dw, dh = self.cfg["depth_size"]
        fps = self.cfg["fps"]
        self.pipe = rs.pipeline()
        c = rs.config()
        c.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        c.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, fps)
        profile = self.pipe.start(c)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.fx, self.fy = intr.fx, intr.fy
        self.align = rs.align(rs.stream.color)
        self.filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
        log.info("RealSense started: color %dx%d, depth %dx%d", w, h, dw, dh)

    def capture(self) -> Frame:
        with self._lock:
            if self.pipe is None:
                self.start()
            for _ in range(self.cfg["warmup_frames"]):
                self.pipe.wait_for_frames(5000)
            depth_acc = []
            color = None
            for _ in range(5):  # average 5 aligned depth frames to cut noise
                frames = self.align.process(self.pipe.wait_for_frames(5000))
                d = frames.get_depth_frame()
                for f in self.filters:
                    d = f.process(d)
                depth_acc.append(np.asanyarray(d.get_data()).astype(np.float32) * self.depth_scale)
                color = np.asanyarray(frames.get_color_frame().get_data()).copy()
            depth = np.stack(depth_acc)
            depth[depth <= 0] = np.nan
            with np.errstate(all="ignore"):
                depth = np.nanmedian(depth, axis=0).astype(np.float32)
            return Frame(color=color, depth=depth, fx=self.fx, fy=self.fy, t=time.time())

    def close(self):
        if self.pipe is not None:
            self.pipe.stop()
            self.pipe = None


class OpenCVCamera:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.cap = None

    def capture(self) -> Frame:
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.cfg["opencv_index"])
            w, h = self.cfg["color_size"]
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        img = None
        for _ in range(self.cfg["warmup_frames"]):
            ok, img = self.cap.read()
        if img is None:
            raise RuntimeError("OpenCV camera returned no image")
        return Frame(color=img, t=time.time())

    def close(self):
        if self.cap is not None:
            self.cap.release()


class MockCamera:
    """Draws the simulated fruit on a dark floor with a matching depth map (camera 0.19 m above the floor)."""
    FLOOR_M = 0.188

    def __init__(self, cfg: dict, delay: float = 0.2):
        self.cfg = cfg
        self.delay = delay

    def capture(self) -> Frame:
        w, h = self.cfg["color_size"]
        rng = np.random.default_rng(WORLD.seed)
        img = np.full((h, w, 3), 24, np.uint8)
        img = cv2.add(img, rng.integers(0, 6, img.shape, dtype=np.uint8))
        cv2.circle(img, (w // 2, h // 2), int(0.27 * h), (40, 110, 220), -1)  # cradle flange (orange)
        depth = np.full((h, w), self.FLOOR_M, np.float32)
        fx = fy = 0.95 * w
        if WORLD.fruit_present:
            r = WORLD.ripeness
            # peel colour: green -> yellow -> spotted -> brown/black (BGR)
            stops = [(0.0, (40, 150, 70)), (0.3, (40, 200, 225)), (0.6, (35, 165, 200)), (1.0, (25, 45, 70))]
            for (a, ca), (b, cb) in zip(stops, stops[1:]):
                if a <= r <= b:
                    k = (r - a) / (b - a)
                    base = tuple(int(ca[i] + k * (cb[i] - ca[i])) for i in range(3))
                    break
            mask = np.zeros((h, w), np.uint8)
            ax = (int(0.30 * w), int(0.11 * h)) if WORLD.fruit_type == "banana" else (int(0.12 * w), int(0.16 * h))
            cv2.ellipse(mask, (w // 2, h // 2), ax, 15, 0, 360, 255, -1)
            img[mask > 0] = base
            n_spots = int(max(0, r - 0.45) * 120)
            ys, xs = np.nonzero(mask)
            for _ in range(n_spots):
                i = rng.integers(0, len(xs))
                cv2.circle(img, (int(xs[i]), int(ys[i])), int(rng.integers(2, 6)), (20, 35, 55), -1)
            img = cv2.GaussianBlur(img, (3, 3), 0)
            height = 0.035 if WORLD.fruit_type == "banana" else 0.065
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            bump = height * np.sqrt(np.clip(dist / max(dist.max(), 1), 0, 1))
            depth = (self.FLOOR_M - bump).astype(np.float32)
        depth += rng.normal(0, 0.0008, depth.shape).astype(np.float32)
        time.sleep(self.delay)
        return Frame(color=img, depth=depth, fx=fx, fy=fy, t=time.time())

    def close(self):
        pass


def make_camera(cfg: dict):
    kind = cfg["kind"]
    if kind == "realsense":
        try:
            return RealSenseCamera(cfg)
        except Exception as e:  # pragma: no cover - hardware only
            log.warning("RealSense unavailable (%s); falling back to OpenCV webcam", e)
            kind = "opencv"
    if kind == "opencv":
        return OpenCVCamera(cfg)
    return MockCamera(cfg)
