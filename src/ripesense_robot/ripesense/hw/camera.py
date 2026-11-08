"""RealSense RGB + aligned depth capture, with webcam and mock fallbacks.

Run on the Pi to test:  python -m ripesense.hw.camera --test
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class Frame:
    bgr: np.ndarray                 # HxWx3 uint8
    depth_mm: np.ndarray | None     # HxW uint16 (mm), aligned to color; None for webcam
    fx: float
    fy: float
    source: str
    ir: np.ndarray | None = None          # 850 nm IR image (projector off), RealSense / mock only
    ir_depth_mm: np.ndarray | None = None  # depth in the IR camera's own frame (pixel-aligned to ir)
    ir_fx: float = 380.0


class RealSenseCamera:
    """Starts the pipeline per capture so the camera isn't drawing power or
    heating the chamber between scans. Start-up costs ~1-2 s on a Pi 5."""

    def __init__(self, cfg: dict):
        import pyrealsense2 as rs  # built from source on the Pi (deploy/build_librealsense.sh)
        self.rs = rs
        self.cfg = cfg
        self.ir_light = None       # set by the controller: callable(bool) for the 850 nm LEDs

    def capture(self) -> Frame:
        rs, c = self.rs, self.cfg
        pipe = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(rs.stream.color, c["width"], c["height"], rs.format.bgr8, 15)
        conf.enable_stream(rs.stream.depth, c.get("depth_width", 424), c.get("depth_height", 240), rs.format.z16, 15)
        if c.get("ir_bruise", True):
            conf.enable_stream(rs.stream.infrared, 1, c.get("depth_width", 424), c.get("depth_height", 240), rs.format.y8, 15)
        profile = pipe.start(conf)
        try:
            scale_m = profile.get_device().first_depth_sensor().get_depth_scale()
            align = rs.align(rs.stream.color)
            for _ in range(c.get("warmup_frames", 15)):   # auto-exposure settle
                pipe.wait_for_frames()
            # Median of 5 depth frames cuts speckle noise on the fruit surface.
            depths, color = [], None
            for _ in range(5):
                fs = align.process(pipe.wait_for_frames())
                color = np.asanyarray(fs.get_color_frame().get_data()).copy()
                depths.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32))
            d = np.median(np.stack(depths), axis=0)
            d = d * scale_m * 1000.0  # → mm
            d[d <= 0] = 0
            intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            ir, ir_d, ir_fx = self._ir(pipe, profile, scale_m) if c.get("ir_bruise", True) else (None, None, 380.0)
            return Frame(color, d.astype(np.uint16), intr.fx, intr.fy, "realsense", ir, ir_d, ir_fx)
        finally:
            pipe.stop()

    def _ir(self, pipe, profile, scale_m):
        """Left IR imager with the dot projector OFF (dots would look like spots)
        and the 850 nm LEDs ON (ir_light callback). Depth is native to this
        imager, so the unaligned depth frame is pixel-aligned with the IR."""
        rs = self.rs
        try:
            dev = profile.get_device().first_depth_sensor()
            if dev.supports(rs.option.emitter_enabled):
                dev.set_option(rs.option.emitter_enabled, 0)
            if self.ir_light:
                self.ir_light(True)
            for _ in range(6):                      # emitter-off + LED settle
                pipe.wait_for_frames()
            irs, ds = [], []
            for _ in range(3):
                fs = pipe.wait_for_frames()
                irs.append(np.asanyarray(fs.get_infrared_frame(1).get_data()).astype(np.float32))
                ds.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32))
            ir = np.median(np.stack(irs), axis=0).astype(np.uint8)
            dd = np.median(np.stack(ds), axis=0) * scale_m * 1000.0
            fx = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile().get_intrinsics().fx
            return ir, dd.astype(np.uint16), fx
        except Exception as e:  # noqa: BLE001
            log.warning("IR capture failed: %s", e)
            return None, None, 380.0
        finally:
            if self.ir_light:
                self.ir_light(False)


class _WebcamReader:
    """One background reader per camera index, shared by every WebcamCamera
    (creating several WebcamCamera objects must not open the device twice)."""

    _shared: dict = {}
    _guard = __import__("threading").Lock()

    @classmethod
    def get(cls, index: int, width: int, height: int):
        with cls._guard:
            r = cls._shared.get(index)
            if r is None:
                r = cls._shared[index] = cls(index, width, height)
            return r

    def __init__(self, index, width, height):
        import threading
        self.index, self.width, self.height = index, width, height
        self._frame = None
        self._lock = threading.Lock()
        self._cap = None
        threading.Thread(target=self._run, daemon=True, name=f"webcam{index}").start()

    def _open(self):
        import cv2
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2) if hasattr(cv2, "CAP_V4L2") else cv2.VideoCapture(self.index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return cap if cap.isOpened() else None

    def _run(self):
        while True:
            if self._cap is None:
                self._cap = self._open()
                if self._cap is None:
                    log.warning("webcam %s not available; retrying", self.index)
                    time.sleep(2)
                    continue
            ok, f = self._cap.read()
            if not ok:
                self._cap.release()
                self._cap = None
                time.sleep(0.5)
                continue
            with self._lock:
                self._frame = f

    def latest(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()


class WebcamCamera:
    """Any UVC webcam (a USB camera, or the RealSense colour stream without
    librealsense). The device stays open in a shared background reader so a
    live preview and scans can use it together; capture() returns the newest
    frame. No depth."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._reader = _WebcamReader.get(cfg.get("webcam_index", 0), cfg.get("width", 640), cfg.get("height", 480))

    def latest(self):
        return self._reader.latest()

    def capture(self) -> Frame:
        t0 = time.time()
        while self.latest() is None:
            if time.time() - t0 > 5:
                raise RuntimeError("webcam returned no frame (is it plugged in? try RIPESENSE_WEBCAM=1)")
            time.sleep(0.05)
        time.sleep(0.15)  # let auto-exposure catch up after any LED change
        # Nominal intrinsics; only used for rough mm estimates without depth.
        return Frame(self.latest(), None, 615.0, 615.0, "webcam")


class MockCamera:
    def __init__(self, cfg: dict):
        self.cfg = cfg

    def capture(self) -> Frame:
        from ..sim import fruit as sim
        time.sleep(0.3)
        bgr, depth = sim.current.render(self.cfg["width"], self.cfg["height"], self.cfg["tray_depth_mm"])
        ir, ir_d = sim.current.render_ir(424, 240, self.cfg["tray_depth_mm"])
        return Frame(bgr, depth, 615.0, 615.0, "mock", ir, ir_d, 380.0)


def make_camera(cfg: dict, mock: bool):
    c = cfg["camera"]
    if mock:
        return MockCamera(c)
    if c.get("backend") == "realsense":
        try:
            return RealSenseCamera(c)
        except Exception as e:  # noqa: BLE001
            log.warning("RealSense unavailable (%s); falling back to webcam", e)
    return WebcamCamera(c)


if __name__ == "__main__":
    import cv2

    from .. import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    cfg = config.load()
    cam = make_camera(cfg, cfg["mock"])
    f = cam.capture()
    out = config.data_dir()
    cv2.imwrite(str(out / "test_rgb.png"), f.bgr)
    if f.depth_mm is not None:
        cv2.imwrite(str(out / "test_depth.png"), f.depth_mm)
        valid = f.depth_mm[f.depth_mm > 0]
        print(f"depth mm: min={valid.min() if valid.size else 0} median={np.median(valid) if valid.size else 0}")
    print(f"{f.source}: rgb {f.bgr.shape} fx={f.fx:.1f} → saved to {out}")
