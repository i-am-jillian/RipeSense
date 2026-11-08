"""Hardware layer. build_devices() returns real drivers, falling back to mocks when a part is missing."""
from __future__ import annotations

from dataclasses import dataclass

from .camera import Frame, make_camera
from .gas import GasMonitor, make_gas
from .io import make_io
from .spectral import make_spectral


@dataclass
class Devices:
    camera: object
    spectral: object
    gas: GasMonitor
    io: object

    def close(self):
        self.gas.stop()
        try:
            self.camera.close()
        finally:
            self.io.close()


def build_devices(cfg) -> Devices:
    sim = cfg["app"]["simulate"]
    io = make_io(cfg["pins"], sim)
    camera = make_camera(cfg["camera"])
    spectral = make_spectral(cfg["spectral"], io)
    gas = GasMonitor(make_gas(cfg["gas"]), cfg["gas"])
    gas.start()
    return Devices(camera=camera, spectral=spectral, gas=gas, io=io)


__all__ = ["Devices", "Frame", "build_devices"]
