"""A simulated fruit and chamber, so every mock device tells the same story.

ripeness: 0.0 = hard green, 0.35 = ripe, 0.65 = overripe, 0.85+ = spoiled.
"""
from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field


@dataclass
class SimWorld:
    fruit_present: bool = True
    fruit_type: str = "banana"
    ripeness: float = 0.4
    door_closed: bool = True
    door_closed_at: float = field(default_factory=time.time)
    door_opened_at: float | None = None
    seed: int = 0
    puck: bool = False          # white calibration puck in the seat
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_fruit(self, fruit_type: str | None = None, ripeness: float | None = None, present: bool = True):
        with self._lock:
            if fruit_type:
                self.fruit_type = fruit_type
            self.ripeness = random.uniform(0.0, 1.0) if ripeness is None else max(0.0, min(1.0, ripeness))
            self.fruit_present = present
            self.door_closed = True
            self.door_closed_at = time.time()
            self.seed = random.randint(0, 10_000)

    def open_door(self):
        with self._lock:
            self.door_closed = False
            self.door_opened_at = time.time()
            self.fruit_present = False

    def close_door(self):
        with self._lock:
            self.door_closed = True
            self.door_closed_at = time.time()


WORLD = SimWorld()
