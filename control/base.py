"""Abstract controller interface. Subclass this for either pure-Python
controllers or C-bridge controllers (see c_bridge.py)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class MotorCommand:
    duty_l: int = 0       # 0..MOTOR_PWM_MAX_COUNT (10000)
    duty_r: int = 0
    servo: int = 3120     # SERVO_MIN_COUNT..SERVO_MAX_COUNT (1920..4320; 3120 = center)
    dir_l: int = 1        # 1 or -1
    dir_r: int = 1


class AbstractController(ABC):
    @abstractmethod
    def init(self) -> None:
        """Called once before the first tick() — zero internal state."""

    @abstractmethod
    def tick(self, sensors: dict, t_ms: float) -> MotorCommand:
        """Given a sensor dict (see sensors.sample_sensors output) and the
        simulated time in ms, return a motor command."""

    def reset(self) -> None:
        self.init()

    def close(self) -> None:
        """Release any external resources (e.g., loaded C library)."""
