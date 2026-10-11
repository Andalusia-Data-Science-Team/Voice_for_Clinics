"""Optional stepped Locust load shape for MedVoice.

Enable this shape by loading both files::

    locust -f locustfile.py,step_load_shape.py --host http://localhost:7777
    locust -f locustfile.py,step_load_shape.py --headless \
        --host http://localhost:7777 --csv results --html report.html

Environment variables:

    LOCUST_SHAPE_STEPS=10,20,40,60,80,100  User count at each step
    LOCUST_SHAPE_STEP_SECONDS=120            Hold each step for two minutes
    LOCUST_SHAPE_SPAWN_RATE=10               Users started per second

The normal Locust ``-u``, ``-r``, and ``-t`` options remain unchanged when this
file is not loaded. When this shape is loaded, the shape controls user counts;
use ``LOCUST_SHAPE_*`` to adjust it.
"""

from __future__ import annotations

import os

from locust import LoadTestShape


def _positive_float(name: str, default: str) -> float:
    """Read a positive floating-point environment setting."""
    value = float(os.getenv(name, default))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _steps() -> list[int]:
    """Read and validate the configured user-count steps."""
    values = [int(item.strip()) for item in os.getenv(
        "LOCUST_SHAPE_STEPS", "10,20,40,60,80,100"
    ).split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise ValueError("LOCUST_SHAPE_STEPS must contain positive integers")
    return values


class MedVoiceStepLoadShape(LoadTestShape):
    """Increase the simulated user count at fixed, configurable intervals."""

    step_users = _steps()
    step_duration = _positive_float("LOCUST_SHAPE_STEP_SECONDS", "120")
    spawn_rate = _positive_float("LOCUST_SHAPE_SPAWN_RATE", "10")

    def tick(self) -> tuple[int, float] | None:
        """Return the target users and spawn rate for the current step."""
        step_index = int(self.get_run_time() // self.step_duration)
        if step_index >= len(self.step_users):
            return None
        return self.step_users[step_index], self.spawn_rate
