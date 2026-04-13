"""
Proprioceptive sensors - sensors that detect the agent's own state.
"""

import numpy as np
from core.sensor.base import Sensor
from core.world.world_object import WorldObject

class OrientationSensor(Sensor):
    """
    Sensor that returns the agent's own orientation.
    Useful for reducing oscillatory turning and improving steering stability.
    """

    def __init__(self):
        super().__init__()

    def interact(self, other: WorldObject) -> None:
        """Proprioceptive sensors don't interact with other objects."""
        pass

    def update(self) -> None:
        # Don't call parent update since we don't need spatial transforms
        pass

    def output(self) -> float:
        """Return normalized orientation in range [-1, 1]."""
        if self.owner is None:
            return 0.0

        # Normalize orientation from [0, 2π] to [-1, 1]
        # 0 rad = 0, π rad = 0, -π rad = 0, ±π/2 = ±1
        orientation = self.owner.orientation
        if orientation > np.pi:
            orientation -= 2 * np.pi

        # Map [-π, π] to [-1, 1]
        return orientation / np.pi

    def display(self) -> None:
        """No visual display for proprioceptive sensors."""
        pass
