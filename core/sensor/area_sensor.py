from core.sensor.base import Sensor
from core.world.world_object import WorldObject

class AreaSensor(Sensor):
    def interact(self, other: WorldObject) -> None:
        if not self.match_function or not self.evaluate_function:
            return
        point, _ = other.nearest_point(self.location)
        if self.is_inside(point):
            self.evaluate_function(other, point)
