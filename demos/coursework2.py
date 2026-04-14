"""
coursework2.py – Co-evolutionary Predator–Prey Simulation with Zones
Stage 1: Environment and zone implementation.

World layout (800 × 600):
    ResourceZone  – large green circle at the centre of the map.
                    Prey gain +1 food reward every timestep they spend inside.
    SafeZone      – blue circle in the bottom-left corner.
                    Predators are blocked from eating prey that are inside.

Agents:
    Prey      – 4 sensors (2 predator-proximity + distance to each zone),
                neural-network controlled, fitness = food_collected – 10×eaten.
    Predator  – 2 prey-proximity sensors, neural-network controlled,
                fitness = number of prey successfully eaten outside SafeZone.

Both populations evolve independently via a Genetic Algorithm (roulette selection).
"""

import os
import csv
import time

import numpy as np

from core.agent.ffn_agent import EvolvableFFNAgent
from core.evolve.evolver import Evolver
from core.sensor.base import Sensor
from core.sensor.implementation import proximity_sensor
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm
from core.utils import GA_SELECTION_TYPE, WORLD_DISPLAY_PARAMETERS
from core.evolve.population import Population
from core.world.world_object import WorldObject

IS_DEMO  = True
DEMO_NAME = "Coursework 2"
CLASS_NAME = "CoevSimulation"

# ── World dimensions ─────────────────────────────────────────────────────────
WORLD_W: float = WORLD_DISPLAY_PARAMETERS.width   # 800.0
WORLD_H: float = WORLD_DISPLAY_PARAMETERS.height  # 600.0

# ── Zone geometry ────────────────────────────────────────────────────────────
"""
    Diagonal Opposite Corners layout:
    SafeZone at bottom-left corner (100, 100) and ResourceZone at top-right corner (700, 500).
    This creates a strong safety versus reward trade-off, requiring prey agents
    to travel diagonally across open space to obtain food while increasing exposure to predators.
"""
SAFE_ZONE_CENTER  = np.array([100.0, 100.0], np.float32)
SAFE_ZONE_RADIUS  = 60.0

RESOURCE_ZONE_CENTER = np.array([700.0, 500.0], np.float32)
RESOURCE_ZONE_RADIUS = 100.0

# ── Sensor parameters ────────────────────────────────────────────────────────
MAX_SENSOR_RANGE  = 300.0
ZONE_SENSOR_RANGE = float(max(WORLD_W, WORLD_H))   # full-world scale


# ════════════════════════════════════════════════════════════════════════════
# Zone classes
# ════════════════════════════════════════════════════════════════════════════

class Zone(WorldObject):
    """
    A static circular region drawn as a semi-transparent coloured disk.
    Zones have no physical effect on agents; interaction is handled by
    explicit containment checks in agent code.
    """

    def __init__(self, center: np.ndarray, radius: float, colour: list):
        # Pass the fixed centre so _reset_random["location"] = False
        super().__init__(location=center.copy(), radius=radius, solid=False)
        self._zone_center: np.ndarray = center.copy()
        self.colour: list             = list(colour)

    # Override initialise so the zone never gets a random position
    def initialise(self) -> None:
        self._start_location    = self._zone_center.copy()
        self._start_orientation = 0.0
        self.location           = self._zone_center.copy()
        self.orientation        = 0.0
        self.dead               = False
        self.initialised        = True

    def contains(self, point: np.ndarray) -> bool:
        """Return True if *point* lies inside this zone."""
        return float(np.linalg.norm(point - self._zone_center)) <= self.radius

    def draw(self) -> None:
        """Draw a translucent filled disk with a solid border."""
        from OpenGL.GL import (glEnable, glDisable, glBlendFunc, glColor4f, glLineWidth,
                               GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        from OpenGL.GLU import gluNewQuadric, gluQuadricDrawStyle, gluDisk, gluDeleteQuadric, GLU_FILL, GLU_LINE
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # Filled translucent interior
        glColor4f(self.colour[0], self.colour[1], self.colour[2], 0.25)
        disk = gluNewQuadric()
        gluQuadricDrawStyle(disk, GLU_FILL)
        gluDisk(disk, 0, self.radius, 48, 1)
        gluDeleteQuadric(disk)

        # Solid border ring
        glColor4f(self.colour[0], self.colour[1], self.colour[2], 0.85)
        glLineWidth(2.0)
        border = gluNewQuadric()
        gluQuadricDrawStyle(border, GLU_LINE)
        gluDisk(border, self.radius - 2.0, self.radius, 48, 1)
        gluDeleteQuadric(border)

        glDisable(GL_BLEND)


class ResourceZone(Zone):
    """Central green foraging area – prey gain +1 food per timestep inside."""
    def __init__(self):
        super().__init__(
            RESOURCE_ZONE_CENTER,
            RESOURCE_ZONE_RADIUS,
            [0.2, 0.85, 0.2, 1.0]   # green
        )


class SafeZone(Zone):
    """Blue refuge – predators cannot eat prey inside."""
    def __init__(self, center: np.ndarray = None):
        super().__init__(
            center if center is not None else SAFE_ZONE_CENTER,
            SAFE_ZONE_RADIUS,
            [1.0, 0.2, 0.2, 1.0]    # red
        )


# ════════════════════════════════════════════════════════════════════════════
# Custom zone-distance sensor
# ════════════════════════════════════════════════════════════════════════════

class ZoneDistanceSensor(Sensor):
    """
    Outputs a scalar in [0, 1] representing the normalised Euclidean distance
    from the owning agent to a fixed world point (a zone centre):

        0.0  →  agent is exactly at the zone centre
        1.0  →  agent is at max_range or beyond

    Unlike beam sensors this sensor does not need to sweep other agents;
    it computes its value directly from the owner's position each timestep.
    """

    def __init__(self, zone_center: np.ndarray, max_range: float):
        # Give the sensor a concrete location so WorldObject.__init__ doesn't
        # randomise it; we never actually use the inherited location field.
        super().__init__(
            location=np.array([0.0, 0.0], np.float32),
            orientation=0.0
        )
        self._zone_center: np.ndarray = zone_center.copy()
        self._max_range: float        = max_range
        self._value: float            = 1.0   # default: "far away"

    # Minimal initialise – just set required attributes
    def initialise(self) -> None:
        self._start_location    = np.array([0.0, 0.0], np.float32)
        self._start_orientation = 0.0
        self.location           = self._start_location.copy()
        self.orientation        = 0.0
        self.dead               = False
        self.initialised        = True

    def update(self) -> None:
        """Recompute normalised distance from owner to zone centre."""
        if self.owner is not None:
            dist = float(np.linalg.norm(self.owner.location - self._zone_center))
            self._value = min(1.0, dist / self._max_range)

    def interact(self, other) -> None:
        pass   # No beam sweep needed

    def display(self) -> None:
        pass   # No visual representation

    def output(self) -> float:
        return self._value


# ════════════════════════════════════════════════════════════════════════════
# Agent classes
# ════════════════════════════════════════════════════════════════════════════

class Prey(EvolvableFFNAgent, Evolver):
    """
    Neural-network prey agent.

    Sensor inputs (4):
        pred_left     – proximity to nearest predator in left  arc
        pred_right    – proximity to nearest predator in right arc
        resource_dist – normalised distance to ResourceZone centre
        safe_dist     – normalised distance to SafeZone centre

    Fitness = food_collected – 10 × times_eaten  (clamped to ≥ 0)
    """

    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        self.times_eaten:    int = 0
        self.food_collected: int = 0

        # Predator-proximity sensors (threat detection)
        self.add_sensor(
            "pred_left",
            proximity_sensor(Predator, np.pi / 4, MAX_SENSOR_RANGE,  np.pi / 8, True)
        )
        self.add_sensor(
            "pred_right",
            proximity_sensor(Predator, np.pi / 4, MAX_SENSOR_RANGE, -np.pi / 8, True)
        )

        # Zone-distance sensors (navigation cues)
        self.add_sensor("resource_dist", ZoneDistanceSensor(RESOURCE_ZONE_CENTER, ZONE_SENSOR_RANGE))
        self.add_sensor("safe_dist",     ZoneDistanceSensor(SAFE_ZONE_CENTER,     ZONE_SENSOR_RANGE))

        self._interaction_range = MAX_SENSOR_RANGE
        # 4 hidden nodes; inputs (4) and outputs (2) are inferred automatically
        self.add_brain(4)

        self.solid      = False
        self.radius     = 10.0
        self._min_speed = 0.0
        self._max_speed = 100.0

    def control(self):
        super().control()
        # Map tanh outputs [-1,1] → [0,1] for wheel speeds
        for k in self.controls:
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)

    def update(self):
        super().update()
        # Reward foraging: +1 each timestep spent inside ResourceZone
        for obj in self.world._objects:
            if isinstance(obj, ResourceZone) and obj.contains(self.location):
                self.food_collected += 1
                break

    def eaten(self):
        """Called by a Predator that successfully catches this prey."""
        self.times_eaten += 1
        self.location = self.world.random_location()
        self.trail.clear()

    def get_fitness(self) -> float:
        return max(0.0, float(self.food_collected) - 10.0 * self.times_eaten)

    def reset(self):
        self.times_eaten    = 0
        self.food_collected = 0


class Predator(EvolvableFFNAgent, Evolver):
    """
    Neural-network predator agent.

    Sensor inputs (2):
        prey_left  – proximity to nearest prey in left  arc
        prey_right – proximity to nearest prey in right arc

    Predators are physically blocked from entering any SafeZone: after each
    movement step, the agent is pushed back to the zone boundary if it has
    crossed into one.  Because predators cannot enter a SafeZone, prey inside
    are automatically protected – no extra check is needed in on_collision.

    Fitness = number of prey successfully eaten.
    """

    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        self.prey_eaten: int = 0

        self.add_sensor(
            "prey_left",
            proximity_sensor(Prey, np.pi / 4, MAX_SENSOR_RANGE,  np.pi / 8, True)
        )
        self.add_sensor(
            "prey_right",
            proximity_sensor(Prey, np.pi / 4, MAX_SENSOR_RANGE, -np.pi / 8, True)
        )

        self._interaction_range = MAX_SENSOR_RANGE
        self.add_brain(4)

        self.solid      = False
        self._min_speed = 0.0
        self._max_speed = 110.0   # slightly faster than prey
        self.radius     = 15.0

    def control(self):
        super().control()
        for k in self.controls:
            self.controls[k] = 0.5 * (self.controls[k] + 1.0)

    def update(self):
        super().update()
        # Physical exclusion: push predator out of any SafeZone it has entered
        for obj in self.world._objects:
            if not isinstance(obj, SafeZone):
                continue
            diff = self.location - obj._zone_center
            dist = float(np.linalg.norm(diff))
            min_dist = obj.radius + self.radius
            if dist < min_dist:
                # Move predator to the zone boundary (plus its own radius)
                if dist > 0:
                    direction = diff / dist
                else:
                    direction = np.array([1.0, 0.0], np.float32)
                self.location = obj._zone_center + direction * min_dist
                # Kill velocity component pointing into the zone
                self.velocity -= np.dot(self.velocity, direction) * direction

    def on_collision(self, other):
        if isinstance(other, Prey):
            # Predator is already blocked from SafeZones, so any touching prey is fair game.
            self.prey_eaten += 1
            other.eaten()
        super().on_collision(other)

    def get_fitness(self) -> float:
        return float(self.prey_eaten)

    def reset(self):
        self.prey_eaten = 0


# ════════════════════════════════════════════════════════════════════════════
# Simulation
# ════════════════════════════════════════════════════════════════════════════

class CoevSimulation(Simulation):
    """
    Co-evolutionary predator–prey simulation with resource and safe zones.

    Stage 1: establishes the 2-D environment, zone objects, and baseline
    agent interactions so that further stages (evolved behaviours, analysis)
    can be built on top.
    """

    def __init__(self):
        super().__init__("Coursework2")

        self.runs        = 1
        self.generations = 50
        self.assessments = 2
        self.timesteps   = 1000

        # Persistent zone objects – re-injected into the world each assessment
        self._zones = [
            SafeZone(SAFE_ZONE_CENTER),   # bottom-left corner (100, 100)
            ResourceZone(),                # top-right corner (700, 500)
        ]

        # Fitness history for logging / CSV export
        self._prey_history:     list[float] = []
        self._predator_history: list[float] = []

        # Co-evolving populations (30 individuals, 10 per assessment team)
        pop_size = 30
        ga_prey = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)
        ga_pred = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)

        self.add("prey",     Population(pop_size, Prey,     ga_prey, team_size=15))
        self.add("predator", Population(pop_size, Predator, ga_pred, team_size=3))

    # ── Override begin_assessment to inject zones before world.initialise() ─
    def begin_assessment(self) -> None:
        self.log_begin_assessment()
        time.sleep(self.sleep_betwen_logs)

        self._timestep = 0

        # Let each population place its current team into the world
        for obj in self.contents.values():
            obj.begin_assessment()
            obj.add_to_world()

        # Add the static zone objects so they appear in world._objects
        # (and are included in the world.initialise() call below)
        for zone in self._zones:
            self.world.add_object(zone)

        # Initialise all objects and agents that are now in the world
        self.world.initialise()

    # ── Per-generation logging ───────────────────────────────────────────────
    def log_end_generation(self) -> None:
        prey_avgs = self.contents["prey"].average_member_fitness()
        prey_avg  = sum(prey_avgs) / len(prey_avgs)
        self._prey_history.append(prey_avg)

        pred_avgs = self.contents["predator"].average_member_fitness()
        pred_avg  = sum(pred_avgs) / len(pred_avgs)
        self._predator_history.append(pred_avg)

        self.log.info(
            f"Gen {self._generation + 1:>3}/{self.generations} | "
            f"Prey avg fitness: {prey_avg:8.3f} | "
            f"Predator avg fitness: {pred_avg:8.3f}"
        )

        # Save CSV on the final generation
        if len(self._prey_history) == self.generations:
            self._save_results()

    def _save_results(self) -> None:
        desktop  = os.path.join(os.path.expanduser("~"), "Desktop")
        filename = os.path.join(
            desktop, f"coursework2_{self.generations}gens.csv"
        )
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Generation", "Prey_Fitness", "Predator_Fitness"])
            for i in range(self.generations):
                writer.writerow([i + 1,
                                 self._prey_history[i],
                                 self._predator_history[i]])
        self.log.info(f"Results saved → {filename}")
