"""
coursework2_medium.py – Co-evolutionary Predator–Prey Simulation with Zones (Medium Distance)
Medium diagonal distance layout variant.

World layout (800 × 600):
    ResourceZone  – green circle at (600, 400)
                    Prey gain +1 food reward every timestep they spend inside.
    SafeZone      – red circle at (200, 200)
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
from core.utils import GA_SELECTION_TYPE, GA_FITNESS_FIX, WORLD_DISPLAY_PARAMETERS2
from core.evolve.population import Population
from core.world.world_object import WorldObject

IS_DEMO  = True
DEMO_NAME = "Coursework 2 - Sensor Added to Predator Two-Stage Evolution"
CLASS_NAME = "CoevSimulationTwoStage"

# ── World dimensions ─────────────────────────────────────────────────────────
WORLD_W: float = WORLD_DISPLAY_PARAMETERS2.width   # 800.0
WORLD_H: float = WORLD_DISPLAY_PARAMETERS2.height  # 600.0

# ── Zone geometry ────────────────────────────────────────────────────────────
"""
    Medium Diagonal Distance layout:
    SafeZone at (200, 200) and ResourceZone at (600, 400).
    This creates a moderate safety versus reward trade-off with diagonal zones
    at medium distance from each other in a square world.
"""
SAFE_ZONE_CENTER  = np.array([200.0, 200.0], np.float32)
SAFE_ZONE_RADIUS  = 80.0

RESOURCE_ZONE_CENTER = np.array([600.0, 400.0], np.float32)
RESOURCE_ZONE_RADIUS = 150.0

# ── Two-stage evolution parameters ───────────────────────────────────────────
STAGE1_GENS = 20   # prey-only phase; predators introduced from gen STAGE1_GENS+1

# ── Energy parameters ────────────────────────────────────────────────────────
PREY_INIT_ENERGY: float = 100.0
PREY_MAX_ENERGY:  float = 150.0   # cap prevents unbounded accumulation in resource zone
PRED_INIT_ENERGY: float = 100.0   # keep at 100: drain=0.06×2000=120 > 100, so must hunt
PRED_MAX_ENERGY:  float = 250.0   # larger body (r=15 vs 10) stores more energy per capture

# ── Sensor parameters ─────────────────────────────────────────────────────────
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

    Fitness = survival_time + food_collected×3 + resource_visits×20 + final_energy
    """

    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        self.times_eaten:      int   = 0
        self.food_collected:   int   = 0
        self.energy:           float = PREY_INIT_ENERGY
        self.survival_time:    int   = 0
        self.resource_visits:  int   = 0
        self._was_in_resource: bool  = False
        self._turning_total:   float = 0.0   # Σ |left−right| per tick

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
        self._turning_total += abs(self.controls["left"] - self.controls["right"])

    def update(self):
        super().update()

        # Passive energy drain
        self.energy -= 0.06

        # Check if inside resource zone
        inside_resource = False
        for obj in self.world._objects:
            if isinstance(obj, ResourceZone) and obj.contains(self.location):
                inside_resource = True
                # Food reward: +1 per tick
                self.food_collected += 1
                # Energy gain: +0.35 per tick (capped at PREY_MAX_ENERGY)
                self.energy = min(self.energy + 0.35, PREY_MAX_ENERGY)
                break

        # Track resource visits (edge detection: entering from outside)
        if inside_resource and not self._was_in_resource:
            self.resource_visits += 1
        self._was_in_resource = inside_resource

        # Track survival time (increments each tick while alive)
        self.survival_time += 1

        # Death if energy depleted
        if self.energy <= 0.0:
            self.dead = True

    def eaten(self):
        self.times_eaten += 1
        self.location = self.world.random_location()
        self.trail.clear()

    def get_fitness(self) -> float:
        avg_spin = self._turning_total / max(self.survival_time, 1)
        energy_death_penalty = 300.0 if self.dead else 0.0
        return float(
            self.survival_time
            + self.food_collected * 1.0        # was 0.8; foraging is the primary signal
            + self.resource_visits * 5.0       # was 0.3; must be meaningful for zone navigation
            + max(self.energy, 0.0) * 0.5      # was 1.0; cap at 0, halve to reduce double-dip with food
            - self.times_eaten * 150.0         # evasion pressure: being caught is costly
            - energy_death_penalty             # explicit penalty on top of shorter survival_time
            - avg_spin * 150.0
        )

    def reset(self):
        self.times_eaten      = 0
        self.food_collected   = 0
        self.energy           = PREY_INIT_ENERGY
        self.survival_time    = 0
        self.resource_visits  = 0
        self._was_in_resource = False
        self._turning_total   = 0.0


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

    Fitness = prey_captured×20 + final_energy
    """

    def __init__(self):
        EvolvableFFNAgent.__init__(self)
        Evolver.__init__(self)

        self.prey_eaten:       int   = 0
        self.energy:           float = PRED_INIT_ENERGY
        self.survival_time:    int   = 0
        self._turning_total:   float = 0.0   # Σ |left−right| per tick
        self.dead:             bool  = False 

        self.add_sensor(
            "prey_left",
            proximity_sensor(Prey, np.pi / 4, MAX_SENSOR_RANGE,  np.pi / 8, True)
        )
        self.add_sensor(
            "prey_right",
            proximity_sensor(Prey, np.pi / 4, MAX_SENSOR_RANGE, -np.pi / 8, True)
        )
        # Zone-distance sensors (navigation cues)
        self.add_sensor("resource_dist", ZoneDistanceSensor(RESOURCE_ZONE_CENTER, ZONE_SENSOR_RANGE))
        self.add_sensor("safe_dist",     ZoneDistanceSensor(SAFE_ZONE_CENTER,     ZONE_SENSOR_RANGE))

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
        self._turning_total += abs(self.controls["left"] - self.controls["right"])

    def update(self):
        super().update()

        self.survival_time += 1

        # Passive energy drain
        self.energy -= 0.09

        # Death if energy depleted
        if self.energy <= 0.0:
            self.dead = True

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
            # Energy gain from successful capture (capped at PRED_MAX_ENERGY)
            self.energy = min(self.energy + 20.0, PRED_MAX_ENERGY)
            other.eaten()
        super().on_collision(other)

    def get_fitness(self) -> float:
        avg_spin = self._turning_total / max(self.survival_time, 1)
        energy_death_penalty = 100.0 if self.dead else 0.0
        return float(
            self.prey_eaten * 20
            + self.energy
            - avg_spin * 30.0
            - energy_death_penalty
        )

    def reset(self):
        self.prey_eaten     = 0
        self.energy         = PRED_INIT_ENERGY
        self.survival_time  = 0
        self._turning_total = 0.0


# ════════════════════════════════════════════════════════════════════════════
# Simulation
# ════════════════════════════════════════════════════════════════════════════

class CoevSimulationTwoStage(Simulation):
    """
    Co-evolutionary predator–prey simulation with resource and safe zones.
    Medium distance diagonal layout variant.
    """

    def __init__(self):
        super().__init__("Coursework2_TwoStage")
        self.world._display_params = WORLD_DISPLAY_PARAMETERS2  # match 800×600 viewport

        self.display_on  = False  # Disable rendering for faster evolution
        self.runs        = 1
        self.generations = 200
        self.stage1_gens = STAGE1_GENS  # prey-only phase length
        self.assessments = 2
        self.timesteps   = 2000

        # Persistent zone objects – re-injected into the world each assessment
        self._zones = [
            SafeZone(SAFE_ZONE_CENTER),   # at (200, 200)
            ResourceZone(),                # at (600, 400)
        ]

        # Fitness history for logging / CSV export
        self._prey_history:     list[float] = []
        self._predator_history: list[float] = []

        # Per-generation metrics tracking
        self._avg_prey_energy:     list[float] = []
        self._prey_starved:        list[float] = []  # Prey that died from energy depletion
        self._avg_food_collected:  list[float] = []
        self._avg_times_eaten:     list[float] = []
        self._avg_predator_energy: list[float] = []
        self._pred_starved:        list[float] = []  # Predators that died from energy depletion

        # Temporary storage for metrics (captured before reset, accumulated across assessments)
        self._temp_metrics = {}
        self._assessment_metrics = []  # Store metrics from each assessment

        # Co-evolving populations (30 individuals, 15 prey and 5 predators per assessment)
        pop_size = 30
        ga_prey = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE,
                                   fitness_fix=GA_FITNESS_FIX.FIX)
        ga_pred = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE,
                                   fitness_fix=GA_FITNESS_FIX.FIX)

        self.add("prey",     Population(pop_size, Prey,     ga_prey, team_size=15))
        self.add("predator", Population(pop_size, Predator, ga_pred, team_size=5))

    def _in_stage1(self) -> bool:
        return self._generation < self.stage1_gens

    # ── Override begin_assessment to inject zones before world.initialise() ─
    def begin_assessment(self) -> None:
        self.log_begin_assessment()
        time.sleep(self.sleep_betwen_logs)

        self._timestep = 0

        # Log the one-time transition into stage 2
        if self._generation == self.stage1_gens and self._assessment == 0:
            self.log.info("=" * 64)
            self.log.info(
                f"  STAGE 2 START (Gen {self._generation + 1}): "
                "Predators now active!"
            )
            self.log.info("=" * 64)

        # Prepare teams for all populations (does NOT add to world yet)
        for obj in self.contents.values():
            obj.begin_assessment()

        # Stage 1 – prey only; Stage 2 – both prey and predators
        self.contents["prey"].add_to_world()
        if not self._in_stage1():
            self.contents["predator"].add_to_world()

        # Add the static zone objects
        for zone in self._zones:
            self.world.add_object(zone)

        # Initialise all objects and agents that are now in the world
        self.world.initialise()

    # ── Override end_assessment to capture metrics before reset ──────────────
    def end_assessment(self) -> None:
        # Capture metrics from active team BEFORE they are reset
        prey_pop = self.contents["prey"]
        all_prey = prey_pop.team if prey_pop.team_size != -1 else prey_pop.members
        all_prey = [agent for agent in all_prey if isinstance(agent, Prey)]

        pred_pop = self.contents["predator"]
        all_pred = pred_pop.team if pred_pop.team_size != -1 else pred_pop.members
        all_pred = [agent for agent in all_pred if isinstance(agent, Predator)]

        # Store metrics from this assessment
        # Count deaths: agents marked as dead (energy <= 0)
        assessment_metrics = {
            'prey_starved': sum(1 for prey in all_prey if prey.dead),
            'avg_prey_energy': sum(prey.energy for prey in all_prey) / len(all_prey) if all_prey else 0.0,
            'min_prey_energy': min((prey.energy for prey in all_prey), default=0.0),
            'avg_food': sum(prey.food_collected for prey in all_prey) / len(all_prey) if all_prey else 0.0,
            'avg_times_eaten': sum(prey.times_eaten for prey in all_prey) / len(all_prey) if all_prey else 0.0,
            'avg_pred_energy': sum(pred.energy for pred in all_pred) / len(all_pred) if all_pred else 0.0,
            'pred_starved': sum(1 for pred in all_pred if pred.dead),
        }

        # Accumulate metrics across assessments
        self._assessment_metrics.append(assessment_metrics)

        # Call parent's end_assessment (which will reset agents)
        super().end_assessment()

    # ── Per-generation logging ───────────────────────────────────────────────
    def log_end_generation(self) -> None:
        # Fitness tracking
        prey_avgs = self.contents["prey"].average_member_fitness()
        prey_avg  = sum(prey_avgs) / len(prey_avgs)
        self._prey_history.append(prey_avg)

        pred_avgs = self.contents["predator"].average_member_fitness()
        pred_avg  = sum(pred_avgs) / len(pred_avgs)
        self._predator_history.append(pred_avg)

        # Average metrics across all assessments in this generation
        num_assessments = len(self._assessment_metrics)
        if num_assessments > 0:
            prey_starved = sum(m['prey_starved'] for m in self._assessment_metrics) / num_assessments
            avg_prey_energy = sum(m['avg_prey_energy'] for m in self._assessment_metrics) / num_assessments
            min_prey_energy = min(m['min_prey_energy'] for m in self._assessment_metrics)
            avg_food = sum(m['avg_food'] for m in self._assessment_metrics) / num_assessments
            avg_times_eaten = sum(m['avg_times_eaten'] for m in self._assessment_metrics) / num_assessments
            avg_pred_energy = sum(m['avg_pred_energy'] for m in self._assessment_metrics) / num_assessments
            pred_starved = sum(m['pred_starved'] for m in self._assessment_metrics) / num_assessments
        else:
            prey_starved = avg_prey_energy = min_prey_energy = 0.0
            avg_food = avg_times_eaten = avg_pred_energy = pred_starved = 0.0

        # Clear assessment metrics for next generation
        self._assessment_metrics.clear()

        self._prey_starved.append(prey_starved)
        self._avg_prey_energy.append(avg_prey_energy)
        self._avg_food_collected.append(avg_food)
        self._avg_times_eaten.append(avg_times_eaten)
        self._avg_predator_energy.append(avg_pred_energy)
        self._pred_starved.append(pred_starved)

        stage_tag = "S1" if self._in_stage1() else "S2"
        self.log.info(
            f"[{stage_tag}] Gen {self._generation + 1:>3}/{self.generations} | "
            f"Prey fit: {prey_avg:7.2f} | Pred fit: {pred_avg:7.2f} | "
            f"Prey E: {avg_prey_energy:5.1f} (starved:{prey_starved:.1f}, eaten:{avg_times_eaten:.1f}) | "
            f"Pred E: {avg_pred_energy:5.1f} (starved:{pred_starved:.1f})"
        )

        # Save CSV on the final generation
        if len(self._prey_history) == self.generations:
            self._save_results()

    def _save_results(self) -> None:
        # Save to results/ directory in project root
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        filename = os.path.join(
            results_dir, f"SensorAddtoPredTwoStageEvolution_{self.generations}gens.csv"
        )
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Generation",
                "Stage",
                "Prey_Fitness",
                "Predator_Fitness",
                "Avg_Prey_Energy",
                "Prey_Starved",
                "Prey_Eaten",
                "Avg_Food_Collected",
                "Avg_Predator_Energy",
                "Pred_Starved"
            ])
            for i in range(self.generations):
                writer.writerow([
                    i + 1,
                    1 if i < self.stage1_gens else 2,
                    self._prey_history[i],
                    self._predator_history[i],
                    self._avg_prey_energy[i],
                    self._prey_starved[i],
                    self._avg_times_eaten[i],
                    self._avg_food_collected[i],
                    self._avg_predator_energy[i],
                    self._pred_starved[i]
                ])
        self.log.info(f"Results saved → {filename}")
        self._plot_results(results_dir)

    def _plot_results(self, results_dir: str) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            self.log.warning("matplotlib not installed – skipping plot generation")
            return

        gens    = list(range(1, self.generations + 1))
        s1      = self.stage1_gens
        gens_s2 = gens[s1:]
        BOUNDARY_X = s1 + 0.5

        def smooth(data, window=5):
            if len(data) < window:
                return list(data)
            smoothed = np.convolve(data, np.ones(window) / window, mode="same").tolist()
            # Fix edge artifacts: use original values for last few points
            smoothed[-3:] = data[-3:]
            return smoothed

        prey_sm = smooth(self._prey_history)
        pred_sm = smooth(self._predator_history)

        fig, (ax_prey, ax_pred) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={"hspace": 0.08}
        )
        fig.suptitle("Prey vs Predator Fitness – Pred with Zone Sensors Added",
                     fontsize=13, fontweight="bold", y=0.98)

        # ── shared stage shading helper ──────────────────────────────────────
        def add_stage_bands(ax):
            ax.axvspan(1, BOUNDARY_X, alpha=0.07, color="steelblue", zorder=0)
            ax.axvspan(BOUNDARY_X, self.generations + 0.5,
                       alpha=0.07, color="tomato", zorder=0)
            ax.axvline(BOUNDARY_X, color="dimgray", linestyle="--",
                       linewidth=1.4, alpha=0.8, zorder=1)
            ax.grid(True, alpha=0.25)
            ax.set_xlim(1, self.generations)

        # ── top panel: prey ──────────────────────────────────────────────────
        add_stage_bands(ax_prey)
        ax_prey.plot(gens, self._prey_history,
                     color="steelblue", alpha=0.25, linewidth=0.9)
        ax_prey.plot(gens, prey_sm,
                     color="steelblue", linewidth=2.2, label="Prey fitness")
        ax_prey.set_ylabel("Prey Fitness", fontsize=11)
        ax_prey.legend(fontsize=10, loc="upper left")
        # Stage label annotations (top panel only)
        ax_prey.text(s1 / (2 * self.generations), 0.96,
                     f"Stage 1  (prey only, {s1} gens)",
                     transform=ax_prey.transAxes, ha="center", va="top",
                     fontsize=9, color="steelblue", alpha=0.85)
        ax_prey.text((s1 + (self.generations - s1) * 0.5) / self.generations, 0.96,
                     f"Stage 2  (co-evolution, {self.generations - s1} gens)",
                     transform=ax_prey.transAxes, ha="center", va="top",
                     fontsize=9, color="dimgray", alpha=0.85)

        # ── bottom panel: predator ───────────────────────────────────────────
        add_stage_bands(ax_pred)
        ax_pred.plot(gens_s2,
                     [self._predator_history[i] for i in range(s1, self.generations)],
                     color="tomato", alpha=0.25, linewidth=0.9)
        ax_pred.plot(gens_s2, pred_sm[s1:],
                     color="tomato", linewidth=2.2, label="Predator fitness")
        ax_pred.set_ylabel("Predator Fitness", fontsize=11)
        ax_pred.set_xlabel("Generation", fontsize=11)
        ax_pred.legend(fontsize=10, loc="upper left")

        plot_path = os.path.join(
            results_dir, f"SensorAddtoPredTwoStageEvolution_{self.generations}gens.png"
        )
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        self.log.info(f"Plot saved  → {plot_path}")
