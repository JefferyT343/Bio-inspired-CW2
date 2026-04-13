from dataclasses import dataclass
from core.world.world_object import WorldObject
from core.agent.ffn_agent import EvolvableFFNAgent
from core.utils import ColourPalette, ColourType as CT
import numpy as np
import json
import csv
import os
from datetime import datetime

from core.sensor.implementation import (
    nearest_angle_sensor,
    nearest_distance_sensor,
    proximity_sensor,
    collision_sensor,
)
from core.sensor.proprioceptive import OrientationSensor
from core.simulation import Simulation
from core.evolve.genetic_algorithm import GeneticAlgorithm, GA_SELECTION_TYPE
from core.evolve.base import Group
from core.evolve.population import Population

IS_DEMO = True
DEMO_NAME = "Evo Mouse"
CLASS_NAME = "EvoMouseSimulation"

@dataclass(frozen=True)
class SensorConfig:
    name: str
    sensor_range: float = 400.0
    include_angle: bool = True
    include_distance: bool = False
    include_beams: bool = False
    beam_scope: float = np.pi / 3
    beam_orientation: float = np.pi / 4
    beam_simple: bool = True
    include_contact: bool = False
    include_self_orientation: bool = False  # A2: proprioceptive sensor

DEFAULT_SENSOR_CONFIG = SensorConfig(name="angle_only")
DEFAULT_HIDDEN_NODES = 12


class Cheese(WorldObject):
    def __init__(self):
        super().__init__()
        self.radius = 5.0
        self.colour = ColourPalette[CT.YELLOW]

    def eaten(self):
        self.location = self.world.random_location()


class EvoMouse(EvolvableFFNAgent):
    def __init__(self, sensor_config: SensorConfig = DEFAULT_SENSOR_CONFIG, hidden_nodes: int = DEFAULT_HIDDEN_NODES):
        super().__init__()

        self.last_cheese = 0
        self.cheese_found = 0
        self._cheese_scores = []  # Track cheese collected per assessment
        sensor_range = sensor_config.sensor_range

        if sensor_config.include_angle:
            self.add_sensor("angle", nearest_angle_sensor(Cheese, sensor_range))
        if sensor_config.include_distance:
            self.add_sensor("distance", nearest_distance_sensor(Cheese, sensor_range))
        if sensor_config.include_beams:
            scope = sensor_config.beam_scope
            orientation = sensor_config.beam_orientation
            self.add_sensor(
                "left_beam",
                proximity_sensor(
                    Cheese,
                    scope,
                    sensor_range,
                    orientation=orientation,
                    simple=sensor_config.beam_simple,
                ),
            )
            self.add_sensor(
                "right_beam",
                proximity_sensor(
                    Cheese,
                    scope,
                    sensor_range,
                    orientation=-orientation,
                    simple=sensor_config.beam_simple,
                ),
            )
        if sensor_config.include_contact:
            self.add_sensor("front_contact", collision_sensor(Cheese))
        if sensor_config.include_self_orientation:#self
            self.add_sensor("self_orientation", OrientationSensor())

        self._interaction_range = sensor_range
        self.radius = 10
        self.add_brain(hidden_nodes)

    def control(self):
        super().control()
        for key in self.controls.keys():
            self.controls[key] = self.controls[key] + 0.5

    def on_collision(self, obj):
        if isinstance(obj, Cheese):
            self.cheese_found += 1
            obj.eaten()

    def get_fitness(self):
        return self.cheese_found

    def store_fitness(self):
        # Store both fitness and cheese before reset
        super().store_fitness()
        self._cheese_scores.append(self.cheese_found)

    @property
    def average_cheese(self):
        if len(self._cheese_scores) != 0:
            return sum(self._cheese_scores) / len(self._cheese_scores)
        else:
            return 0.0

    def reset(self):
        self.cheese_found = 0
        super().reset()


class EvoMouseSimulation(Simulation):
    def __init__(
        self,
        sensor_config: SensorConfig = DEFAULT_SENSOR_CONFIG,
        hidden_nodes: int = DEFAULT_HIDDEN_NODES,
        population_size: int = 30,
        generations: int = 100,
        timesteps: int = 300,
        assessments: int = 1,
        experiment_name: str = "default",#test
        output_dir: str = "experiments/results",
        seed: int = None,
    ):
        super().__init__("EvoMouse")

        self.generations = generations
        self.assessments = assessments
        self.timesteps = timesteps

        self.population_size = population_size
        self.algortihm = GeneticAlgorithm(0.25, 0.1, selection=GA_SELECTION_TYPE.ROULETTE)

        self.fitness_history = []
        self.best_fitness_history = []
        self.best_cheese_history = []  # Track actual cheese collected

        # Experiment tracking
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.seed = seed if seed is not None else np.random.randint(0, 100000)
        self.sensor_config = sensor_config
        self.hidden_nodes = hidden_nodes

        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Set random seed
        np.random.seed(self.seed)

        self.add(
            "mice",
            Population(
                self.population_size,
                EvoMouse,
                self.algortihm,
                sensor_config=sensor_config,
                hidden_nodes=hidden_nodes,
            ),
        )
        self.add("cheese", Group(self.population_size, Cheese))

    def log_end_generation(self):
        averages = self.contents["mice"].average_member_fitness()
        average = sum(averages) / len(averages)
        self.fitness_history.append(average)

        best = max(averages) if averages else 0.0
        self.best_fitness_history.append(best)

        # Track actual cheese collected by best agent
        best_agent = self.contents["mice"].get_best_member()
        if best_agent and hasattr(best_agent, 'average_cheese'):
            self.best_cheese_history.append(best_agent.average_cheese)
        else:
            self.best_cheese_history.append(0)

        self.log.info(f"Generation {self._generation}: Avg Fitness: {average:.3f}, Best: {best:.3f}")

    def log_end_run(self):
        # Save CSV data
        csv_path = os.path.join(self.output_dir, f"{self.experiment_name}_seed{self.seed}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "avg_fitness", "best_fitness"])
            for gen, (avg, best) in enumerate(zip(self.fitness_history, self.best_fitness_history)):
                writer.writerow([gen, avg, best])

        # Save JSON metadata
        json_path = os.path.join(self.output_dir, f"{self.experiment_name}_seed{self.seed}_config.json")
        config_data = {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "population_size": self.population_size,
                "generations": self.generations,
                "timesteps": self.timesteps,
                "assessments": self.assessments,
                "hidden_nodes": self.hidden_nodes,
            },
            "sensor_config": {
                "name": self.sensor_config.name,
                "sensor_range": self.sensor_config.sensor_range,
                "include_angle": self.sensor_config.include_angle,
                "include_distance": self.sensor_config.include_distance,
                "include_beams": self.sensor_config.include_beams,
                "beam_scope": self.sensor_config.beam_scope,
                "beam_orientation": self.sensor_config.beam_orientation,
                "beam_simple": self.sensor_config.beam_simple,
                "include_contact": self.sensor_config.include_contact,
                "include_self_orientation": self.sensor_config.include_self_orientation,
            },
            "results": {
                "final_avg_fitness": float(self.fitness_history[-1]) if self.fitness_history else 0,
                "final_best_fitness": float(self.best_fitness_history[-1]) if self.best_fitness_history else 0,
                "max_fitness_achieved": float(max(self.best_fitness_history)) if self.best_fitness_history else 0,
            }
        }
        with open(json_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        # Save plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, linewidth=2, label='Average Fitness')
        plt.plot(self.best_fitness_history, linewidth=2, label='Best Fitness', alpha=0.7)
        plt.xlabel("Generation", fontsize=12)
        plt.ylabel("Fitness (Cheese Found)", fontsize=12)
        plt.title(f"{self.experiment_name.replace('_', ' ').title()} (seed={self.seed})", fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_path = os.path.join(self.output_dir, f"{self.experiment_name}_seed{self.seed}.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        self.log.info(f"Results saved to {self.output_dir}")
