"""
Standalone runner for TwoStageEvolution simulation (headless mode).
Can change the demo name and simulation class to run other demos in headless mode as well.
e.g. to run the coursework2_medium demo change to:
from demos.coursework2_medium import CourseSimMedium, sim = CourseSimMedium()
"""
import sys
sys.path.insert(0, '.')

from demos.TwoStageEvolution import CoevSimulationTwoStage

if __name__ == "__main__":
    sim = CoevSimulationTwoStage()
    sim.run_simulation_no_render()
