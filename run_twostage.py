"""
Standalone runner for TwoStageEvolution simulation (headless mode).
"""
import sys
sys.path.insert(0, '.')

from demos.TwoStageEvolution import CoevSimulationTwoStage

if __name__ == "__main__":
    sim = CoevSimulationTwoStage()
    sim.run_simulation_no_render()
