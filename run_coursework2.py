"""
Standalone runner for coursework2_medium simulation (headless mode).
"""
import sys
sys.path.insert(0, '.')

from demos.coursework2_medium import CoevSimulationMedium

if __name__ == "__main__":
    sim = CoevSimulationMedium()
    sim.run_simulation_no_render()
