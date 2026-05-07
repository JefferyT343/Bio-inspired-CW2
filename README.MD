# PyBEAST++ — Coursework 2: Co-evolutionary Predator–Prey Simulation

PyBEAST++ is a refactor of the University of Leeds courseware
(contributions from James Borgars and John Hodrien): bug fixes, NumPy-based
vectors, higher framerate, simpler setup, and a Qt GUI replacing WxWidgets.

This repository contains our group's Coursework 2 solution: a co-evolutionary predator–prey simulation with safe / resource zones, plus several experiment variants (zone distance, predator sensors, resource depletion).

The project can be found on GitHub with the link:
https://github.com/JefferyT343/Bio-inspired-CW2.git
---

## 1. Project layout

```
Bio-inspired-CW2/
├── main.py                  ← GUI entry point (interactive demo picker)
├── run_twostage.py          ← Headless runner: TwoStageEvolution
├── plot_twostage_results.py ← Re-plot existing CSV results in more detail
├── environment.yml          ← Conda environment spec
├── apptainer.def, run.sh    ← Apptainer (Singularity) image build & run
├── core/                    ← Engine: world, agents, sensors, GA
├── gui/                     ← Qt GUI
├── demos/                   ← Coursework simulations (one file = one variant)
└── results/                 ← CSV + PNG output (auto-created on first run)
```

---

## 2. Installation

### Option 1

Tested on School of Computer Science Linux machines and Windows 11.
Requires a pre-installed Anaconda / Miniconda.

```bash
# First-time setup
conda env create -f environment.yml
conda activate pybeast

# Subsequent sessions
conda activate pybeast
```

### Option 2

A bash script `run.sh` is provided that builds the image on first run and
re-uses it on subsequent runs (the working directory is bind-mounted so
output persists). See the upstream PyBEAST Apptainer repository for
detailed instructions.

---

## 3. Running a simulation

There are two run modes — pick whichever fits your goal.

### 3.1  GUI mode  —  visual, interactive

Launch the demo picker, then choose a simulation from the dropdown:

```bash
python main.py
```

### 3.2  Headless mode  —  fast, for evolution runs

Headless skips rendering, so a 200-generation co-evolution run finishes
much faster. Two ready-to-go launchers are provided:

```bash
python run_twostage.py        # runs demos/TwoStageEvolution.py
```

To run a different variant headless, copy one of these launchers and
change the import + class name:

```python
from demos.<file_stem> import <CLASS_NAME>

if __name__ == "__main__":
    sim = <CLASS_NAME>()
    sim.run_simulation_no_render()
```
**CLASS_NAME can be found at the top of each demo file.**
---

## 5. Output and plotting

When a simulation finishes, two files are written into `results/`:

* `results/<run_name>_<N>gens.csv` — per-generation metrics
  (prey/predator fitness, energy, food collected, deaths, and for
  `ResourceDepletion.py` two extra resource-depletion diagnostic columns).
* `results/<run_name>_<N>gens.png` — fitness curves with stage shading.

For richer multi-panel plots from an existing CSV (energy, deaths, food
collected, etc.), run:

```bash
python plot_twostage_results.py
```

Edit the `results_dir` / target CSV name at the top of that file to point at the run you want to re-plot.
