<p align="center">
  <img
    src="https://github.com/user-attachments/assets/12e78228-7b02-4d1e-a115-cba22114cee3"
    width="640"
    height="480"
    alt="image"
  />
</p>


This repository contains the full experimental framework for the CS 776 Final Project, exploring **niching, speciation, and peak coverage** in evolutionary algorithms. Three algorithm families are implemented and evaluated across standard multimodal benchmarks (F1, F2, F3):

- Genetic Algorithm (GA)
- Vanilla Population-Based Incremental Learning (PBIL)
- Island-based PBIL with multiple coordination modes

All scripts are designed for **reproducible, large-scale experimental sweeps**, and results are written to a standardized directory structure for analysis and reporting.

---

## Command Reference Overview

This document summarizes the **command-line interfaces**, **available flags**, and **algorithmic behavior** for each executable script. The descriptions directly reflect the argument parsers and logic implemented in the current codebase.

Scripts covered:
- `GA.py`
- `vanilla_pbil.py`
- `pbil_island.py`

---

# Genetic Algorithm (GA.py)

## Command Syntax

```
python GA.py --func <F1|F2|F3> --runs <int> [--seed <int>]
```

## Flags

| Flag | Default | Description |
|-----|--------|-------------|
| `--func` | F1 | Benchmark function: F1, F2, or F3 |
| `--runs` | 30 | Number of independent GA runs |
| `--seed` | None | Base RNG seed; run *i* uses `seed + i` |

## Behavior

- Population size: **100 individuals**, each encoded as a 30-bit chromosome
- Fitness evaluations per run: **20,000** (200 generations)
- Selection: tournament selection (size = 2)
- Variation operators:
  - One-point crossover (probability 0.9)
  - Bit-flip mutation (0.001 per bit)
- Elitism: best individual preserved each generation
- Niching: **phenotypic restricted mating** and **restricted tournament replacement (RTR)**

Each invocation writes results to:

```
run_GA/<FUNC>/run_N/
```

including convergence plots, final solution distributions, and summary statistics.

---

# Vanilla PBIL (vanilla_pbil.py)

## Command Syntax

```
python vanilla_pbil.py --func <F1|F2|F3> --runs <int> \
                       [--seed <int>] [--total_evals <int>] [--num_samples <int>]
```

## Flags

| Flag | Default | Description |
|-----|--------|-------------|
| `--func` | F1 | Benchmark function |
| `--runs` | 30 | Number of independent PBIL runs |
| `--seed` | None | Base RNG seed; run *i* uses `seed + i` |
| `--total_evals` | 20000 | Total fitness evaluations per run |
| `--num_samples` | 50 | Samples drawn per generation |

## Behavior

- Maintains a **single probability vector** `p ∈ [0,1]^30`, initialized near 0.5
- Each generation:
  - Samples `num_samples` bitstrings
  - Evaluates fitness
  - Updates `p` toward the best individual (learning rate α = 0.1)
  - Applies probabilistic mutation toward 0.5
- Number of generations = `total_evals / num_samples`

Results are written to:

```
run_PBIL/<FUNC>/run_N/
```

including convergence curves, final solution plots, and peak occupancy statistics.

---

# Island PBIL (pbil_island.py)

Island PBIL runs **multiple PBIL instances (islands)** in parallel, with optional coordination mechanisms designed to promote multimodal coverage.

## Command Syntax

```
python pbil_island.py --func <F1|F2|F3> --mode <none|base|proportional|peak_repulsion|peak_walk> \
                      --num_islands <int> --runs <int> \
                      [--seed <int>] [--total_evals <int>] [--num_samples <int>] \
                      [--niching_interval <int>] [--sigma_niche <float>] \
                      [--nudge_strength <float>] [--no_nudge]
```

## Flags

| Flag | Default | Description |
|-----|--------|-------------|
| `--func` | F1 | Benchmark function |
| `--mode` | none | Island coordination mode |
| `--num_islands` | 5 | Number of PBIL islands |
| `--runs` | 30 | Independent runs |
| `--seed` | None | Base RNG seed |
| `--total_evals` | 20000 | Total evaluations per run (shared across islands) |
| `--num_samples` | 50 | Samples per island per generation |
| `--niching_interval` | 10 | Interval for niching logic (when applicable) |
| `--sigma_niche` | F1/F2: 0.1, F3: 1.0 | Phenotypic distance threshold |
| `--nudge_strength` | 1.0 | Step size scaling for nudging |
| `--no_nudge` | off | Disable nudging (use random reinitialization) |

---

## Island PBIL Modes

### Mode: `none`

- Islands are completely independent
- Equivalent to running multiple PBIL instances concurrently
- No information sharing or coordination

---

### Mode: `base` (Speciation)

- Designed for cases where **number of islands ≈ number of peaks**
- Periodically clusters islands by phenotype
- Within each niche:
  - Best island is retained
  - Extra islands are nudged or reinitialized to explore elsewhere

---

### Mode: `proportional`

- Designed for **large island counts** (e.g., 50–100)
- Niche occupancy is proportional to niche fitness
- Stronger peaks attract more islands via cloning
- Weaker peaks retain fewer representatives

---

### Mode: `peak_repulsion`

- Uses **known peak locations** (Deb–Goldberg for F1/F2, Himmelblau minima for F3)
- Ensures at most one island occupies each peak
- Colliding islands are actively repelled and reassigned
- Guarantees peak coverage when enough islands exist

---

### Mode: `peak_walk`

- Explicit **goal-directed peak coverage** with continuous trajectories
- Each island:
  - Starts from a random initialization
  - Is assigned a target peak
  - Gradually *walks* through the search space via probabilistic nudging
- PBIL learning is temporarily suppressed during long-distance travel to prevent basin capture
- Once an island reaches a peak, it claims ownership and remains tethered
- Collision handling dynamically reroutes islands to uncovered peaks

This mode produces the characteristic **smooth island trace plots**, showing islands traversing plains before ascending peaks.

---

## Outputs

All Island PBIL runs write to:

```
run_IslandPBIL/<FUNC>/run_N/
```

Artifacts include:
- Best-fitness spaghetti plots
- Final island solutions
- Peak occupancy histograms
- Representative solutions per peak
- Island trajectory plots (`island_traces.png`)

---

## Folder Structure

```
run_<MODEL>/<FUNCTION>/run_#/
```

Where:
- `<MODEL>` ∈ {GA, PBIL, IslandPBIL}
- `<FUNCTION>` ∈ {F1, F2, F3}
- `run_#` increments automatically

Each directory contains CSV logs, JSON summaries, and PNG figures suitable for direct inclusion in the final report.

---

## Helper Scripts

- `run_all.sh` / `run_all_full_coverage.sh`
  - Executes comprehensive experimental sweeps across all algorithms, modes, and parameter settings

Run with:

```
bash run_all.sh
```

---

This framework is designed to support **reproducible multimodal optimization research**, with a clear separation between algorithm logic, experimental control, and result analysis.

