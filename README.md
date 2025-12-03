# CS776_Niching_PBIL
CS 776 Final Research Project. Niching and Speciation in Parallel Population-Based Incremental Learning

CS 776 Final Project — Command Reference
======================================

This text file summarizes the command-line interfaces for the three
evolutionary algorithms provided in this project: **Genetic Algorithm (GA)**,
**vanilla Population-Based Incremental Learning (PBIL)**, and **Island PBIL**.
It lists available flags, their default values, and the overall behavior of
each program. These descriptions mirror the argument parsers implemented in
the Python scripts.

-------------------------------------------------------------------------------
1. Genetic Algorithm (`GA.py`)
-------------------------------------------------------------------------------

**Command Syntax**

    python GA.py --func <F1|F2|F3> --runs <int> [--seed <int>]

**Flags**

| Flag       | Default | Description                                               |
|------------|---------|-----------------------------------------------------------|
| `--func`   | `F1`    | Selects the benchmark function. Must be one of `F1`, `F2`, or `F3`. |
| `--runs`   | `30`    | Number of independent GA runs. Each run produces its own set of outputs. |
| `--seed`   | `None`  | Base random number seed. If provided, run *i* uses seed + i.            |

**Behavior**

- Population size: 100 individuals with 30‐bit chromosomes.
- Generations: 200 (20,000 total fitness evaluations per run).
- Operators: one-point crossover (probability 0.9) and bit-flip mutation (0.001 per bit).
- Selection: tournament selection (size 2).
- Niching: phenotypic restricted mating plus restricted tournament replacement (RTR).
- Elitism: the best individual is preserved each generation.

Each invocation writes results into a directory structure `run_GA/<FUNC>/run_N/`.

-------------------------------------------------------------------------------
2. Vanilla PBIL (`vanilla_pbil.py`)
-------------------------------------------------------------------------------

**Command Syntax**

    python vanilla_pbil.py --func <F1|F2|F3> --runs <int> \
                          [--seed <int>] [--total_evals <int>] [--num_samples <int>]

**Flags**

| Flag             | Default | Description                                                                    |
|------------------|---------|--------------------------------------------------------------------------------|
| `--func`         | `F1`    | Benchmark to optimize. One of `F1`, `F2`, or `F3`.                            |
| `--runs`         | `30`    | Number of independent PBIL runs.                                              |
| `--seed`         | `None`  | Base seed; run *i* uses seed + i if provided.                                 |
| `--total_evals`  | `20000` | Total fitness evaluations allowed per run.                                    |
| `--num_samples`  | `50`    | Number of individuals sampled from the probability vector each generation.    |

**Behavior**

- Maintains a single probability vector `p` (length 30). Each bit is initialized to 0.5.
- Samples `num_samples` candidate solutions every generation.
- Updates `p` toward the best sampled individual using a learning rate (α=0.1).
- Occasional mutation perturbs entries of `p` toward 0.5.
- Runs until `total_evals` evaluations are consumed. Number of generations = `total_evals / num_samples`.

Outputs are written to `run_PBIL/<FUNC>/run_N/` with statistics and plots of convergence, final solutions, and peak occupancy.

-------------------------------------------------------------------------------
3. Island PBIL (`pbil_island.py`)
-------------------------------------------------------------------------------

**Command Syntax**

    python pbil_island.py --func <F1|F2|F3> --mode <none|base|proportional> \
                          --num_islands <int> --runs <int> \
                          [--seed <int>] [--total_evals <int>] \
                          [--num_samples <int>] [--niching_interval <int>] \
                          [--sigma_niche <float>]

**Flags**

| Flag               | Default       | Description                                                                                                 |
|--------------------|---------------|-------------------------------------------------------------------------------------------------------------|
| `--func`           | `F1`          | Benchmark function. One of `F1`, `F2`, or `F3`.                                                             |
| `--mode`           | `none`        | Niching mode: `none` (independent islands), `base` (speciation), or `proportional` (peak-height reallocation). |
| `--num_islands`    | `5`           | Number of PBIL islands running in parallel.                                                                 |
| `--runs`           | `30`          | Number of independent island PBIL runs.                                                                     |
| `--seed`           | `None`        | Base random seed; run *i* uses seed + i if provided.                                                       |
| `--total_evals`    | `20000`       | Total evaluations per run; evaluations are divided among islands.                                           |
| `--num_samples`    | `50`          | Samples per island per generation.                                                                          |
| `--niching_interval` | `10`        | Apply the niching reallocation step every `niching_interval` generations (only for `base` or `proportional` modes). |
| `--sigma_niche`    | See below     | Distance threshold for niche detection. Defaults to 0.1 for F1/F2 and 1.0 for F3 if not provided.            |

**Behavior**

- Islands maintain separate probability vectors and sample populations independently.  When more than one island is used the per‑generation updates are executed in parallel using separate processes via the ``multiprocessing`` module with the ``fork`` start method.
- **Mode `none`**: islands never coordinate; this is equivalent to `num_islands` independent PBIL runs, now executed concurrently.
- **Mode `base`** (speciation): appropriate when the number of islands matches the number of peaks.  At each niching interval the algorithm groups islands by phenotype; each niche retains its best island and resets any additional members to explore other regions.
- **Mode `proportional`** (peak‑height allocation): suited to a large number of islands.  After each niching interval, islands are assigned to niches in proportion to niche fitness by cloning representatives or resetting islands until the desired distribution is reached.
- Niching modes use `--niching_interval` to control how often niche detection occurs, and `--sigma_niche` to define the phenotypic distance threshold.

Results are stored in `run_IslandPBIL/<FUNC>/run_N/` directories, including aggregated fitness curves, final island solutions, peak occupancy histograms, and representative solutions per peak.

-------------------------------------------------------------------------------
4. Folder Structure
-------------------------------------------------------------------------------

All algorithms write their outputs into a nested directory hierarchy:

    run_<MODEL>/<FUNCTION>/run_#/...

where `<MODEL>` is one of `GA`, `PBIL`, or `IslandPBIL`; `<FUNCTION>` is the benchmark ID (`F1`, `F2`, or `F3`); and `run_#` is an incrementing index for each invocation. Within each `run_#` directory you will find CSV files, JSON summaries, and PNG plots summarizing the run’s performance.

-------------------------------------------------------------------------------
5. Helpful Scripts
-------------------------------------------------------------------------------

- `Makefile` provides one-line commands such as `make ga_f2` or `make islands_prop_f2` to run standard experiments.
- `run_all.sh` executes a full sweep across all functions, modes, and island configurations for comprehensive results. Run it with:

      bash run_all.sh

-------------------------------------------------------------------------------
