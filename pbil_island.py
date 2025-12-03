"""Island-based PBIL variants.

This module defines basic and niching versions of a PBIL algorithm that
maintain separate probability vectors for several islands.  Each island
updates its own probability vector towards the best sampled individual.
An optional niching mechanism periodically groups islands by phenotype and
reinitialises or clones probability vectors to spread coverage across
peaks.  In addition, the island updates are now parallelised using the
`multiprocessing` module with the `fork` start method when multiple islands
are configured.
"""
import numpy as np
import benchmarks as bm
import json
import csv
import matplotlib

# Disable interactive backends for plotting
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Use multiprocessing to parallelise island updates when num_islands > 1
import multiprocessing

# ---- Known peak / optimum positions ----

# For F1 and F2 in [0,1], Deb & Goldberg peaks are approximately here:
PEAK_POS_1D = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)

# For Himmelblau (F3), these are the 4 known global minima.
HIMMELBLAU_MINIMA = np.array([
    [ 3.0,       2.0      ],
    [-2.805118,  3.131312 ],
    [-3.779310, -3.283186 ],
    [ 3.584428, -1.848126 ],
], dtype=float)


class IslandPBILBase:
    """Baseline island PBIL optimiser.

    Maintains one probability vector per island and updates each of them
    independently towards the best sampled individual.  For efficiency
    the island updates are performed in parallel using `multiprocessing`
    when more than one island is present.  Subclasses may override the
    ``after_generation`` hook to implement niching or other coordination.
    """

    def __init__(
        self,
        func_id: str,
        num_islands: int = 5,
        total_evals: int = 20000,
        num_samples: int = 50,
        alpha: float = 0.1,
        pmutate_prob: float = 0.02,
        mutation_shift: float = 0.05,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        """Initialise the baseline island PBIL.

        Arguments correspond to the benchmark identifier, number of islands,
        total function evaluations, samples per island per generation, and
        learning and mutation rates.  A custom ``rng`` can be supplied for
        reproducibility; otherwise ``default_rng`` is used.
        """
        self.func_id = func_id.upper()
        self.num_islands = num_islands
        self.total_evals = total_evals
        self.num_samples = num_samples
        self.alpha = alpha
        self.pmutate_prob = pmutate_prob
        self.mutation_shift = mutation_shift
        self.rng = rng if rng is not None else np.random.default_rng()
        # Per‑island probability vectors (initially all 0.5)
        self.p = np.full((num_islands, 30), 0.5, dtype=float)
        # Determine decoding and objective functions
        if self.func_id in ("F1", "F2"):
            self.decode = bm.bits_to_real_1d
        elif self.func_id == "F3":
            self.decode = bm.bits_to_real_2d
        else:
            raise ValueError(f"Unknown function id: {func_id}")
        self.func = bm.get_function(func_id)
        # Number of generations = total_evals / (num_islands * num_samples)
        denom = max(1, self.num_islands * self.num_samples)
        self.n_generations = max(1, self.total_evals // denom)
        # Logs: one list per island
        self.best_fitness_per_gen: List[List[float]] = [list() for _ in range(num_islands)]
        self.best_x_per_gen: List[List[Any]] = [list() for _ in range(num_islands)]

    def _sample_population(self, island_idx: int) -> np.ndarray:
        """Return a sample of ``num_samples`` bitstrings for an island."""
        return (self.rng.random((self.num_samples, 30)) < self.p[island_idx]).astype(np.int8)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate a population and return an array of fitnesses."""
        fitness = np.empty(self.num_samples, dtype=float)
        if self.func_id in ("F1", "F2"):
            for i, bits in enumerate(population):
                fitness[i] = self.func(self.decode(bits))
        else:
            for i, bits in enumerate(population):
                x1, x2 = self.decode(bits)
                fitness[i] = self.func(x1, x2)
        return fitness

    def _update_island(self, island_idx: int) -> Tuple[float, Any]:
        """Update a single island and record its best fitness and phenotype."""
        pop = self._sample_population(island_idx)
        fitness = self._evaluate_population(pop)
        idx_best = int(np.argmax(fitness))
        best_bits = pop[idx_best]
        best_fit = float(fitness[idx_best])
        best_decoded = self.decode(best_bits)
        # Update probability vector
        self.p[island_idx] = (1.0 - self.alpha) * self.p[island_idx] + self.alpha * best_bits
        # Mutation
        mask = self.rng.random(self.p[island_idx].shape) < self.pmutate_prob
        self.p[island_idx][mask] = (1.0 - self.mutation_shift) * self.p[island_idx][mask] + self.mutation_shift * 0.5
        # Log
        self.best_fitness_per_gen[island_idx].append(best_fit)
        self.best_x_per_gen[island_idx].append(best_decoded)
        return best_fit, best_decoded

    def after_generation(self, gen: int) -> None:
        """Hook for subclasses; called after each generation."""
        pass

    def run(self) -> Dict[str, Any]:
        """Execute the island PBIL for the configured number of generations.

        When more than one island is present the per‑generation updates
        are dispatched concurrently using the ``multiprocessing`` module
        with the ``fork`` start method.  At the end of the run one
        individual is sampled from each island's probability vector.
        """
        ctx: Optional[multiprocessing.context.BaseContext] = None
        if self.num_islands > 1:
            ctx = multiprocessing.get_context("fork")
        for gen in range(self.n_generations):
            if self.num_islands > 1 and ctx is not None:
                args_list: List[Tuple[np.ndarray, str, int, float, float, float, int]] = []
                for k in range(self.num_islands):
                    seed = int(self.rng.integers(0, 2**32))
                    args_list.append((self.p[k].copy(), self.func_id, self.num_samples,
                                      self.alpha, self.pmutate_prob, self.mutation_shift, seed))
                with ctx.Pool(self.num_islands) as pool:
                    results = pool.map(_island_worker, args_list)
                for k, (new_p, best_fit, best_decoded) in enumerate(results):
                    self.p[k] = new_p
                    self.best_fitness_per_gen[k].append(best_fit)
                    self.best_x_per_gen[k].append(best_decoded)
            else:
                for k in range(self.num_islands):
                    self._update_island(k)
            self.after_generation(gen)
        final_decoded: List[Any] = []
        final_fitness: List[float] = []
        for k in range(self.num_islands):
            bits = (self.rng.random(30) < self.p[k]).astype(np.int8)
            decoded = self.decode(bits)
            if self.func_id in ("F1", "F2"):
                fit = self.func(decoded)
            else:
                fit = self.func(*decoded)  # type: ignore[arg-type]
            final_decoded.append(decoded)
            final_fitness.append(float(fit))
        return {
            "best_fitness_per_gen": self.best_fitness_per_gen,
            "best_x_per_gen": self.best_x_per_gen,
            "final_p": self.p.copy(),
            "final_sample_decoded": final_decoded,
            "final_fitness": final_fitness,
        }

# Helper for multiprocessing: perform one PBIL generation for a single island.
# The function is defined at module scope so that it can be pickled by
# multiprocessing.  It takes a copy of the island's probability vector and
# algorithm parameters, runs one generation of sampling, evaluation and
# update, and returns the updated probability vector along with the best
# fitness and decoded phenotype.
def _island_worker(args: Tuple[np.ndarray, str, int, float, float, float, int]) -> Tuple[np.ndarray, float, Any]:
    """Worker function for a single island PBIL generation.

    Takes a copy of the island's probability vector and algorithm
    parameters, samples a population, evaluates it, updates the vector
    towards the best individual, applies mutation, and returns the
    updated vector along with the best fitness and decoded phenotype.
    """
    p_row, func_id, num_samples, alpha, pmutate_prob, mutation_shift, seed = args
    rng = np.random.default_rng(seed)
    # Decode and objective functions
    if func_id in ("F1", "F2"):
        decode = bm.bits_to_real_1d
    else:
        decode = bm.bits_to_real_2d
    func = bm.get_function(func_id)
    rand = rng.random((num_samples, 30))
    pop = (rand < p_row).astype(np.int8)
    fitness = np.empty(num_samples, dtype=float)
    if func_id in ("F1", "F2"):
        for i, bits in enumerate(pop):
            fitness[i] = func(decode(bits))
    else:
        for i, bits in enumerate(pop):
            x1, x2 = decode(bits)
            fitness[i] = func(x1, x2)
    idx_best = int(np.argmax(fitness))
    best_bits = pop[idx_best]
    best_fit = float(fitness[idx_best])
    best_decoded = decode(best_bits)
    new_p = (1.0 - alpha) * p_row + alpha * best_bits
    mask = rng.random(new_p.shape) < pmutate_prob
    new_p[mask] = (1.0 - mutation_shift) * new_p[mask] + mutation_shift * 0.5
    return new_p, best_fit, best_decoded


class IslandPBILNiching(IslandPBILBase):
    """Island PBIL with optional niching.

    Inherits the baseline PBIL and adds periodic niche detection.  In
    ``base`` mode, excess islands in a niche are reset to explore other
    regions; in ``proportional`` mode, the number of islands per niche is
    adjusted according to niche fitnesses.
    """

    def __init__(
        self,
        func_id: str,
        num_islands: int = 5,
        total_evals: int = 20000,
        num_samples: int = 50,
        alpha: float = 0.1,
        pmutate_prob: float = 0.02,
        mutation_shift: float = 0.05,
        mode: str = 'base',
        niching_interval: int = 10,
        sigma_niche: float | None = None,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        super().__init__(
            func_id=func_id,
            num_islands=num_islands,
            total_evals=total_evals,
            num_samples=num_samples,
            alpha=alpha,
            pmutate_prob=pmutate_prob,
            mutation_shift=mutation_shift,
            rng=rng,
        )
        self.mode = mode.lower()
        if sigma_niche is None:
            # Default threshold: 0.1 for 1D, 1.0 for 2D (squared distance)
            sigma_niche = 0.1 if func_id.upper() in ('F1', 'F2') else 1.0
        self.sigma_niche = sigma_niche
        self.niching_interval = niching_interval

    def _decode_best_per_island(self) -> Tuple[List[Any], List[float]]:
        """Return current best decoded phenotype and fitness for each island.

        Uses the last recorded generation's best values.
        """
        decoded_list: List[Any] = []
        fitness_list: List[float] = []
        for k in range(self.num_islands):
            # Use last logged best phenotype and fitness
            if not self.best_fitness_per_gen[k]:
                # If no entry (e.g., before first generation), use neutral values
                decoded_list.append(None)
                fitness_list.append(float('-inf'))
                continue
            fitness_list.append(self.best_fitness_per_gen[k][-1])
            decoded_list.append(self.best_x_per_gen[k][-1])
        return decoded_list, fitness_list

    def _cluster_islands(self, decoded_list: List[Any], fitness_list: List[float]) -> Tuple[List[int], List[List[int]]]:
        """Cluster islands into niches using greedy phenotypic distance clustering.

        Parameters
        ----------
        decoded_list : list
            Current decoded phenotype per island.
        fitness_list : list
            Current fitness per island.

        Returns
        -------
        Tuple[List[int], List[List[int]]]
            A tuple (representatives, niches) where:
                representatives: list of representative island indices (one per niche)
                niches: list of lists; each inner list contains indices of islands
                    assigned to that niche.
        """
        # Sort islands by fitness descending; store indices
        idxs = list(range(self.num_islands))
        # If fitness_list entries may contain -inf for empty logs, treat them as lowest
        sorted_islands = sorted(idxs, key=lambda i: fitness_list[i], reverse=True)
        representatives: List[int] = []
        niches: List[List[int]] = []
        for idx in sorted_islands:
            # Skip islands with undefined decoded phenotype
            pheno = decoded_list[idx]
            if pheno is None:
                continue
            if not representatives:
                # First niche
                representatives.append(idx)
                niches.append([idx])
            else:
                # Compute distances to existing representatives
                dists = []
                for rep in representatives:
                    rep_pheno = decoded_list[rep]
                    # Cast to correct distance function based on dimensionality
                    if self.func_id in ('F1', 'F2'):
                        d = bm.distance_1d(pheno, rep_pheno)  # type: ignore[arg-type]
                    else:
                        d = bm.distance_2d(pheno, rep_pheno)  # type: ignore[arg-type]
                    dists.append(d)
                # Determine if this island starts a new niche
                min_dist = min(dists)
                if min_dist > self.sigma_niche:
                    representatives.append(idx)
                    niches.append([idx])
                else:
                    # Assign to the nearest niche
                    nearest = int(np.argmin(dists))
                    niches[nearest].append(idx)
        return representatives, niches

    def _reinitialize_island(self, island_idx: int) -> None:
        """Reinitialise a given island's probability vector.

        The new probability vector is uniform (0.5 for all bits) plus small
        random noise to break symmetry.
        """
        # Uniform 0.5 plus random noise in [-0.05, 0.05]
        noise = (self.rng.random(30) - 0.5) * 0.1
        new_p = 0.5 + noise
        # Clamp to [0,1]
        new_p = np.clip(new_p, 0.0, 1.0)
        self.p[island_idx] = new_p

    def _clone_island(self, target_idx: int, source_idx: int) -> None:
        """Clone the probability vector of a source island into a target island.

        Adds small random noise to diversify the clone.
        """
        # Copy p vector from source
        new_p = self.p[source_idx].copy()
        # Add small noise in [-0.05,0.05]
        noise = (self.rng.random(30) - 0.5) * 0.1
        new_p = new_p + noise
        new_p = np.clip(new_p, 0.0, 1.0)
        self.p[target_idx] = new_p

    def after_generation(self, gen: int) -> None:
        # Only perform niching at specified intervals
        if (gen + 1) % self.niching_interval != 0:
            return
        # Gather current best decoded phenotypes and fitness per island
        decoded_list, fitness_list = self._decode_best_per_island()
        # If we have no valid decoded values, skip
        if all(pheno is None for pheno in decoded_list):
            return
        # Cluster islands into niches
        representatives, niches = self._cluster_islands(decoded_list, fitness_list)
        if not niches:
            return
        # Process based on mode
        if self.mode == 'base':
            # In base mode, we aim for one island per niche when #islands == #peaks
            # For each niche, keep the representative and reinitialize other members
            for rep, members in zip(representatives, niches):
                # members includes rep; we keep rep and reset others
                for idx in members:
                    if idx == rep:
                        continue
                    self._reinitialize_island(idx)
        elif self.mode == 'proportional':
            # Proportional occupancy: compute weights and desired counts
            M = self.num_islands
            C = len(niches)
            # Compute w_c = max(f_c, 0)
            reps_fitness = [fitness_list[rep] for rep in representatives]
            # Handle negative fitness values: if all <= 0, assign equal weights
            max_pos = [max(f, 0.0) for f in reps_fitness]
            if sum(max_pos) > 0:
                weights = [f / sum(max_pos) for f in max_pos]
            else:
                # Equal weights
                weights = [1.0 / C] * C
            # Desired counts
            desired_counts = [int(round(w * M)) for w in weights]
            # Adjust to ensure sum equals M
            diff = M - sum(desired_counts)
            # If diff > 0, add 1 to some niches; if diff < 0, subtract 1
            i = 0
            while diff != 0 and C > 0:
                idx = i % C
                if diff > 0:
                    desired_counts[idx] += 1
                    diff -= 1
                else:  # diff < 0
                    if desired_counts[idx] > 0:
                        desired_counts[idx] -= 1
                        diff += 1
                i += 1
            # Now redistribute islands according to desired_counts
            # Build a list of islands that are currently free (to be reassigned)
            free_islands: List[int] = []
            # For each niche, handle excess and deficit
            for niche_idx, members in enumerate(niches):
                current = len(members)
                target = desired_counts[niche_idx]
                if current > target:
                    # Excess: randomly select which members (excluding rep) to free
                    # Determine representative index within this niche
                    rep = representatives[niche_idx]
                    # Candidates for removal: all except rep
                    candidates = [idx for idx in members if idx != rep]
                    self.rng.shuffle(candidates)
                    to_remove = candidates[: (current - target)]
                    for idx in to_remove:
                        # Reinitialise these islands and add to free pool
                        self._reinitialize_island(idx)
                        free_islands.append(idx)
                        # Remove idx from members
                        members.remove(idx)
                elif current < target:
                    # Deficit: need to fill up to target
                    needed = target - current
                    for _ in range(needed):
                        if free_islands:
                            idx = free_islands.pop()
                        else:
                            # If no free island, randomly choose one that is not rep
                            candidates = [j for j in range(self.num_islands)
                                          if j not in representatives]
                            if not candidates:
                                # Nothing to reassign; break
                                break
                            idx = self.rng.choice(candidates)
                        # Clone representative's p into idx
                        rep = representatives[niche_idx]
                        self._clone_island(idx, rep)
                        members.append(idx)
            # Any remaining free islands remain reinitialised (uniform) and will
            # wander until they latch onto a niche
        else:
            raise ValueError(f"Unknown niching mode: {self.mode}")

    def run(self) -> Dict[str, Any]:
        """Run the niching island PBIL and return logs and final results."""
        return super().run()


def run_island_pbil(
    func_id: str,
    mode: str = 'none',
    num_islands: int = 5,
    total_evals: int = 20000,
    num_samples: int = 50,
    alpha: float = 0.1,
    pmutate_prob: float = 0.02,
    mutation_shift: float = 0.05,
    niching_interval: int = 10,
    sigma_niche: float | None = None,
    num_runs: int = 1,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of island‑PBIL run dictionaries across ``num_runs`` experiments."""
    results: List[Dict[str, Any]] = []
    for i in range(num_runs):
        rng = np.random.default_rng(None if seed is None else seed + i)
        if mode == 'none':
            obj = IslandPBILBase(
                func_id=func_id,
                num_islands=num_islands,
                total_evals=total_evals,
                num_samples=num_samples,
                alpha=alpha,
                pmutate_prob=pmutate_prob,
                mutation_shift=mutation_shift,
                rng=rng,
            )
        else:
            obj = IslandPBILNiching(
                func_id=func_id,
                num_islands=num_islands,
                total_evals=total_evals,
                num_samples=num_samples,
                alpha=alpha,
                pmutate_prob=pmutate_prob,
                mutation_shift=mutation_shift,
                mode=mode,
                niching_interval=niching_interval,
                sigma_niche=sigma_niche,
                rng=rng,
            )
        results.append(obj.run())
    return results

def prepare_run_directory(model_name: str, func_id: str) -> Path:
    """Create and return a fresh ``run_n`` directory for an island‑PBIL experiment."""
    base = Path(f"run_{model_name}")
    base.mkdir(exist_ok=True)

    func_dir = base / func_id
    func_dir.mkdir(exist_ok=True)

    # Determine next run number
    existing = [d for d in func_dir.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not existing:
        next_run = 1
    else:
        nums = [
            int(d.name.split("_")[1])
            for d in existing
            if d.name.split("_")[1].isdigit()
        ]
        next_run = max(nums) + 1

    run_dir = func_dir / f"run_{next_run}"
    run_dir.mkdir()

    return run_dir

# ---------- Helpers for aggregating island histories ----------

def aggregate_best_over_islands(result_dict):
    """
    From a single island PBIL run, compute best-over-all-islands per generation.

    result_dict['best_fitness_per_gen'] is a list of lists:
        per_island[k][g] = fitness of best individual on island k at generation g.
    """
    per_island = result_dict["best_fitness_per_gen"]
    if not per_island:
        return []

    num_islands = len(per_island)
    num_gens = len(per_island[0])
    agg = []
    for g in range(num_gens):
        best_g = max(per_island[k][g] for k in range(num_islands))
        agg.append(best_g)
    return agg


# ---------- Plotting ----------

def island_pbil_plot_spaghetti(results, func_id: str, out_path: Path) -> None:
    """Spaghetti of best-over-islands fitness per generation across runs."""
    plt.figure()
    for res in results:
        y = aggregate_best_over_islands(res)
        x = list(range(len(y)))
        plt.plot(x, y, alpha=0.4)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness over islands")
    plt.title(f"Island PBIL best fitness vs generations ({func_id})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def island_pbil_plot_1d_function_with_points(results, func_id: str, out_path: Path) -> None:
    """Plot F1/F2 and overlay final solutions from all islands across runs."""
    if func_id.upper() == "F1":
        f = bm.f1
    elif func_id.upper() == "F2":
        f = bm.f2
    else:
        raise ValueError("island_pbil_plot_1d_function_with_points only valid for F1/F2")

    xs = np.linspace(0.0, 1.0, 1000)
    ys = [f(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label=func_id)

    finals = []
    for res in results:
        # final_sample_decoded is a list: one decoded phenotype per island
        finals.extend(res["final_sample_decoded"])
    finals = [float(x) for x in finals]
    fvals = [f(x) for x in finals]

    plt.scatter(finals, fvals, c="red", s=25, alpha=0.7, label="island finals")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{func_id} with final Island PBIL islands")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def island_pbil_plot_himmelblau_with_points(results, out_path: Path) -> None:
    """Contour of Himmelblau + final solutions from all islands across runs."""
    x = np.linspace(-6, 6, 200)
    y = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = bm.f3(X[i, j], Y[i, j])

    plt.figure()
    cs = plt.contourf(X, Y, Z, levels=50)
    plt.colorbar(cs, label="F3(x1,x2) = -Himmelblau")

    finals_x1 = []
    finals_x2 = []
    for res in results:
        for decoded in res["final_sample_decoded"]:
            x1, x2 = decoded
            finals_x1.append(x1)
            finals_x2.append(x2)

    plt.scatter(finals_x1, finals_x2, c="red", s=25, alpha=0.7, label="island finals")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Island PBIL on Himmelblau (F3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Saving CSV / JSON ----------

def save_island_pbil_results_to_files(results, func_id: str, mode: str, num_islands: int, run_dir: Path) -> None:
    """
    Save per-run aggregate best fitness curves and summaries.

    Note: to keep things manageable, we save only the best-over-islands curve
    per run. If you want per-island CSVs later, we can extend this.
    """
    for i, res in enumerate(results):
        idx = i + 1

        # Aggregate best over islands per generation
        agg = aggregate_best_over_islands(res)
        csv_path = run_dir / f"island_pbil_run{idx}_best_over_islands.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness_over_islands"])
            for gen, fit in enumerate(agg):
                writer.writerow([gen, fit])

        # JSON summary
        finals = res["final_sample_decoded"]
        finals_fitness = res["final_fitness"]
        json_path = run_dir / f"island_pbil_run{idx}_summary.json"
        summary = {
            "func_id": func_id,
            "mode": mode,
            "num_islands": num_islands,
            "run_index": idx,
            "final_island_count": len(finals),
            "best_final_fitness": max(finals_fitness) if finals_fitness else None,
        }
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2)

# ----------------- Island PBIL peak analysis helpers -----------------

def _assign_peak_1d(x: float, threshold: float = 0.1) -> int | None:
    """
    Assign a 1D point x in [0,1] to the nearest F1/F2 peak index,
    or return None if it is farther than `threshold`.
    """
    diffs = np.abs(PEAK_POS_1D - x)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None


def _assign_peak_2d(x1: float, x2: float, threshold: float = 1.0) -> int | None:
    """
    Assign a 2D point (x1,x2) to the nearest Himmelblau minimum,
    or return None if it is farther than `threshold`.
    """
    point = np.array([x1, x2])
    diffs = np.linalg.norm(HIMMELBLAU_MINIMA - point, axis=1)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None

def _island_collect_all_points(results, func_id: str):
    """Collect all final island phenotypes across runs."""
    pts = []
    func_id = func_id.upper()
    if func_id in ("F1", "F2"):
        for res in results:
            pts.extend([float(x) for x in res["final_sample_decoded"]])
    else:
        for res in results:
            pts.extend([tuple(x) for x in res["final_sample_decoded"]])
    return pts

def island_pbil_plot_peak_representatives(results, func_id: str, out_path: Path) -> None:
    """
    For Island PBIL: best island per peak, aggregated across runs.
    """
    func_id = func_id.upper()
    pts = _island_collect_all_points(results, func_id)

    if func_id == "F1":
        f = bm.f1
    elif func_id == "F2":
        f = bm.f2
    else:
        def f_pair(p):
            return bm.f3(p[0], p[1])

    best_per_peak = {}

    if func_id in ("F1", "F2"):
        for x in pts:
            peak = _assign_peak_1d(x)
            if peak is None:
                continue
            fit = f(x)
            if peak not in best_per_peak or fit > best_per_peak[peak][0]:
                best_per_peak[peak] = (fit, x)
    else:
        for (x1, x2) in pts:
            peak = _assign_peak_2d(x1, x2)
            if peak is None:
                continue
            fit = f_pair((x1, x2))
            if peak not in best_per_peak or fit > best_per_peak[peak][0]:
                best_per_peak[peak] = (fit, (x1, x2))

    if func_id in ("F1", "F2"):
        xs = np.linspace(0.0, 1.0, 1000)
        ys = [f(x) for x in xs]
        plt.figure()
        plt.plot(xs, ys, label=func_id)

        rep_x = [best_per_peak[k][1] for k in sorted(best_per_peak)]
        rep_y = [f(x) for x in rep_x]
        plt.scatter(rep_x, rep_y, c="red", s=80, marker="x", label="best per peak")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{func_id}: Island PBIL niche representatives")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        x = np.linspace(-6, 6, 200)
        y = np.linspace(-6, 6, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = bm.f3(X[i, j], Y[i, j])

        plt.figure()
        cs = plt.contourf(X, Y, Z, levels=50)
        plt.colorbar(cs, label="F3 (fitness)")

        rep_x1 = []
        rep_x2 = []
        for k in sorted(best_per_peak):
            x1, x2 = best_per_peak[k][1]
            rep_x1.append(x1)
            rep_x2.append(x2)

        plt.scatter(rep_x1, rep_x2, c="red", s=80, marker="x", label="best per minima")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title("F3: Island PBIL niche representatives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def island_pbil_plot_peak_occupancy(results, func_id: str, out_path: Path) -> None:
    """
    For Island PBIL: bar plot of how many islands end on each peak (across runs).
    """
    func_id = func_id.upper()
    pts = _island_collect_all_points(results, func_id)

    if func_id in ("F1", "F2"):
        num_peaks = len(PEAK_POS_1D)
        counts = np.zeros(num_peaks, dtype=int)
        for x in pts:
            peak = _assign_peak_1d(x)
            if peak is not None:
                counts[peak] += 1
        labels = [f"Peak {i+1}" for i in range(num_peaks)]
    else:
        num_peaks = HIMMELBLAU_MINIMA.shape[0]
        counts = np.zeros(num_peaks, dtype=int)
        for (x1, x2) in pts:
            peak = _assign_peak_2d(x1, x2)
            if peak is not None:
                counts[peak] += 1
        labels = [f"Min {i+1}" for i in range(num_peaks)]

    plt.figure()
    plt.bar(range(num_peaks), counts)
    plt.xticks(range(num_peaks), labels)
    plt.ylabel("Number of islands (aggregated over runs)")
    plt.title(f"{func_id}: Island PBIL peak occupancy")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _island_best_run_index(results):
    """
    Choose best Island PBIL run by best final island fitness.
    """
    best_idx = 0
    best_val = -np.inf
    for i, res in enumerate(results):
        # final_fitness is assumed to be list of per-island fitnesses
        if not res["final_fitness"]:
            continue
        val = float(max(res["final_fitness"]))
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def island_pbil_plot_single_run_final(results, func_id: str, out_path: Path) -> None:
    """
    Plot final island solutions from the single best Island PBIL run.
    """
    func_id = func_id.upper()
    idx = _island_best_run_index(results)
    res = results[idx]
    finals = res["final_sample_decoded"]

    if func_id == "F1":
        f = bm.f1
    elif func_id == "F2":
        f = bm.f2
    else:
        def f_pair(p):
            return bm.f3(p[0], p[1])

    if func_id in ("F1", "F2"):
        xs = np.linspace(0.0, 1.0, 1000)
        ys = [f(x) for x in xs]

        plt.figure()
        plt.plot(xs, ys, label=func_id)

        finals = [float(x) for x in finals]
        fvals = [f(x) for x in finals]
        plt.scatter(finals, fvals, c="red", s=30, alpha=0.8, label="island finals")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{func_id}: Island PBIL final islands (best run #{idx+1})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        x = np.linspace(-6, 6, 200)
        y = np.linspace(-6, 6, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = bm.f3(X[i, j], Y[i, j])

        plt.figure()
        cs = plt.contourf(X, Y, Z, levels=50)
        plt.colorbar(cs, label="F3 (fitness)")

        finals_x1 = [p[0] for p in finals]
        finals_x2 = [p[1] for p in finals]
        plt.scatter(finals_x1, finals_x2, c="red", s=30, alpha=0.8, label="island finals")

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"F3: Island PBIL final islands (best run #{idx+1})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


# ---------- Main entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Island PBIL (with optional niching) on Deb & Goldberg benchmarks")
    parser.add_argument("--func", type=str, default="F1", choices=["F1", "F2", "F3"], help="Benchmark function")
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=["none", "base", "proportional"],
        help="Niching mode: 'none', 'base' (speciation), 'proportional'",
    )
    parser.add_argument("--num_islands", type=int, default=5, help="Number of islands")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent Island PBIL runs")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--total_evals", type=int, default=20000, help="Total evaluations per run")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per island per generation")
    parser.add_argument("--niching_interval", type=int, default=10, help="Niching interval (generations)")
    parser.add_argument("--sigma_niche", type=float, default=None, help="Niche distance threshold (optional override)")
    args = parser.parse_args()

    func_id = args.func.upper()
    num_runs = args.runs
    mode = args.mode
    num_islands = args.num_islands

    # Default sigma if not provided
    if args.sigma_niche is None:
        if func_id in ("F1", "F2"):
            sigma_niche = 0.1
        else:
            sigma_niche = 1.0
    else:
        sigma_niche = args.sigma_niche

    run_dir = prepare_run_directory("IslandPBIL", func_id)
    print(f"[IslandPBIL] Running {num_runs} runs on {func_id} with mode='{mode}', {num_islands} islands. Output in {run_dir}")

    results = run_island_pbil(
        func_id=func_id,
        mode=mode,
        num_islands=num_islands,
        total_evals=args.total_evals,
        num_samples=args.num_samples,
        alpha=0.1,
        pmutate_prob=0.02,
        mutation_shift=0.05,
        niching_interval=args.niching_interval,
        sigma_niche=sigma_niche,
        num_runs=num_runs,
        seed=args.seed,
    )

    save_island_pbil_results_to_files(results, func_id, mode, num_islands, run_dir)

    island_pbil_plot_spaghetti(results, func_id, run_dir / "best_fitness_spaghetti.png")
    island_pbil_plot_peak_representatives(results, func_id, run_dir / "island_peak_representatives.png")
    island_pbil_plot_peak_occupancy(results, func_id, run_dir / "island_peak_occupancy.png")
    island_pbil_plot_single_run_final(results, func_id, run_dir / "island_best_run_final_points.png")

    if func_id in ("F1", "F2"):
        island_pbil_plot_1d_function_with_points(results, func_id, run_dir / f"{func_id.lower()}_final_points.png")
    else:
        island_pbil_plot_himmelblau_with_points(results, run_dir / "himmelblau_final_points.png")

    print("[IslandPBIL] Done.")


if __name__ == "__main__":
    main()
    