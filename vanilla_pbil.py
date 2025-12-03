"""Simple population‑based incremental learning implementation."""
import numpy as np
import benchmarks as bm
import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from typing import List, Dict, Any

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


class PBIL:
    """Population‑based incremental learner.

    Uses a single probability vector to model binary chromosomes, updating
    it towards the best sampled individual each generation.  Optional
    mutation perturbs entries back towards 0.5 to maintain diversity.
    """

    def __init__(
        self,
        func_id: str,
        total_evals: int = 20000,
        num_samples: int = 50,
        alpha: float = 0.1,
        pmutate_prob: float = 0.02,
        mutation_shift: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.func_id = func_id.upper()
        self.total_evals = total_evals
        self.num_samples = num_samples
        self.alpha = alpha
        self.pmutate_prob = pmutate_prob
        self.mutation_shift = mutation_shift
        self.rng = rng if rng is not None else np.random.default_rng()
        # Probability vector initialised to 0.5 (uniform ignorance)
        self.p = np.full(30, 0.5, dtype=float)
        # Determine decoding and fitness functions
        if self.func_id in ('F1', 'F2'):
            self.decode = bm.bits_to_real_1d
        elif self.func_id == 'F3':
            self.decode = bm.bits_to_real_2d
        else:
            raise ValueError(f"Unknown function id: {func_id}")
        self.func = bm.get_function(func_id)

    def _sample_population(self) -> np.ndarray:
        """Return ``num_samples`` binary chromosomes drawn from ``p``."""
        return (self.rng.random((self.num_samples, 30)) < self.p).astype(np.int8)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Return an array of fitness values for the sampled population."""
        fitness = np.empty(self.num_samples, dtype=float)
        if self.func_id in ("F1", "F2"):
            for i, bits in enumerate(population):
                fitness[i] = self.func(self.decode(bits))
        else:
            for i, bits in enumerate(population):
                x1, x2 = self.decode(bits)
                fitness[i] = self.func(x1, x2)
        return fitness

    def _mutate_probability_vector(self) -> None:
        """Randomly perturb the probability vector towards 0.5."""
        mask = self.rng.random(self.p.shape) < self.pmutate_prob
        self.p[mask] = (1.0 - self.mutation_shift) * self.p[mask] + self.mutation_shift * 0.5

    def run(self) -> Dict[str, Any]:
        """Execute the PBIL optimizer and return its history and final result.

        Returns
        -------
        dict
            Contains keys:
            - 'best_fitness_per_gen': list of best fitness values per generation
            - 'best_x_per_gen': list of best decoded phenotypes per generation
            - 'final_p': final probability vector (np.ndarray of shape (30,))
            - 'final_sample_bits': one sampled bitstring from final p
            - 'final_sample_decoded': decoded phenotype of final sample
            - 'final_fitness': fitness of final sample
        """
        # Determine number of generations based on total evaluations
        n_generations = max(1, self.total_evals // self.num_samples)
        best_fitness_per_gen: List[float] = []
        best_x_per_gen: List[Any] = []
        for _ in range(n_generations):
            # Sample population
            pop = self._sample_population()
            # Evaluate
            fitness = self._evaluate_population(pop)
            # Identify best individual
            idx_best = int(np.argmax(fitness))
            best_bits = pop[idx_best]
            # Decode best
            best_decoded = self.decode(best_bits)
            best_fit = fitness[idx_best]
            best_fitness_per_gen.append(float(best_fit))
            best_x_per_gen.append(best_decoded)
            # Update probability vector toward best bits
            # p_j = (1 - alpha) * p_j + alpha * s_j (s_j is 0 or 1)
            self.p = (1.0 - self.alpha) * self.p + self.alpha * best_bits
            # Apply mutation to probability vector
            self._mutate_probability_vector()
        # Generate a final sample and decode it
        final_sample_bits = (self.rng.random(30) < self.p).astype(np.int8)
        final_decoded = self.decode(final_sample_bits)
        if self.func_id in ('F1', 'F2'):
            final_fit = self.func(final_decoded)
        else:
            final_fit = self.func(*final_decoded)  # type: ignore[arg-type]
        return {
            'best_fitness_per_gen': best_fitness_per_gen,
            'best_x_per_gen': best_x_per_gen,
            'final_p': self.p.copy(),
            'final_sample_bits': final_sample_bits,
            'final_sample_decoded': final_decoded,
            'final_fitness': float(final_fit),
        }


def run_pbil(
    func_id: str,
    num_runs: int = 1,
    total_evals: int = 20000,
    num_samples: int = 50,
    alpha: float = 0.1,
    pmutate_prob: float = 0.02,
    mutation_shift: float = 0.05,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    """Return a list of PBIL run dictionaries across ``num_runs`` experiments."""
    results: List[Dict[str, Any]] = []
    for i in range(num_runs):
        rng = np.random.default_rng(None if seed is None else seed + i)
        pbil = PBIL(
            func_id=func_id,
            total_evals=total_evals,
            num_samples=num_samples,
            alpha=alpha,
            pmutate_prob=pmutate_prob,
            mutation_shift=mutation_shift,
            rng=rng,
        )
        results.append(pbil.run())
    return results

def prepare_run_directory(model_name: str, func_id: str) -> Path:
    """Create and return a fresh ``run_n`` directory for a given model/function."""
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


# ---------- Plotting helpers ----------

def pbil_plot_spaghetti_best_fitness(results, func_id: str, out_path: Path) -> None:
    """Spaghetti plot of best fitness per generation across PBIL runs."""
    plt.figure()
    for res in results:
        y = res["best_fitness_per_gen"]
        x = list(range(len(y)))
        plt.plot(x, y, alpha=0.4)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(f"PBIL best fitness vs generations ({func_id})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pbil_plot_1d_function_with_points(results, func_id: str, out_path: Path) -> None:
    """Plot F1/F2 and overlay final PBIL solutions."""
    if func_id.upper() == "F1":
        f = bm.f1
    elif func_id.upper() == "F2":
        f = bm.f2
    else:
        raise ValueError("pbil_plot_1d_function_with_points only valid for F1/F2")

    xs = np.linspace(0.0, 1.0, 1000)
    ys = [f(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label=func_id)

    finals = [res["final_sample_decoded"] for res in results]
    finals = [float(x) for x in finals]
    fvals = [f(x) for x in finals]

    plt.scatter(finals, fvals, c="red", s=40, alpha=0.8, label="PBIL final")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{func_id} with final PBIL solutions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def pbil_plot_himmelblau_with_points(results, out_path: Path) -> None:
    """Contour of Himmelblau + final PBIL solutions."""
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
        x1, x2 = res["final_sample_decoded"]
        finals_x1.append(x1)
        finals_x2.append(x2)

    plt.scatter(finals_x1, finals_x2, c="red", s=40, alpha=0.8, label="PBIL final")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("PBIL on Himmelblau (F3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# ---------- Saving CSV / JSON ----------

def save_pbil_results_to_files(results, func_id: str, run_dir: Path) -> None:
    """Save per-run PBIL curves and summaries."""
    for i, res in enumerate(results):
        idx = i + 1

        # CSV: best fitness per generation
        csv_path = run_dir / f"pbil_run{idx}_best_fitness.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness"])
            for gen, fit in enumerate(res["best_fitness_per_gen"]):
                writer.writerow([gen, fit])

        # JSON summary
        json_path = run_dir / f"pbil_run{idx}_summary.json"
        summary = {
            "func_id": func_id,
            "run_index": idx,
            "final_best_fitness": float(res["final_fitness"]),
            "final_decoded": res["final_sample_decoded"],
        }
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2)

# ----------------- PBIL peak analysis helpers -----------------


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


def _pbil_collect_all_points(results, func_id: str):
    """Collect final PBIL phenotypes across runs."""
    if func_id in ("F1", "F2"):
        return [float(res["final_sample_decoded"]) for res in results]
    else:
        return [tuple(res["final_sample_decoded"]) for res in results]

def pbil_plot_peak_representatives(results, func_id: str, out_path: Path) -> None:
    """
    For vanilla PBIL: best final solution per peak (aggregated over runs).
    """
    func_id = func_id.upper()
    pts = _pbil_collect_all_points(results, func_id)

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
        plt.title(f"{func_id}: PBIL niche representatives")
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
        plt.title("F3: PBIL niche representatives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def pbil_plot_peak_occupancy(results, func_id: str, out_path: Path) -> None:
    """
    For vanilla PBIL: bar plot of how many runs converged to each peak.
    """
    func_id = func_id.upper()
    pts = _pbil_collect_all_points(results, func_id)

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
    plt.ylabel("Number of runs")
    plt.title(f"{func_id}: PBIL peak occupancy (final solution per run)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _pbil_best_run_index(results):
    """
    Choose best PBIL run by final_fitness.
    """
    best_idx = 0
    best_val = -np.inf
    for i, res in enumerate(results):
        val = float(res["final_fitness"])
        if val > best_val:
            best_val = val
            best_idx = i
    return best_idx


def pbil_plot_single_run_final(results, func_id: str, out_path: Path) -> None:
    """
    Plot final PBIL solution from the single best run.
    """
    func_id = func_id.upper()
    idx = _pbil_best_run_index(results)
    res = results[idx]
    final = res["final_sample_decoded"]

    if func_id == "F1":
        f = bm.f1
    elif func_id == "F2":
        f = bm.f2
    else:
        def f_pair(p):
            return bm.f3(p[0], p[1])

    if func_id in ("F1", "F2"):
        x = float(final)
        xs = np.linspace(0.0, 1.0, 1000)
        ys = [f(v) for v in xs]

        plt.figure()
        plt.plot(xs, ys, label=func_id)
        plt.scatter([x], [f(x)], c="red", s=60, label="PBIL final")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{func_id}: PBIL final solution (best run #{idx+1})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
    else:
        x1, x2 = final
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

        plt.scatter([x1], [x2], c="red", s=60, label="PBIL final")
        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.title(f"F3: PBIL final solution (best run #{idx+1})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


# ---------- Main entry point ----------

def main():
    parser = argparse.ArgumentParser(description="Vanilla PBIL on Deb & Goldberg benchmarks")
    parser.add_argument("--func", type=str, default="F1", choices=["F1", "F2", "F3"], help="Benchmark function")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent PBIL runs")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--total_evals", type=int, default=20000, help="Total evaluations per run")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per generation")
    args = parser.parse_args()

    func_id = args.func.upper()
    num_runs = args.runs

    run_dir = prepare_run_directory("PBIL", func_id)
    print(f"[PBIL] Running {num_runs} runs on {func_id}. Output in {run_dir}")

    results = run_pbil(
        func_id=func_id,
        num_runs=num_runs,
        total_evals=args.total_evals,
        num_samples=args.num_samples,
        alpha=0.1,
        pmutate_prob=0.02,
        mutation_shift=0.05,
        seed=args.seed,
    )

    save_pbil_results_to_files(results, func_id, run_dir)

    # visuals
    pbil_plot_spaghetti_best_fitness(results, func_id, run_dir / "best_fitness_spaghetti.png")
    pbil_plot_peak_representatives(results, func_id, run_dir / "pbil_peak_representatives.png")
    pbil_plot_peak_occupancy(results, func_id, run_dir / "pbil_peak_occupancy.png")
    pbil_plot_single_run_final(results, func_id, run_dir / "pbil_best_run_final_points.png")

    # Function + final points
    if func_id in ("F1", "F2"):
        pbil_plot_1d_function_with_points(results, func_id, run_dir / f"{func_id.lower()}_final_points.png")
    else:
        pbil_plot_himmelblau_with_points(results, run_dir / "himmelblau_final_points.png")

    print("[PBIL] Done.")


if __name__ == "__main__":
    main()