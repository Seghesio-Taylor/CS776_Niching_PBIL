import numpy as np
from typing import List, Tuple, Dict, Callable, Any
import json
import csv
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import benchmarks as bm


PEAK_POS_1D = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
HIMMELBLAU_MINIMA = np.array([
    [ 3.0,       2.0      ],
    [-2.805118,  3.131312 ],
    [-3.779310, -3.283186 ],
    [ 3.584428, -1.848126 ],
], dtype=float)


class GeneticAlgorithm:
    def __init__(
        self,
        func_id: str,
        population_size: int = 100,
        generations: int = 200,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.001,
        sigma_mating: float | None = None,
        tournament_size: int = 2,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.func_id = func_id.upper()
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        if sigma_mating is None:
            if self.func_id in ('F1', 'F2'):
                sigma_mating = 0.05
            else:
                sigma_mating = 1.0
        self.sigma_mating = sigma_mating
        self.tournament_size = tournament_size
        self.rng = rng if rng is not None else np.random.default_rng()
        if self.func_id in ('F1', 'F2'):
            self.decode = bm.bits_to_real_1d
            self.distance = bm.distance_1d
        elif self.func_id == 'F3':
            self.decode = bm.bits_to_real_2d
            self.distance = bm.distance_2d
        else:
            raise ValueError(f"Unknown function id: {func_id}")
        self.func = bm.get_function(func_id)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        fitness = np.empty(self.population_size, dtype=float)
        if self.func_id in ('F1', 'F2'):
            for i in range(self.population_size):
                x = self.decode(population[i])  # float
                fitness[i] = self.func(x)
        else:  # F3
            for i in range(self.population_size):
                x1, x2 = self.decode(population[i])
                fitness[i] = self.func(x1, x2)
        return fitness

    def _tournament_select(self, fitness: np.ndarray) -> int:
        contenders = self.rng.integers(0, self.population_size, size=self.tournament_size)
        best_idx = contenders[0]
        best_fit = fitness[best_idx]
        for idx in contenders[1:]:
            fit = fitness[idx]
            if fit > best_fit:
                best_idx = idx
                best_fit = fit
        return best_idx

    def _restricted_tournament_replacement(
        self,
        offspring: np.ndarray,
        population: np.ndarray,
        decoded_phenotypes: list[Any],
        window_size: int = 5,
    ) -> None:

        pop_size = population.shape[0]
        if pop_size == 0:
            return

        w = min(window_size, pop_size)
        indices = self.rng.choice(pop_size, size=w, replace=False)
        off_pheno = self.decode(offspring)
        distances = []
        for idx in indices:
            cand_pheno = decoded_phenotypes[idx]
            distances.append(self.distance(off_pheno, cand_pheno))

        replace_idx = int(indices[int(np.argmin(distances))])
        population[replace_idx] = offspring.copy()
        decoded_phenotypes[replace_idx] = off_pheno



    def _select_parent2(self, parent1_idx: int, population: np.ndarray, fitness: np.ndarray,
                        decoded_phenotypes: List[Any]) -> int:
        parent_pheno = decoded_phenotypes[parent1_idx]
        dists = []
        for pheno in decoded_phenotypes:
            dists.append(self.distance(parent_pheno, pheno))
        dists = np.array(dists)
        neighbours = [i for i in range(self.population_size)
                      if i != parent1_idx and dists[i] < self.sigma_mating]
        if neighbours:
            k = min(self.tournament_size, len(neighbours))
            contenders = self.rng.choice(neighbours, size=k, replace=False)
            best_idx = contenders[0]
            best_fit = fitness[best_idx]
            for idx in contenders[1:]:
                if fitness[idx] > best_fit:
                    best_idx = idx
                    best_fit = fitness[idx]
            return best_idx
        else:
            idxs = [i for i in range(self.population_size) if i != parent1_idx]
            closest_idx = idxs[0]
            min_dist = dists[closest_idx]
            for idx in idxs[1:]:
                if dists[idx] < min_dist:
                    closest_idx = idx
                    min_dist = dists[idx]
            return closest_idx

    def _crossover_mutate(self, parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
        child = parent1.copy()
        if self.rng.random() < self.crossover_prob:
            cp = self.rng.integers(1, parent1.shape[0])
            child[:cp] = parent1[:cp]
            child[cp:] = parent2[cp:]
        mutation_mask = self.rng.random(child.shape) < self.mutation_prob
        child = np.bitwise_xor(child, mutation_mask.astype(np.int8))
        return child

    def run(self) -> Dict[str, Any]:
        population = self.rng.integers(0, 2, size=(self.population_size, 30), dtype=np.int8)
        best_fitness_per_gen: List[float] = []
        best_x_per_gen: List[Any] = []

        for gen in range(self.generations):
            decoded = [self.decode(ind) for ind in population]
            fitness = self._evaluate_population(population)
            best_idx = int(np.argmax(fitness))
            best_fitness_per_gen.append(float(fitness[best_idx]))
            best_x_per_gen.append(decoded[best_idx])
            new_population = population.copy()
            new_population[0] = population[best_idx].copy()
            decoded[0] = decoded[best_idx]
            num_offspring = self.population_size - 1

            for _ in range(num_offspring):
                p1_idx = self._tournament_select(fitness)
                p2_idx = self._select_parent2(p1_idx, population, fitness, decoded)
                parent1 = population[p1_idx]
                parent2 = population[p2_idx]
                child = self._crossover_mutate(parent1, parent2)
                self._restricted_tournament_replacement(
                    child,
                    new_population,
                    decoded,
                    window_size=5,
                )

            population = new_population

        decoded = [self.decode(ind) for ind in population]
        fitness = self._evaluate_population(population)
        return {
            'best_fitness_per_gen': best_fitness_per_gen,
            'best_x_per_gen': best_x_per_gen,
            'final_population': population.copy(),
            'final_decoded': decoded,
            'final_fitness': fitness.tolist(),
        }


def run_ga(
    func_id: str,
    num_runs: int = 1,
    population_size: int = 100,
    generations: int = 200,
    crossover_prob: float = 0.9,
    mutation_prob: float = 0.001,
    sigma_mating: float | None = None,
    tournament_size: int = 2,
    seed: int | None = None,
) -> List[Dict[str, Any]]:

    results: List[Dict[str, Any]] = []
    for i in range(num_runs):
        rng = np.random.default_rng(None if seed is None else seed + i)
        ga = GeneticAlgorithm(
            func_id=func_id,
            population_size=population_size,
            generations=generations,
            crossover_prob=crossover_prob,
            mutation_prob=mutation_prob,
            sigma_mating=sigma_mating,
            tournament_size=tournament_size,
            rng=rng,
        )
        results.append(ga.run())
    return results

def prepare_run_directory(model_name: str, func_id: str) -> Path:
    base = Path(f"run_{model_name}")
    base.mkdir(exist_ok=True)
    func_dir = base / func_id
    func_dir.mkdir(exist_ok=True)
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

def plot_spaghetti_best_fitness(results: List[Dict[str, Any]], func_id: str, out_path: Path) -> None:
    plt.figure()
    for r, res in enumerate(results):
        y = res["best_fitness_per_gen"]
        x = list(range(len(y)))
        plt.plot(x, y, alpha=0.4)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.title(f"GA best fitness vs generations ({func_id})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_1d_function_with_points(func_id: str, results: List[Dict[str, Any]], out_path: Path) -> None:
    if func_id.upper() == "F1":
        f = bm.f1
    elif func_id.upper() == "F2":
        f = bm.f2
    else:
        raise ValueError("plot_1d_function_with_points only valid for F1/F2")

    xs = np.linspace(0.0, 1.0, 1000)
    ys = [f(x) for x in xs]

    plt.figure()
    plt.plot(xs, ys, label=func_id)
    finals = []
    for res in results:
        finals.extend(res["final_decoded"])
    finals = [float(x) for x in finals]
    fvals = [f(x) for x in finals]
    plt.scatter(finals, fvals, c="red", s=20, alpha=0.7, label="final individuals")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(f"{func_id} with final GA individuals")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_himmelblau_with_points(results: List[Dict[str, Any]], out_path: Path) -> None:
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
        for (x1, x2) in res["final_decoded"]:
            finals_x1.append(x1)
            finals_x2.append(x2)
    plt.scatter(finals_x1, finals_x2, c="red", s=20, alpha=0.7, label="final individuals")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("GA on Himmelblau (F3)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_ga_results_to_files(
    results: List[Dict[str, Any]],
    func_id: str,
    run_dir: Path,
) -> None:
    for i, res in enumerate(results):
        csv_path = run_dir / f"ga_run{i+1}_best_fitness.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "best_fitness"])
            for gen, fit in enumerate(res["best_fitness_per_gen"]):
                writer.writerow([gen, fit])

        json_path = run_dir / f"ga_run{i+1}_summary.json"
        summary = {
            "func_id": func_id,
            "run_index": i + 1,
            "final_best_fitness": max(res["best_fitness_per_gen"]),
            "final_population_size": len(res["final_decoded"]),
        }
        with json_path.open("w") as f:
            json.dump(summary, f, indent=2)

def _assign_peak_1d(x: float, threshold: float = 0.1) -> int | None:
    diffs = np.abs(PEAK_POS_1D - x)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None


def _assign_peak_2d(x1: float, x2: float, threshold: float = 1.0) -> int | None:
    point = np.array([x1, x2])
    diffs = np.linalg.norm(HIMMELBLAU_MINIMA - point, axis=1)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None


def _ga_collect_all_points(results, func_id: str):
    pts = []
    if func_id in ("F1", "F2"):
        for res in results:
            pts.extend(res["final_decoded"])
    else:
        for res in results:
            pts.extend(res["final_decoded"])
    return pts

def ga_plot_peak_representatives(results, func_id: str, out_path: Path) -> None:
    func_id = func_id.upper()
    pts = _ga_collect_all_points(results, func_id)
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
            x = float(x)
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
        plt.title(f"{func_id}: GA niche representatives")
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
        plt.title("F3: GA niche representatives")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def ga_plot_peak_occupancy(results, func_id: str, out_path: Path) -> None:
    func_id = func_id.upper()
    pts = _ga_collect_all_points(results, func_id)

    if func_id in ("F1", "F2"):
        num_peaks = len(PEAK_POS_1D)
        counts = np.zeros(num_peaks, dtype=int)
        for x in pts:
            peak = _assign_peak_1d(float(x))
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
    plt.ylabel("Count")
    plt.title(f"{func_id}: GA peak occupancy (final population, all runs)")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _ga_best_run_index(results):
    best_idx = 0
    best_val = -np.inf
    for i, res in enumerate(results):
        last_fit = float(res["best_fitness_per_gen"][-1])
        if last_fit > best_val:
            best_val = last_fit
            best_idx = i
    return best_idx


def ga_plot_single_run_final(results, func_id: str, out_path: Path) -> None:
    func_id = func_id.upper()
    idx = _ga_best_run_index(results)
    res = results[idx]
    finals = res["final_decoded"]

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
        plt.scatter(finals, fvals, c="red", s=25, alpha=0.8, label="final individuals")

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title(f"{func_id}: GA final population (best run #{idx+1})")
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
        plt.scatter(finals_x1, finals_x2, c="red", s=25, alpha=0.8, label="final individuals")


def main():
    parser = argparse.ArgumentParser(description="GA with phenotypic niching on Deb & Goldberg benchmarks")
    parser.add_argument("--func", type=str, default="F1", choices=["F1", "F2", "F3"], help="Benchmark function")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent GA runs")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    args = parser.parse_args()

    func_id = args.func.upper()
    num_runs = args.runs

    run_dir = prepare_run_directory("GA", func_id)
    print(f"[GA] Running {num_runs} runs on {func_id}. Output in {run_dir}")

    results = run_ga(
        func_id=func_id,
        num_runs=num_runs,
        population_size=100,
        generations=200,
        crossover_prob=0.9,
        mutation_prob=0.001,
        sigma_mating=None,
        tournament_size=2,
        seed=args.seed,
    )

    save_ga_results_to_files(results, func_id, run_dir)
    plot_spaghetti_best_fitness(results, func_id, run_dir / "best_fitness_spaghetti.png")
    ga_plot_peak_representatives(results, func_id, run_dir / "ga_peak_representatives.png")
    ga_plot_peak_occupancy(results, func_id, run_dir / "ga_peak_occupancy.png")
    ga_plot_single_run_final(results, func_id, run_dir / "ga_best_run_final_points.png")

    if func_id in ("F1", "F2"):
        plot_1d_function_with_points(func_id, results, run_dir / f"{func_id.lower()}_final_points.png")
    else:
        plot_himmelblau_with_points(results, run_dir / "himmelblau_final_points.png")

    print("[GA] Done.")


if __name__ == "__main__":
    main()
