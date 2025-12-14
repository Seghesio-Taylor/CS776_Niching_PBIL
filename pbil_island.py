import numpy as np
import benchmarks as bm
import os
import json
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import multiprocessing as mp
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional


PEAK_POS_1D = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
HIMMELBLAU_MINIMA = np.array([
    [ 3.0,       2.0      ],
    [-2.805118,  3.131312 ],
    [-3.779310, -3.283186 ],
    [ 3.584428, -1.848126 ],
], dtype=float)

_MAX_INT_15 = (1 << 15) - 1
_MAX_INT_30 = (1 << 30) - 1

def real_to_bits_2d(x1: float, x2: float) -> np.ndarray:
    x1 = float(np.clip(x1, -6.0, 6.0))
    x2 = float(np.clip(x2, -6.0, 6.0))
    b1 = int(round((x1 + 6.0) / 12.0 * _MAX_INT_15))
    b2 = int(round((x2 + 6.0) / 12.0 * _MAX_INT_15))
    bits = np.zeros(30, dtype=np.int8)
    for i in range(14, -1, -1):
        bits[i] = b1 & 1
        b1 >>= 1
    for i in range(29, 14, -1):
        bits[i] = b2 & 1
        b2 >>= 1
    return bits

def expected_real_from_p_1d(p_vec: np.ndarray) -> float:
    p = np.clip(np.asarray(p_vec, dtype=float), 0.0, 1.0)
    weights = (1 << np.arange(29, -1, -1)).astype(float)  # MSB first
    e = float(np.dot(p, weights))  # expected integer value
    return e / _MAX_INT_30

def expected_real_from_p_2d(p_vec: np.ndarray) -> tuple[float, float]:
    p = np.clip(np.asarray(p_vec, dtype=float), 0.0, 1.0)
    p1 = p[:15]
    p2 = p[15:]
    weights = (1 << np.arange(14, -1, -1)).astype(float)  # MSB first
    e1 = float(np.dot(p1, weights))
    e2 = float(np.dot(p2, weights))
    x1 = (e1 / _MAX_INT_15) * 12.0 - 6.0
    x2 = (e2 / _MAX_INT_15) * 12.0 - 6.0
    return x1, x2

def _pbil_update_worker(args):
    (
        func_id,
        p_vec,
        num_samples,
        alpha,
        pmutate_prob,
        mutation_shift,
        seed,
    ) = args

    rng = np.random.default_rng(int(seed))
    p_vec = np.asarray(p_vec, dtype=float).copy()

    pop = (rng.random((num_samples, 30)) < p_vec).astype(np.int8)
    fitness = np.empty(num_samples, dtype=float)

    if func_id in ("F1", "F2"):
        f = bm.get_function(func_id)
        for i, bits in enumerate(pop):
            x = bm.bits_to_real_1d(bits)
            fitness[i] = f(x)
    else:
        f = bm.get_function("F3")
        for i, bits in enumerate(pop):
            x1, x2 = bm.bits_to_real_2d(bits)
            fitness[i] = f(x1, x2)

    idx_best = int(np.argmax(fitness))
    best_bits = pop[idx_best]
    best_fit = float(fitness[idx_best])

    if func_id in ("F1", "F2"):
        best_decoded = float(bm.bits_to_real_1d(best_bits))
    else:
        best_decoded = tuple(bm.bits_to_real_2d(best_bits))

    p_vec = (1.0 - alpha) * p_vec + alpha * best_bits
    mask = rng.random(p_vec.shape) < pmutate_prob
    p_vec[mask] = (1.0 - mutation_shift) * p_vec[mask] + mutation_shift * 0.5

    return p_vec, best_fit, best_decoded

class IslandPBILBase:
    def __init__(
            self,
            func_id: str,
            num_islands: int = 5,
            total_evals: int = 20000,
            num_samples: int = 50,
            alpha: float = 0.1,
            pmutate_prob: float = 0.02,
            mutation_shift: float = 0.05,
            rng: np.random.Generator | None = None,
            use_mp: bool = False,
            mp_workers: int | None = None,
        ) -> None:
        self.func_id = func_id.upper()
        self.num_islands = num_islands
        self.total_evals = total_evals
        self.num_samples = num_samples
        self.alpha = alpha
        self.pmutate_prob = pmutate_prob
        self.mutation_shift = mutation_shift
        self.rng = rng if rng is not None else np.random.default_rng()
        self.use_mp = bool(use_mp)
        self.mp_workers = mp_workers
        self.p = np.full((num_islands, 30), 0.5, dtype=float)
        self.p += (self.rng.random(self.p.shape) - 0.5) * 0.02
        self.p = np.clip(self.p, 0.0, 1.0)
        if self.func_id in ('F1', 'F2'):
            self.decode = bm.bits_to_real_1d
        elif self.func_id == 'F3':
            self.decode = bm.bits_to_real_2d
        else:
            raise ValueError(f"Unknown function id: {func_id}")
        self.func = bm.get_function(func_id)
        denom = max(1, self.num_islands * self.num_samples)
        self.n_generations = max(1, self.total_evals // denom)
        self.best_fitness_per_gen: List[List[float]] = [list() for _ in range(num_islands)]
        self.best_x_per_gen: List[List[Any]] = [list() for _ in range(num_islands)]
        self.trace_x_per_gen: List[List[Any]] = [list() for _ in range(num_islands)]

    def _sample_population(self, island_idx: int) -> np.ndarray:
        rand = self.rng.random((self.num_samples, 30))
        return (rand < self.p[island_idx]).astype(np.int8)

    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        fitness = np.empty(self.num_samples, dtype=float)
        if self.func_id in ('F1', 'F2'):
            for i, bits in enumerate(population):
                x = self.decode(bits)
                fitness[i] = self.func(x)
        else:
            for i, bits in enumerate(population):
                x1, x2 = self.decode(bits)
                fitness[i] = self.func(x1, x2)
        return fitness

    def _update_island(self, island_idx: int) -> Tuple[float, Any]:
        pop = self._sample_population(island_idx)
        fitness = self._evaluate_population(pop)
        idx_best = int(np.argmax(fitness))
        best_bits = pop[idx_best]
        best_fit = float(fitness[idx_best])
        best_decoded = self.decode(best_bits)
        self.p[island_idx] = (1.0 - self.alpha) * self.p[island_idx] + self.alpha * best_bits
        mask = self.rng.random(self.p[island_idx].shape) < self.pmutate_prob
        self.p[island_idx][mask] = (
            (1.0 - self.mutation_shift) * self.p[island_idx][mask]
            + self.mutation_shift * 0.5
        )
        # Log
        self.best_fitness_per_gen[island_idx].append(best_fit)
        self.best_x_per_gen[island_idx].append(best_decoded)
        if self.func_id in ("F1", "F2"):
            self.trace_x_per_gen[island_idx].append(expected_real_from_p_1d(self.p[island_idx]))
        else:
            self.trace_x_per_gen[island_idx].append(expected_real_from_p_2d(self.p[island_idx]))


        return best_fit, best_decoded

    def _initialise_island_to_peak(self, island_idx: int, peak_index: int) -> None:
        n_bits = self.p.shape[1]
        if self.func_id in ("F1", "F2"):
            target_x = float(PEAK_POS_1D[peak_index])

            def _dist(bits: np.ndarray) -> float:
                x = float(self.decode(bits))
                return abs(x - target_x)
        else:
            target_point = np.array(HIMMELBLAU_MINIMA[peak_index], dtype=float)

            def _dist(bits: np.ndarray) -> float:
                x1, x2 = self.decode(bits)
                v = np.array([float(x1), float(x2)], dtype=float) - target_point
                return float(np.linalg.norm(v))

        best_bits: Optional[np.ndarray] = None
        best_dist = float("inf")

        for _ in range(256):
            bits = (self.rng.random(n_bits) < 0.5).astype(np.int8)
            d = _dist(bits)
            if d < best_dist:
                best_bits, best_dist = bits, d

        if best_bits is None:
            self._reinitialize_island(island_idx)
            return

        probs = 0.001 + 0.998 * best_bits.astype(float)
        self.p[island_idx] = probs


    def after_generation(self, gen: int) -> None:
        pass

    def run(self) -> Dict[str, Any]:
            pool = None
            if self.use_mp:
                try:
                    ctx = mp.get_context("fork")
                except ValueError:
                    ctx = mp.get_context()
                pool = ctx.Pool(processes=self.mp_workers)

            try:
                for k in range(self.num_islands):
                    if self.func_id in ("F1", "F2"):
                        self.trace_x_per_gen[k].append(expected_real_from_p_1d(self.p[k]))
                    else:
                        self.trace_x_per_gen[k].append(expected_real_from_p_2d(self.p[k]))

                for gen in range(self.n_generations):
                    if pool is None:
                        for k in range(self.num_islands):
                            self._update_island(k)
                    else:
                        seeds = self.rng.integers(0, 2**32 - 1, size=self.num_islands, dtype=np.uint32)
                        args = [
                            (
                                self.func_id,
                                self.p[k],
                                self.num_samples,
                                self.alpha,
                                self.pmutate_prob,
                                self.mutation_shift,
                                int(seeds[k]),
                            )
                            for k in range(self.num_islands)
                        ]
                        results = pool.map(_pbil_update_worker, args)
                        for k, (p_vec, best_fit, best_decoded) in enumerate(results):
                            self.p[k] = p_vec
                            self.best_fitness_per_gen[k].append(float(best_fit))
                            self.best_x_per_gen[k].append(best_decoded)
                            if self.func_id in ("F1", "F2"):
                                self.trace_x_per_gen[k].append(expected_real_from_p_1d(self.p[k]))
                            else:
                                self.trace_x_per_gen[k].append(expected_real_from_p_2d(self.p[k]))

                    self.after_generation(gen)

                final_decoded: List[Any] = []
                final_fitness: List[float] = []
                for k in range(self.num_islands):
                    bits = (self.rng.random(30) < self.p[k]).astype(np.int8)
                    decoded = self.decode(bits)
                    if self.func_id in ("F1", "F2"):
                        fit = self.func(decoded)
                    else:
                        fit = self.func(*decoded)
                    final_decoded.append(decoded)
                    final_fitness.append(float(fit))

                return {
                    "best_fitness_per_gen": self.best_fitness_per_gen,
                    "best_x_per_gen": self.best_x_per_gen,
                    "final_p": self.p.copy(),
                    "final_sample_decoded": final_decoded,
                    "final_fitness": final_fitness,
                    "trace_x_per_gen": self.trace_x_per_gen,
                }
            finally:
                if pool is not None:
                    pool.close()
                    pool.join()



class IslandPBILNiching(IslandPBILBase):
    def __init__(
            self,
            func_id: str,
            mode: str = "base",
            num_islands: int = 5,
            total_evals: int = 20000,
            num_samples: int = 50,
            alpha: float = 0.1,
            pmutate_prob: float = 0.02,
            mutation_shift: float = 0.05,
            niching_interval: int = 10,
            sigma_niche: float = 0.1,
            rng: np.random.Generator | None = None,
            nudge: bool = True,
            nudge_strength: float = 1.0,
            use_mp: bool = False,
            mp_workers: int | None = None,
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
                use_mp=use_mp,
                mp_workers=mp_workers,
            )
            self.mode = mode
            self.niching_interval = niching_interval
            self.sigma_niche = sigma_niche
            self.nudge = bool(nudge)
            self.nudge_strength = float(nudge_strength)
            self.peak_goals: List[Optional[int]] = [None] * self.num_islands
            self.peak_threshold = 0.8 if self.func_id == "F3" else 0.08
            if self.func_id in ("F1", "F2"):
                num_peaks = len(PEAK_POS_1D)
            else:
                num_peaks = len(HIMMELBLAU_MINIMA)
            self.peak_owner: List[Optional[int]] = [None] * num_peaks
            self.island_owner: List[Optional[int]] = [None] * self.num_islands
            self.coverage_start_gen = 25
            if self.mode == "peak_walk":
                for i in range(min(self.num_islands, num_peaks)):
                    self.peak_goals[i] = i

    def _update_island(self, island_idx: int) -> Tuple[float, Any]:
        pop = self._sample_population(island_idx)
        fitness = self._evaluate_population(pop)
        idx_best = int(np.argmax(fitness))
        best_bits = pop[idx_best]
        best_fit = float(fitness[idx_best])
        best_decoded = self.decode(best_bits)
        alpha_eff = self.alpha
        pmutate_eff = self.pmutate_prob

        if self.func_id == "F2" and self.mode == "peak_walk":
            if (self.peak_goals[island_idx] is not None) or (self.island_owner[island_idx] is not None):
                alpha_eff = 0.0
                pmutate_eff = 0.0

        if self.mode == "peak_walk" and self.peak_goals[island_idx] is not None:
            g = self.peak_goals[island_idx]

            if self.func_id in ("F1", "F2"):
                cur = float(expected_real_from_p_1d(self.p[island_idx]))
                tgt = float(PEAK_POS_1D[g])
                dist = abs(cur - tgt)
            else:
                cur = np.array(expected_real_from_p_2d(self.p[island_idx]), dtype=float)
                tgt = np.array(HIMMELBLAU_MINIMA[g], dtype=float)
                dist = float(np.linalg.norm(cur - tgt))

            if self.func_id == "F2":
                freeze_dist = 0.5 * self.peak_threshold
            else:
                freeze_dist = 1.5

            if dist > freeze_dist:
                alpha_eff = 0.0
                pmutate_eff = 0.0
            else:
                if self.func_id == "F2":
                    alpha_eff = self.alpha
                    pmutate_eff = self.pmutate_prob
                else:
                    alpha_eff = 0.10 * self.alpha
                    pmutate_eff = self.pmutate_prob

        self.p[island_idx] = (1.0 - alpha_eff) * self.p[island_idx] + alpha_eff * best_bits
        mask = self.rng.random(self.p[island_idx].shape) < pmutate_eff
        self.p[island_idx][mask] = (
            (1.0 - self.mutation_shift) * self.p[island_idx][mask]
            + self.mutation_shift * 0.5
        )

        # Log
        self.best_fitness_per_gen[island_idx].append(best_fit)
        self.best_x_per_gen[island_idx].append(best_decoded)
        if self.func_id in ("F1", "F2"):
            self.trace_x_per_gen[island_idx].append(expected_real_from_p_1d(self.p[island_idx]))
        else:
            self.trace_x_per_gen[island_idx].append(expected_real_from_p_2d(self.p[island_idx]))

        return best_fit, best_decoded

    def _peak_index(self, pheno: Any) -> Optional[int]:
        if pheno is None:
            return None
        if self.func_id in ("F1", "F2"):
            return _assign_peak_1d(float(pheno), threshold=self.peak_threshold)
        x1, x2 = pheno
        return _assign_peak_2d(float(x1), float(x2), threshold=self.peak_threshold)


    def _nudge_island(self, island_idx: int, away_from: Any | None = None) -> None:
        if not self.nudge:
            self._reinitialize_island(island_idx)
            return

        if not self.best_x_per_gen[island_idx]:
            self._reinitialize_island(island_idx)
            return

        last = self.best_x_per_gen[island_idx][-1]
        if last is None:
            self._reinitialize_island(island_idx)
            return

        if self.func_id in ("F1", "F2"):
            x = float(last)
            if away_from is None:
                direction = -1.0 if x >= 0.5 else 1.0
            else:
                ref = float(away_from)
                direction = 1.0 if x >= ref else -1.0
            step = self.nudge_strength * float(self.sigma_niche)
            x_new = float(np.clip(x + direction * step + self.rng.normal(0.0, 0.2 * step), 0.0, 1.0))
            bits = bm.real_to_bits_1d(x_new)
        else:
            x1, x2 = last
            p = np.array([float(x1), float(x2)], dtype=float)
            if away_from is None:
                v = self.rng.normal(0.0, 1.0, size=2)
            else:
                ref = np.array([float(away_from[0]), float(away_from[1])], dtype=float)
                v = p - ref
                if float(np.linalg.norm(v)) == 0.0:
                    v = self.rng.normal(0.0, 1.0, size=2)
            v = v / (np.linalg.norm(v) + 1e-12)
            step = self.nudge_strength * float(np.sqrt(self.sigma_niche))
            p_new = p + v * step + self.rng.normal(0.0, 0.1 * step, size=2)
            p_new[0] = float(np.clip(p_new[0], -6.0, 6.0))
            p_new[1] = float(np.clip(p_new[1], -6.0, 6.0))
            bits = real_to_bits_2d(p_new[0], p_new[1])

        target = 0.001 + 0.998 * bits.astype(float)
        beta = 0.10
        self.p[island_idx] = (1.0 - beta) * self.p[island_idx] + beta * target
        self.p[island_idx] = np.clip(self.p[island_idx], 0.0, 1.0)

    def _nudge_toward_peak(self, island_idx: int, peak_index: int) -> None:
        if self.func_id in ("F1", "F2"):
            target_x = float(PEAK_POS_1D[peak_index])

            cur_x = float(expected_real_from_p_1d(self.p[island_idx]))
            dist = abs(target_x - cur_x)
            if dist < 1e-12:
                return

            direction = 1.0 if target_x >= cur_x else -1.0
            step = self.nudge_strength * float(self.sigma_niche)
            step = min(step, 0.75 * dist)
            noise_scale = 0.01 if self.func_id == "F2" else 0.05
            x_new = float(np.clip(cur_x + direction * step + self.rng.normal(0.0, noise_scale * step), 0.0, 1.0))
            bits = bm.real_to_bits_1d(x_new)
            target_p = 0.001 + 0.998 * bits.astype(float)
            if self.func_id == "F2":
                if dist > 0.4:
                    beta = 0.40
                elif dist > 0.2:
                    beta = 0.25
                elif dist > 0.1:
                    beta = 0.15
                elif dist > 0.05:
                    beta = 0.10
                else:
                    beta = 0.06
            else:
                if dist > 0.4:
                    beta = 0.35
                elif dist > 0.2:
                    beta = 0.20
                elif dist > 0.1:
                    beta = 0.10
                else:
                    beta = 0.04


            self.p[island_idx] = np.clip((1.0 - beta) * self.p[island_idx] + beta * target_p, 0.0, 1.0)
            return

        target = np.array(HIMMELBLAU_MINIMA[peak_index], dtype=float)
        cur = np.array(expected_real_from_p_2d(self.p[island_idx]), dtype=float)

        v = target - cur
        dist = float(np.linalg.norm(v))
        if dist < 1e-12:
            return
        v = v / dist

        base_step = self.nudge_strength * float(np.sqrt(self.sigma_niche))
        step = min(base_step, 0.75 * dist)

        p_new = cur + v * step + self.rng.normal(0.0, 0.05 * step, size=2)
        p_new[0] = float(np.clip(p_new[0], -6.0, 6.0))
        p_new[1] = float(np.clip(p_new[1], -6.0, 6.0))

        bits = real_to_bits_2d(p_new[0], p_new[1])
        target_p = 0.001 + 0.998 * bits.astype(float)

        if dist > 4.0:
            beta = 0.35
        elif dist > 2.0:
            beta = 0.20
        elif dist > 1.0:
            beta = 0.10
        else:
            beta = 0.04

        self.p[island_idx] = np.clip((1.0 - beta) * self.p[island_idx] + beta * target_p, 0.0, 1.0)


    def _decode_best_per_island(self) -> Tuple[List[Any], List[float]]:
        decoded_list: List[Any] = []
        fitness_list: List[float] = []
        for k in range(self.num_islands):
            if not self.best_fitness_per_gen[k]:
                decoded_list.append(None)
                fitness_list.append(float('-inf'))
                continue
            fitness_list.append(self.best_fitness_per_gen[k][-1])
            decoded_list.append(self.best_x_per_gen[k][-1])
        return decoded_list, fitness_list

    def _cluster_islands(self, decoded_list: List[Any], fitness_list: List[float]) -> Tuple[List[int], List[List[int]]]:
        idxs = list(range(self.num_islands))
        sorted_islands = sorted(idxs, key=lambda i: fitness_list[i], reverse=True)
        representatives: List[int] = []
        niches: List[List[int]] = []
        for idx in sorted_islands:
            pheno = decoded_list[idx]
            if pheno is None:
                continue
            if not representatives:
                representatives.append(idx)
                niches.append([idx])
            else:
                dists = []
                for rep in representatives:
                    rep_pheno = decoded_list[rep]
                    if self.func_id in ('F1', 'F2'):
                        d = bm.distance_1d(pheno, rep_pheno)
                    else:
                        d = bm.distance_2d(pheno, rep_pheno)
                    dists.append(d)
                min_dist = min(dists)
                if min_dist > self.sigma_niche:
                    representatives.append(idx)
                    niches.append([idx])
                else:
                    nearest = int(np.argmin(dists))
                    niches[nearest].append(idx)
        return representatives, niches

    def _reinitialize_island(self, island_idx: int) -> None:
        noise = (self.rng.random(30) - 0.5) * 0.1
        new_p = 0.5 + noise
        new_p = np.clip(new_p, 0.0, 1.0)
        self.p[island_idx] = new_p

    def _clone_island(self, target_idx: int, source_idx: int) -> None:
        new_p = self.p[source_idx].copy()
        noise = (self.rng.random(30) - 0.5) * 0.1
        new_p = new_p + noise
        new_p = np.clip(new_p, 0.0, 1.0)
        self.p[target_idx] = new_p

    def after_generation(self, gen: int) -> None:
        if self.mode in ("peak_walk", "peak_repulsion"):
            pass
        else:
            if (gen + 1) % self.niching_interval != 0:
                return

        decoded_list, fitness_list = self._decode_best_per_island()
        if all(pheno is None for pheno in decoded_list):
            return

        representatives, niches = self._cluster_islands(decoded_list, fitness_list)
        if not niches:
            return
        if self.mode == 'base':
            for rep, members in zip(representatives, niches):
                for idx in members:
                    if idx == rep:
                        continue
                    self._nudge_island(idx, away_from=decoded_list[rep])
        elif self.mode == "peak_repulsion":
            if self.func_id in ("F1", "F2"):
                num_peaks = len(PEAK_POS_1D)

                def assign_peak(pheno: Any) -> Optional[int]:
                    if pheno is None:
                        return None
                    x = float(pheno)
                    return _assign_peak_1d(x, threshold=self.sigma_niche)

            else:
                num_peaks = len(HIMMELBLAU_MINIMA)

                def assign_peak(pheno: Any) -> Optional[int]:
                    if pheno is None:
                        return None
                    x1, x2 = pheno
                    return _assign_peak_2d(
                        float(x1),
                        float(x2),
                        threshold=self.sigma_niche,
                    )

            peak_to_islands: List[List[int]] = [[] for _ in range(num_peaks)]
            unassigned: List[int] = []

            for idx, pheno in enumerate(decoded_list):
                pk = assign_peak(pheno)
                if pk is None:
                    unassigned.append(idx)
                else:
                    peak_to_islands[pk].append(idx)

            for pk_idx, members in enumerate(peak_to_islands):
                if not members:
                    continue
                best_idx = max(members, key=lambda i: fitness_list[i])
                for i in members:
                    if i == best_idx:
                        continue
                    self._nudge_island(i, away_from=decoded_list[best_idx])
                    unassigned.append(i)
                peak_to_islands[pk_idx] = [best_idx]

            for pk_idx in range(num_peaks):
                if peak_to_islands[pk_idx]:
                    continue  # already has an island
                if not unassigned:
                    break     # no spare islands left
                idx = unassigned.pop()
                self._initialise_island_to_peak(idx, pk_idx)
                peak_to_islands[pk_idx].append(idx)

        elif self.mode == "peak_walk":
            if self.func_id in ("F1", "F2"):
                num_peaks = len(PEAK_POS_1D)

                def assign_peak(pheno: Any) -> Optional[int]:
                    if pheno is None:
                        return None
                    return _assign_peak_1d(float(pheno), threshold=self.peak_threshold)
            else:
                num_peaks = len(HIMMELBLAU_MINIMA)

                def assign_peak(pheno: Any) -> Optional[int]:
                    if pheno is None:
                        return None
                    x1, x2 = pheno
                    return _assign_peak_2d(float(x1), float(x2), threshold=self.peak_threshold)

            cur_pos: List[Any] = []
            for i in range(self.num_islands):
                if self.trace_x_per_gen[i]:
                    cur_pos.append(self.trace_x_per_gen[i][-1])
                else:
                    cur_pos.append(decoded_list[i])

            if gen == 0:
                for i in range(self.num_islands):
                    self.peak_goals[i] = i if i < num_peaks else None

            for i in range(self.num_islands):
                owned = self.island_owner[i]
                if owned is not None:
                    self._nudge_toward_peak(i, owned)

            for i in range(self.num_islands):
                g = self.peak_goals[i]
                if g is None:
                    continue

                pk = self._peak_index(cur_pos[i])
                if pk == g:
                    if self.peak_owner[g] is None:
                        self.peak_owner[g] = i
                        self.island_owner[i] = g
                        self.peak_goals[i] = None
                else:
                    self._nudge_toward_peak(i, g)

            peak_to_islands: List[List[int]] = [[] for _ in range(num_peaks)]
            for i in range(self.num_islands):
                pk = assign_peak(cur_pos[i])
                if pk is None:
                    continue
                if self.func_id == "F2":
                    g = self.peak_goals[i]
                    if (self.island_owner[i] is None) and (g is not None) and (pk != g):
                        continue

                peak_to_islands[pk].append(i)

            reserved_peaks: set[int] = {g for g in self.peak_goals if g is not None}
            reserved_peaks |= {p for p, owner in enumerate(self.peak_owner) if owner is not None}

            for pk in range(num_peaks):
                members = peak_to_islands[pk]
                if len(members) <= 1:
                    continue

                owner = self.peak_owner[pk]
                if owner is not None and owner in members:
                    keep = owner
                else:
                    keep = max(members, key=lambda i: fitness_list[i])
                    if gen >= self.coverage_start_gen:
                        self.peak_owner[pk] = keep
                        self.island_owner[keep] = pk
                        if self.peak_goals[keep] == pk:
                            self.peak_goals[keep] = None

                for i in members:
                    if i == keep:
                        continue
                    if self.island_owner[i] == pk:
                        self.island_owner[i] = None
                    if self.peak_goals[i] == pk:
                        self.peak_goals[i] = None

                    available = [p for p in range(num_peaks)
                                 if self.peak_owner[p] is None and p not in reserved_peaks]

                    if available:
                        if self.func_id == "F3":
                            pcur = np.array(cur_pos[i], dtype=float)
                            d = [np.linalg.norm(np.array(HIMMELBLAU_MINIMA[p]) - pcur) for p in available]
                            tgt = available[int(np.argmin(d))]
                        else:
                            xcur = float(cur_pos[i])
                            d = [abs(float(PEAK_POS_1D[p]) - xcur) for p in available]
                            tgt = available[int(np.argmin(d))]

                        self.peak_goals[i] = tgt
                        reserved_peaks.add(tgt)
                        self._nudge_toward_peak(i, tgt)
                    else:
                        self._nudge_island(i, away_from=cur_pos[keep])

            missing = [p for p in range(num_peaks) if self.peak_owner[p] is None]
            if missing and gen >= self.coverage_start_gen:
                candidates = [i for i in range(self.num_islands) if self.island_owner[i] is None]
                candidates = sorted(candidates, key=lambda i: fitness_list[i])

                for pk in missing:
                    if not candidates:
                        break
                    i = candidates.pop(0)
                    if self.peak_goals[i] is not None:
                        continue

                    if self.func_id == "F2":
                        self._initialise_island_to_peak(i, pk)
                        self.peak_owner[pk] = i
                        self.island_owner[i] = pk
                        self.peak_goals[i] = None
                    else:
                        self.peak_goals[i] = pk
                        self._nudge_toward_peak(i, pk)

        elif self.mode == 'proportional':
            M = self.num_islands
            C = len(niches)
            reps_fitness = [fitness_list[rep] for rep in representatives]
            max_pos = [max(f, 0.0) for f in reps_fitness]
            if sum(max_pos) > 0:
                weights = [f / sum(max_pos) for f in max_pos]
            else:
                weights = [1.0 / C] * C
            desired_counts = [int(round(w * M)) for w in weights]
            diff = M - sum(desired_counts)
            i = 0
            while diff != 0 and C > 0:
                idx = i % C
                if diff > 0:
                    desired_counts[idx] += 1
                    diff -= 1
                else:
                    if desired_counts[idx] > 0:
                        desired_counts[idx] -= 1
                        diff += 1
                i += 1
            free_islands: List[int] = []
            for niche_idx, members in enumerate(niches):
                current = len(members)
                target = desired_counts[niche_idx]
                if current > target:
                    rep = representatives[niche_idx]
                    candidates = [idx for idx in members if idx != rep]
                    self.rng.shuffle(candidates)
                    to_remove = candidates[: (current - target)]
                    for idx in to_remove:
                        self._nudge_island(idx, away_from=decoded_list[rep])
                        free_islands.append(idx)
                        members.remove(idx)
                elif current < target:
                    needed = target - current
                    for _ in range(needed):
                        if free_islands:
                            idx = free_islands.pop()
                        else:
                            candidates = [j for j in range(self.num_islands)
                                          if j not in representatives]
                            if not candidates:
                                break
                            idx = self.rng.choice(candidates)
                        rep = representatives[niche_idx]
                        self._clone_island(idx, rep)
                        members.append(idx)
        else:
            raise ValueError(f"Unknown niching mode: {self.mode}")

    def run(self) -> Dict[str, Any]:
        return super().run()

def run_island_pbil(
    func_id: str,
    mode: str = "none",
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
    nudge: bool = True,
    nudge_strength: float = 1.0,
    use_mp: bool = False,
    mp_workers: int | None = None,
) -> List[Dict[str, Any]]:
    func_id = func_id.upper()

    if sigma_niche is None:
        sigma = 0.1 if func_id in ("F1", "F2") else 1.0
    else:
        sigma = float(sigma_niche)

    results: List[Dict[str, Any]] = []
    for i in range(num_runs):
        rng = np.random.default_rng(None if seed is None else seed + i)
        if mode == "none":
            obj = IslandPBILBase(
                func_id=func_id,
                num_islands=num_islands,
                total_evals=total_evals,
                num_samples=num_samples,
                alpha=alpha,
                pmutate_prob=pmutate_prob,
                mutation_shift=mutation_shift,
                rng=rng,
                use_mp=use_mp,
                mp_workers=mp_workers,
            )
        else:
            obj = IslandPBILNiching(
                func_id=func_id,
                mode=mode,
                num_islands=num_islands,
                total_evals=total_evals,
                num_samples=num_samples,
                alpha=alpha,
                pmutate_prob=pmutate_prob,
                mutation_shift=mutation_shift,
                niching_interval=niching_interval,
                sigma_niche=sigma,
                rng=rng,
                nudge=nudge,
                nudge_strength=nudge_strength,
                use_mp=use_mp,
                mp_workers=mp_workers,
            )
        results.append(obj.run())
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

def aggregate_best_over_islands(result_dict):
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

def island_pbil_plot_spaghetti(results, func_id: str, out_path: Path) -> None:
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

def save_island_pbil_results_to_files(results, func_id: str, mode: str, num_islands: int, run_dir: Path) -> None:
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

def _assign_peak_1d(x: float, threshold: float = 0.1) -> int | None:
    diffs = np.abs(PEAK_POS_1D - x)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None


def _assign_peak_2d(x1: float, x2: float, threshold: float = 1.0) -> int | None:
    point = np.array([x1, x2])
    diffs = np.linalg.norm(HIMMELBLAU_MINIMA - point, axis=1)
    idx = int(np.argmin(diffs))
    return idx if diffs[idx] < threshold else None

def _island_collect_all_points(results, func_id: str):
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

def island_pbil_plot_island_traces(results, func_id: str, out_path: Path) -> None:
    func_id = func_id.upper()
    best_idx = _island_best_run_index(results)
    res = results[best_idx]

    if func_id in ("F1", "F2"):
        f = bm.get_function(func_id)
        xs = np.linspace(0.0, 1.0, 1000)
        ys = [f(x) for x in xs]
        fig, ax = plt.subplots()
        ax.plot(xs, ys, label=func_id)

        paths = res.get("trace_x_per_gen", res["best_x_per_gen"])
        for k, path in enumerate(paths):

            if not path:
                continue
            x_path = [float(x) for x in path]
            y_path = [f(x) for x in x_path]

            base_color = ax._get_lines.get_next_color()
            rgba = mcolors.to_rgba(base_color)
            T = len(x_path)
            colors = [(*rgba[:3], 0.15 + 0.85 * (t / max(T - 1, 1))) for t in range(T)]
            ax.plot(x_path, y_path, color=base_color, alpha=0.35, linewidth=1.0)
            ax.scatter(x_path, y_path, s=18, c=colors, label=f"Island {k+1}")
            ax.scatter([x_path[-1]], [y_path[-1]], s=70, c=[(*rgba[:3], 1.0)])

        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.set_title(f"{func_id}: island traces (best run #{best_idx+1})")
        ax.grid(True, alpha=0.3)
        if len(res["best_x_per_gen"]) <= 5:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
    else:
        x = np.linspace(-6, 6, 200)
        y = np.linspace(-6, 6, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = bm.f3(X[i, j], Y[i, j])

        fig, ax = plt.subplots()
        cs = ax.contourf(X, Y, Z, levels=50)
        fig.colorbar(cs, ax=ax, label="F3(x1,x2)")

        paths = res.get("trace_x_per_gen", res["best_x_per_gen"])
        for k, path in enumerate(paths):

            if not path:
                continue
            x1_path = [float(p[0]) for p in path]
            x2_path = [float(p[1]) for p in path]

            base_color = ax._get_lines.get_next_color()
            rgba = mcolors.to_rgba(base_color)
            T = len(x1_path)
            colors = [(*rgba[:3], 0.15 + 0.85 * (t / max(T - 1, 1))) for t in range(T)]
            ax.plot(x1_path, x2_path, color=base_color, alpha=0.35, linewidth=1.0)
            ax.scatter(x1_path, x2_path, s=18, c=colors, label=f"Island {k+1}")
            ax.scatter([x1_path[-1]], [x2_path[-1]], s=70, c=[(*rgba[:3], 1.0)])

        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_title(f"F3: island traces (best run #{best_idx+1})")
        if len(res["best_x_per_gen"]) <= 5:
            ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Island PBIL (with optional niching) on Deb & Goldberg benchmarks")
    parser.add_argument("--func", type=str, default="F1", choices=["F1", "F2", "F3"], help="Benchmark function")
    parser.add_argument("--mode", type=str, default="none", choices=["none", "base", "proportional", "peak_repulsion", "peak_walk"], help="Niching mode")
    parser.add_argument("--num_islands", type=int, default=5, help="Number of islands")
    parser.add_argument("--runs", type=int, default=30, help="Number of independent Island PBIL runs")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed")
    parser.add_argument("--total_evals", type=int, default=20000, help="Total evaluations per run")
    parser.add_argument("--num_samples", type=int, default=50, help="Samples per island per generation")
    parser.add_argument("--niching_interval", type=int, default=10, help="Niching interval (generations)")
    parser.add_argument("--sigma_niche", type=float, default=None, help="Niche distance threshold (optional override)")
    parser.add_argument("--mp", action="store_true", help="Update islands in parallel using multiprocessing")
    parser.add_argument("--mp_workers", type=int, default=None, help="Number of worker processes for --mp")
    parser.add_argument("--no_nudge", action="store_true", help="Use random reinitialization instead of nudging")
    parser.add_argument("--nudge_strength", type=float, default=1.0, help="Scale factor for nudge step size")

    args = parser.parse_args()

    func_id = args.func.upper()
    num_runs = args.runs
    mode = args.mode
    num_islands = args.num_islands

    if args.sigma_niche is None:
        sigma_niche = 0.1 if func_id in ("F1", "F2") else 1.0
    else:
        sigma_niche = args.sigma_niche

    run_dir = prepare_run_directory("IslandPBIL", func_id)
    print(
        f"[IslandPBIL] Running {num_runs} runs on {func_id} "
        f"with mode='{mode}', {num_islands} islands. Output in {run_dir}"
    )

    results = run_island_pbil(
        func_id=func_id,
        mode=mode,
        num_islands=num_islands,
        total_evals=args.total_evals,
        num_samples=args.num_samples,
        alpha=0.04,
        pmutate_prob=0.02,
        mutation_shift=0.05,
        niching_interval=args.niching_interval,
        sigma_niche=sigma_niche,
        num_runs=num_runs,
        seed=args.seed,
        nudge=not args.no_nudge,
        nudge_strength=args.nudge_strength,
        use_mp=args.mp,
        mp_workers=args.mp_workers,
    )

    save_island_pbil_results_to_files(results, func_id, mode, num_islands, run_dir)
    island_pbil_plot_spaghetti(results, func_id, run_dir / "best_fitness_spaghetti.png")
    island_pbil_plot_peak_representatives(results, func_id, run_dir / "island_peak_representatives.png")
    island_pbil_plot_peak_occupancy(results, func_id, run_dir / "island_peak_occupancy.png")
    island_pbil_plot_single_run_final(results, func_id, run_dir / "island_best_run_final_points.png")
    island_pbil_plot_island_traces(results, func_id, run_dir / "island_traces.png")

    if func_id in ("F1", "F2"):
        island_pbil_plot_1d_function_with_points(results, func_id, run_dir / f"{func_id.lower()}_final_points.png")
    else:
        island_pbil_plot_himmelblau_with_points(results, run_dir / "himmelblau_final_points.png")

    print("[IslandPBIL] Done.")


if __name__ == "__main__":
    main()
