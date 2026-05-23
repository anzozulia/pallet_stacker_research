"""
pallet_packer.brkga_v3 — Goncalves & Resende 2013 BRKGA reproduction.

A from-scratch implementation of the BRKGA-2013 approach for 3D bin
packing (Gonçalves & Resende, IJPE 145, 500-510). Key components:

  * Maximal-space (EMS) representation of free volume per bin
  * Difference Process for EMS splitting on placement (3 children per
    overlap, Lai-Chan 1997 style)
  * DFTRC-2 placement heuristic — place box at corner maximizing
    distance from front-top-right corner of bin
  * Two-part chromosome: BPS (box packing sequence) + VBO (vector
    of box orientations)
  * BRKGA loop with elite preservation, biased elite/non-elite
    crossover, and random mutant injection

Targets the academic BR1-BR7 benchmark. Pure geometric (no support
/ weight / fragility constraints in the inner decoder — these can
be enforced at the outer level via replay-validate).

For real-cargo use with constraints, use the v2 packer
(`pallet_packer.packer.PalletPacker`) which has the full constraint
stack. v3 is for matching academic literature on BR.

References:
  Gonçalves, J.F. & Resende, M.G.C. (2013). "A biased random key
    genetic algorithm for 2D and 3D bin packing problems."
    Int. J. Production Economics 145, 500-510.
  Lai, K.K. & Chan, J.W.M. (1997). "An evolutionary algorithm for the
    rectangular cutting stock problem." Int. J. Industrial Engineering
    4, 130-139. (Source of the 3-EMS difference process.)
"""
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from .models import Box, Pallet, PackerConfig, Placement, Rotation
from .packer import PackResult, PalletState


class MaxSpaceBin:
    """A single bin tracking empty maximal spaces (EMSs) as a numpy array.

    EMS storage: ndarray of shape (n_ems, 2, 3) where
      emss[i, 0, :] = (x_min, y_min, z_min) of EMS i
      emss[i, 1, :] = (x_max, y_max, z_max) of EMS i
    """

    def __init__(self, L: int, W: int, H: int):
        self.L, self.W, self.H = L, W, H
        self.emss = np.array([[[0, 0, 0], [L, W, H]]], dtype=np.int64)
        self.placements: List[Tuple] = []

    def find_best_dftrc(self, dims: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
        """For a box with the given dims (already-chosen rotation),
        return (ems_idx, x, y, z) of the placement that maximizes
        distance to the front-top-right corner. None if no EMS fits."""
        dx, dy, dz = dims
        if self.emss.shape[0] == 0:
            return None
        ems_min = self.emss[:, 0, :]
        ems_size = self.emss[:, 1, :] - ems_min
        fits = ((ems_size[:, 0] >= dx) &
                (ems_size[:, 1] >= dy) &
                (ems_size[:, 2] >= dz))
        if not fits.any():
            return None
        fitting_idx = np.where(fits)[0]
        x_arr = ems_min[fitting_idx, 0]
        y_arr = ems_min[fitting_idx, 1]
        z_arr = ems_min[fitting_idx, 2]
        d_sq = ((self.L - x_arr - dx) ** 2
                + (self.W - y_arr - dy) ** 2
                + (self.H - z_arr - dz) ** 2)
        best = int(np.argmax(d_sq))
        ems_idx = int(fitting_idx[best])
        return (ems_idx, int(x_arr[best]), int(y_arr[best]), int(z_arr[best]))

    def commit(self, ems_idx: int, x: int, y: int, z: int,
               dims: Tuple[int, int, int], box: Box, rot_idx: int) -> None:
        """Place box and update EMS list via vectorized difference process."""
        dx, dy, dz = dims
        bx1, by1, bz1 = x, y, z
        bx2, by2, bz2 = x + dx, y + dy, z + dz

        if self.emss.shape[0] == 0:
            self.emss = np.empty((0, 2, 3), dtype=np.int64)
            self.placements.append((box, rot_idx, x, y, z, dx, dy, dz))
            return

        ems_mins = self.emss[:, 0, :]   # (n, 3)
        ems_maxs = self.emss[:, 1, :]   # (n, 3)

        # Vectorized overlap: box [bx1..bx2) overlaps EMS [ems_min..ems_max).
        overlap_mask = (
            (ems_maxs[:, 0] > bx1) & (ems_mins[:, 0] < bx2) &
            (ems_maxs[:, 1] > by1) & (ems_mins[:, 1] < by2) &
            (ems_maxs[:, 2] > bz1) & (ems_mins[:, 2] < bz2)
        )

        kept_emss = self.emss[~overlap_mask]      # untouched survive
        overlapping = self.emss[overlap_mask]      # need splitting

        # Build child EMSs for each overlapping EMS, 6 per original.
        children: List[np.ndarray] = [kept_emss]
        if overlapping.shape[0] > 0:
            ex1 = overlapping[:, 0, 0]; ey1 = overlapping[:, 0, 1]; ez1 = overlapping[:, 0, 2]
            ex2 = overlapping[:, 1, 0]; ey2 = overlapping[:, 1, 1]; ez2 = overlapping[:, 1, 2]
            # Six difference-process EMS candidates as (lo, hi) ndarrays.
            cand_specs = [
                # +x leftover
                (np.column_stack([np.full_like(ex1, bx2), ey1, ez1]),
                 np.column_stack([ex2, ey2, ez2])),
                # -x leftover
                (np.column_stack([ex1, ey1, ez1]),
                 np.column_stack([np.full_like(ex2, bx1), ey2, ez2])),
                # +y leftover
                (np.column_stack([ex1, np.full_like(ey1, by2), ez1]),
                 np.column_stack([ex2, ey2, ez2])),
                # -y leftover
                (np.column_stack([ex1, ey1, ez1]),
                 np.column_stack([ex2, np.full_like(ey2, by1), ez2])),
                # +z leftover
                (np.column_stack([ex1, ey1, np.full_like(ez1, bz2)]),
                 np.column_stack([ex2, ey2, ez2])),
                # -z leftover
                (np.column_stack([ex1, ey1, ez1]),
                 np.column_stack([ex2, ey2, np.full_like(ez2, bz1)])),
            ]
            for lo, hi in cand_specs:
                valid = (hi > lo).all(axis=1)
                if valid.any():
                    children.append(np.stack([lo[valid], hi[valid]], axis=1))

        all_children = np.concatenate(children, axis=0)
        if all_children.shape[0] == 0:
            self.emss = np.empty((0, 2, 3), dtype=np.int64)
            self.placements.append((box, rot_idx, x, y, z, dx, dy, dz))
            return

        # Vectorized dominance pruning: drop any EMS strictly inscribed by another.
        # EMS i is inscribed in j if mins[j] <= mins[i] AND maxs[i] <= maxs[j], i!=j.
        mins = all_children[:, 0, :]   # (n, 3)
        maxs = all_children[:, 1, :]   # (n, 3)
        # mins_lo[i,j] = all(mins[j] <= mins[i])
        # maxs_hi[i,j] = all(maxs[i] <= maxs[j])
        mins_lo = (mins[None, :, :] <= mins[:, None, :]).all(axis=2)  # (n, n)
        maxs_hi = (maxs[:, None, :] <= maxs[None, :, :]).all(axis=2)  # (n, n)
        dominated_by = mins_lo & maxs_hi
        np.fill_diagonal(dominated_by, False)
        # Also: EMSs identical to another (mutual dominance) — keep only the first.
        # The above keeps both as dominated by each other. Resolve by breaking ties
        # in favor of lower index (keep earlier).
        is_dominated = dominated_by.any(axis=1)
        # For equal pairs, dominated_by[i,j] and dominated_by[j,i] both True.
        # Override: if j < i and they mutually dominate, keep j, drop i.
        # The above is_dominated will mark BOTH true → both dropped. Fix:
        for i in np.where(is_dominated)[0]:
            doms = np.where(dominated_by[i])[0]
            # If all dominators are bidirectional and have higher index → keep i.
            # If any dominator has strictly larger volume or lower index → drop i.
            mutual = doms[dominated_by[doms, i]]
            strict = np.setdiff1d(doms, mutual)
            if strict.size == 0:
                # Only mutual dominators; keep the lowest-indexed one.
                if i < mutual.min():
                    is_dominated[i] = False  # i is the keeper

        self.emss = all_children[~is_dominated]
        self.placements.append((box, rot_idx, x, y, z, dx, dy, dz))


def _rotations_dims(box: Box) -> List[Tuple[Rotation, Tuple[int, int, int]]]:
    out: List[Tuple[Rotation, Tuple[int, int, int]]] = []
    for r in box.allowed_rotations:
        dx, dy, dz = box.dims_for(r)
        out.append((r, (int(round(dx)), int(round(dy)), int(round(dz)))))
    return out


def decode_chromosome(
    chrom: np.ndarray,
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    max_pallets: Optional[int] = None,
    try_all_rotations: bool = True,
) -> PackResult:
    """Decode a 2N random-key chromosome into a PackResult."""
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    bps = chrom[:n]
    vbo = chrom[n:]
    order = np.argsort(bps)

    L = int(round(pallet.length))
    W = int(round(pallet.width))
    H = int(round(pallet.height))

    bins: List[MaxSpaceBin] = []
    unpacked: List[Box] = []
    box_rots = [_rotations_dims(boxes[i]) for i in range(n)]

    for idx in order:
        rots = box_rots[idx]
        n_rots = len(rots)
        if n_rots == 0:
            unpacked.append(boxes[idx])
            continue

        if try_all_rotations:
            rot_indices_to_try = list(range(n_rots))
        else:
            k = min(n_rots - 1, int(vbo[idx] * n_rots))
            rot_indices_to_try = [k]

        placed = False
        for b in bins:
            best = None
            best_score = -1
            for k in rot_indices_to_try:
                _, dims = rots[k]
                choice = b.find_best_dftrc(dims)
                if choice is None:
                    continue
                ems_idx, x, y, z = choice
                dx, dy, dz = dims
                score = ((b.L - x - dx) ** 2 + (b.W - y - dy) ** 2
                         + (b.H - z - dz) ** 2)
                if score > best_score:
                    best_score = score
                    best = (k, ems_idx, x, y, z, dims)
            if best is not None:
                k, ems_idx, x, y, z, dims = best
                b.commit(ems_idx, x, y, z, dims, boxes[idx], k)
                placed = True
                break

        if not placed:
            if max_pallets is not None and len(bins) >= max_pallets:
                unpacked.append(boxes[idx])
                continue
            new_bin = MaxSpaceBin(L, W, H)
            best = None
            best_score = -1
            for k in rot_indices_to_try:
                _, dims = rots[k]
                choice = new_bin.find_best_dftrc(dims)
                if choice is None:
                    continue
                ems_idx, x, y, z = choice
                dx, dy, dz = dims
                score = ((new_bin.L - x - dx) ** 2
                         + (new_bin.W - y - dy) ** 2
                         + (new_bin.H - z - dz) ** 2)
                if score > best_score:
                    best_score = score
                    best = (k, ems_idx, x, y, z, dims)
            if best is None:
                unpacked.append(boxes[idx])
                continue
            k, ems_idx, x, y, z, dims = best
            new_bin.commit(ems_idx, x, y, z, dims, boxes[idx], k)
            bins.append(new_bin)

    pallets_state: List[PalletState] = []
    for i, b in enumerate(bins):
        st = PalletState(pallet, f"P{i + 1:03d}", config)
        for (box, rot_idx, x, y, z, dx, dy, dz) in b.placements:
            rotation = box.allowed_rotations[rot_idx]
            pl = Placement(box=box, rotation=rotation,
                           x=float(x), y=float(y), z=float(z))
            st.placements.append(pl)
            st.total_weight += box.weight
        pallets_state.append(st)
    return PackResult(pallets=pallets_state, unpacked=unpacked)


def fitness(result: PackResult, pallet: Pallet,
            max_pallets: Optional[int] = None) -> float:
    """Lower is better.

    For BR-style single-pallet (max_pallets=1): just maximize volume
    utilization of pallet 1. Unpacked items are EXPECTED in BR (single
    container, fit as much as possible by volume) and aren't penalized.

    For multi-pallet: bin count + least-load fraction (Goncalves-Resende
    2013 §3.4 — biases toward making the least-loaded bin droppable).
    """
    cap = pallet.length * pallet.width * pallet.height
    if max_pallets == 1:
        if result.pallets:
            used = sum(p.box.volume for p in result.pallets[0].placements)
            util = used / cap if cap > 0 else 0.0
        else:
            util = 0.0
        return 1.0 - util
    nb = result.num_pallets + (1 if result.unpacked else 0)
    if not result.pallets:
        return float(nb)
    utils = [sum(p.box.volume for p in st.placements) / cap
             for st in result.pallets]
    least = min(utils) if utils else 0.0
    return nb + least


def brkga_pack(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    population_size: int = 200,
    generations: int = 50,
    elite_fraction: float = 0.25,
    mutant_fraction: float = 0.20,
    p_elite_inherit: float = 0.70,
    time_limit_s: float = 30.0,
    max_pallets: Optional[int] = None,
    seed: int = 42,
    patience: int = 15,
    try_all_rotations: bool = True,
    verbose: bool = False,
    n_populations: int = 1,
    migration_interval: int = 5,
    migrants_per_swap: int = 2,
) -> PackResult:
    """Run BRKGA-2013-style search and return the best PackResult found.

    n_populations > 1 enables multi-population BRKGA: K independent
    populations evolve in parallel, with the top `migrants_per_swap`
    elites swapped between adjacent populations every
    `migration_interval` generations. This is the actual Goncalves-
    Resende 2013 variant — single-population can converge prematurely.
    """
    if n_populations > 1:
        return _multi_population_brkga(
            boxes, pallet, config,
            population_size=population_size,
            generations=generations,
            elite_fraction=elite_fraction,
            mutant_fraction=mutant_fraction,
            p_elite_inherit=p_elite_inherit,
            time_limit_s=time_limit_s,
            max_pallets=max_pallets,
            seed=seed,
            patience=patience,
            try_all_rotations=try_all_rotations,
            verbose=verbose,
            n_populations=n_populations,
            migration_interval=migration_interval,
            migrants_per_swap=migrants_per_swap,
        )
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    rng = np.random.default_rng(seed)
    P = max(10, population_size)
    n_elite = max(1, int(P * elite_fraction))
    n_mutant = max(1, int(P * mutant_fraction))
    n_cross = max(0, P - n_elite - n_mutant)

    pop = rng.random((P, 2 * n))
    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    gens_no_improve = 0
    t0 = time.time()

    for gen in range(generations):
        if time.time() - t0 > time_limit_s:
            if verbose:
                print(f"  [BRKGA] gen {gen}: time budget hit, stopping")
            break

        fitnesses = np.zeros(P)
        results: List[Optional[PackResult]] = [None] * P
        for i in range(P):
            res = decode_chromosome(pop[i], boxes, pallet, config,
                                    max_pallets=max_pallets,
                                    try_all_rotations=try_all_rotations)
            fitnesses[i] = fitness(res, pallet, max_pallets=max_pallets)
            results[i] = res

        sorted_idx = np.argsort(fitnesses)
        best_i = int(sorted_idx[0])
        if fitnesses[best_i] < best_fitness - 1e-9:
            best_fitness = float(fitnesses[best_i])
            best_result = results[best_i]
            gens_no_improve = 0
            if verbose:
                print(f"  [BRKGA] gen {gen}: best fitness={best_fitness:.4f}")
        else:
            gens_no_improve += 1
            if gens_no_improve >= patience:
                if verbose:
                    print(f"  [BRKGA] gen {gen}: patience exhausted")
                break

        new_pop = np.zeros_like(pop)
        new_pop[:n_elite] = pop[sorted_idx[:n_elite]]
        new_pop[n_elite:n_elite + n_mutant] = rng.random((n_mutant, 2 * n))
        elite_pool = pop[sorted_idx[:n_elite]]
        non_elite_pool = pop[sorted_idx[n_elite:]]
        if n_cross > 0:
            pa_idx = rng.integers(0, n_elite, size=n_cross)
            pb_idx = rng.integers(0, len(non_elite_pool), size=n_cross)
            pa_parents = elite_pool[pa_idx]
            pb_parents = non_elite_pool[pb_idx]
            mask = rng.random((n_cross, 2 * n)) < p_elite_inherit
            new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
        pop = new_pop

    if best_result is None:
        return PackResult(pallets=[], unpacked=list(boxes))
    return best_result


def _multi_population_brkga(
    boxes: List[Box], pallet: Pallet, config: PackerConfig,
    *, population_size: int, generations: int,
    elite_fraction: float, mutant_fraction: float, p_elite_inherit: float,
    time_limit_s: float, max_pallets: Optional[int], seed: int,
    patience: int, try_all_rotations: bool, verbose: bool,
    n_populations: int, migration_interval: int, migrants_per_swap: int,
) -> PackResult:
    """Multi-population BRKGA. K independent populations swap elites
    every `migration_interval` generations to escape local optima."""
    n = len(boxes)
    if n == 0:
        return PackResult(pallets=[], unpacked=[])

    K = max(2, n_populations)
    P = max(10, population_size // K) * K  # divide budget across pops
    pop_size = P // K
    n_elite = max(1, int(pop_size * elite_fraction))
    n_mutant = max(1, int(pop_size * mutant_fraction))
    n_cross = max(0, pop_size - n_elite - n_mutant)

    rngs = [np.random.default_rng(seed + i) for i in range(K)]
    pops = [rngs[i].random((pop_size, 2 * n)) for i in range(K)]

    best_fitness = float('inf')
    best_result: Optional[PackResult] = None
    gens_no_improve = 0
    t0 = time.time()

    pop_fits: List[Optional[np.ndarray]] = [None] * K
    pop_results: List[Optional[List]] = [None] * K

    for gen in range(generations):
        if time.time() - t0 > time_limit_s:
            if verbose:
                print(f"  [BRKGA-MP] gen {gen}: time hit")
            break

        # Evaluate each population.
        for k in range(K):
            fits = np.zeros(pop_size)
            results = [None] * pop_size
            for i in range(pop_size):
                res = decode_chromosome(pops[k][i], boxes, pallet, config,
                                        max_pallets=max_pallets,
                                        try_all_rotations=try_all_rotations)
                fits[i] = fitness(res, pallet, max_pallets=max_pallets)
                results[i] = res
            pop_fits[k] = fits
            pop_results[k] = results
            # Track global best.
            best_i = int(np.argmin(fits))
            if fits[best_i] < best_fitness - 1e-9:
                best_fitness = float(fits[best_i])
                best_result = results[best_i]
                gens_no_improve = 0
                if verbose:
                    print(f"  [BRKGA-MP] gen {gen} pop {k}: best={best_fitness:.4f}")

        if gens_no_improve >= patience:
            if verbose:
                print(f"  [BRKGA-MP] gen {gen}: patience exhausted")
            break
        gens_no_improve += 1

        # Migration: every migration_interval gens, copy top `migrants_per_swap`
        # elites from population k to population (k+1) % K, replacing worst.
        if gen > 0 and gen % migration_interval == 0:
            for k in range(K):
                src = k
                dst = (k + 1) % K
                src_sorted = np.argsort(pop_fits[src])[:migrants_per_swap]
                dst_sorted = np.argsort(pop_fits[dst])[-migrants_per_swap:]
                # Replace worst of dst with best of src.
                for s, d in zip(src_sorted, dst_sorted):
                    pops[dst][d] = pops[src][s].copy()

        # Evolve each population independently.
        new_pops = []
        for k in range(K):
            sorted_idx = np.argsort(pop_fits[k])
            new_pop = np.zeros_like(pops[k])
            new_pop[:n_elite] = pops[k][sorted_idx[:n_elite]]
            new_pop[n_elite:n_elite + n_mutant] = rngs[k].random((n_mutant, 2 * n))
            elite_pool = pops[k][sorted_idx[:n_elite]]
            non_elite_pool = pops[k][sorted_idx[n_elite:]]
            if n_cross > 0:
                pa_idx = rngs[k].integers(0, n_elite, size=n_cross)
                pb_idx = rngs[k].integers(0, len(non_elite_pool), size=n_cross)
                pa_parents = elite_pool[pa_idx]
                pb_parents = non_elite_pool[pb_idx]
                mask = rngs[k].random((n_cross, 2 * n)) < p_elite_inherit
                new_pop[n_elite + n_mutant:] = np.where(mask, pa_parents, pb_parents)
            new_pops.append(new_pop)
        pops = new_pops

    if best_result is None:
        return PackResult(pallets=[], unpacked=list(boxes))
    return best_result
