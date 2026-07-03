"""
pallet_packer.models — data models, configuration, and constants.

Pure data layer with no algorithm dependencies. Contains:
  - Rotation: the 6 axis-aligned orientations + the ALL_ROTATIONS /
    THIS_SIDE_UP / NO_ROTATION presets.
  - Box, Pallet, Placement: cargo + container + position records.
  - PackerConfig: the ~50-field configuration that toggles every algorithm
    feature and constraint.

PackResult and PalletState (which hold algorithmic state, not just data)
live in `pallet_packer.packer`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Floating-point tolerance, shared across all modules.
# ---------------------------------------------------------------------------
EPS = 1e-6


def load_tol(limit: float) -> float:
    """Scale-aware comparison tolerance for load/weight limits (round 3,
    F25). The absolute EPS (1e-6) is below one double ulp for limits >=
    ~4.5e9 while the contract allows weights up to 1e12 — an epsilon that
    silently vanishes makes exactly-at-limit stacks flip on float
    accumulation order. Identical to EPS for limits <= 1e3, so
    small-scale behavior is unchanged. Mirrors the inline formula in the
    JIT twins (jit_constraints)."""
    rel = 1e-9 * limit
    return rel if rel > EPS else EPS


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------
class Rotation(Enum):
    """Six axis-aligned orientations.

    Each value is a permutation (a, b, c) of (0, 1, 2) meaning:
        new dx = original_dim[a]
        new dy = original_dim[b]
        new dz = original_dim[c]
    where original_dim = (length, width, height).

    So LWH = (0, 1, 2) is identity: L -> X, W -> Y, H -> Z.
    """
    LWH = (0, 1, 2)  # default upright
    WLH = (1, 0, 2)  # rotated 90 deg around Z (still upright)
    LHW = (0, 2, 1)  # tipped: W is now up
    HWL = (2, 1, 0)  # tipped: L is now up
    WHL = (1, 2, 0)
    HLW = (2, 0, 1)


ALL_ROTATIONS: List[Rotation] = list(Rotation)
THIS_SIDE_UP: List[Rotation] = [Rotation.LWH, Rotation.WLH]  # H stays vertical
NO_ROTATION: List[Rotation] = [Rotation.LWH]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class Box:
    id: str
    length: float  # original X-extent
    width: float   # original Y-extent
    height: float  # original Z-extent
    weight: float = 0.0
    # Max weight (kg) this box can bear on its current top face.
    # Use 0 for "nothing may be placed on top" (fragile).
    max_load_on_top: float = float("inf")
    allowed_rotations: List[Rotation] = field(
        default_factory=lambda: list(ALL_ROTATIONS)
    )
    # Optional fixed group: boxes in the same group must end up on the same pallet.
    group: Optional[str] = None
    # If True, the box must rest on a 100% supported surface (no partial overhang).
    # Used internally for super-blocks formed by block-building so that internal
    # decomposition stays valid.
    requires_full_support: bool = False

    def dims_for(self, rotation: Rotation) -> Tuple[float, float, float]:
        """Return (dx, dy, dz) when oriented according to `rotation`."""
        original = (self.length, self.width, self.height)
        a, b, c = rotation.value
        return original[a], original[b], original[c]

    @property
    def volume(self) -> float:
        return self.length * self.width * self.height


@dataclass
class Pallet:
    length: float                # X extent
    width: float                 # Y extent
    height: float                # Z extent (max stack height)
    max_weight: float = float("inf")
    # Allow box footprints to extend this far beyond pallet edges (mm).
    max_overhang: float = 0.0
    # Pallet CoG envelope: the (x, y) of the pallet's CoG must remain inside
    # [cx_min, cx_max] x [cy_min, cy_max]. Defaults to centred +/- 25% box.
    cog_x_range: Optional[Tuple[float, float]] = None
    cog_y_range: Optional[Tuple[float, float]] = None


@dataclass
class Placement:
    box: Box
    rotation: Rotation
    x: float
    y: float
    z: float

    @property
    def dims(self) -> Tuple[float, float, float]:
        return self.box.dims_for(self.rotation)

    @property
    def dx(self) -> float: return self.dims[0]
    @property
    def dy(self) -> float: return self.dims[1]
    @property
    def dz(self) -> float: return self.dims[2]
    @property
    def x2(self) -> float: return self.x + self.dx
    @property
    def y2(self) -> float: return self.y + self.dy
    @property
    def z2(self) -> float: return self.z + self.dz

    def overlaps(self, other: "Placement") -> bool:
        return (self.x + EPS < other.x2 and self.x2 > other.x + EPS and
                self.y + EPS < other.y2 and self.y2 > other.y + EPS and
                self.z + EPS < other.z2 and self.z2 > other.z + EPS)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class PackerConfig:
    # Stability: fraction of base area that must rest on supporting items or floor.
    # 1.0 = full support / no overhang at all. 0.75 is a common default
    # (matches `py3dbp.support_surface_ratio`).
    support_ratio: float = 0.8
    # Reject placements whose footprint centroid is not over a supporting item.
    require_centroid_supported: bool = True
    # Allow boxes to extend beyond pallet edges (separate from support_ratio,
    # which is about inter-box support). 0 means no overhang of pallet edges.
    allow_pallet_overhang: bool = False
    # Heavier on bottom: when sorting boxes, prefer heavier first.
    heavy_on_bottom: bool = True
    # Pallet CoG envelope (fraction of pallet length/width from centre).
    # 0.25 means the pallet CoG must stay within +/- 25% of centre.
    cog_envelope_fraction: float = 0.25
    # CoG check only kicks in once the pallet is at least this fraction loaded
    # (by weight). With an early/empty pallet, the CoG is naturally off-centre
    # and the check would reject every first box. Set to 0 to always enforce.
    cog_check_min_load_fraction: float = 0.35
    # Load-bearing: enforce that no box's max_load_on_top is exceeded by what
    # ends up resting (recursively) on top of it.
    enforce_load_bearing: bool = True
    # Number of multi-start trials (random seeds) for the metaheuristic.
    # The deterministic seed (heavy → volume → longest-side) hits a strong
    # local optimum that swap-perturbation rarely improves on. Empirically
    # (ANALYSIS.md / docs/reports/01_v1_analysis.md): trials=1 and trials=50
    # give identical results on every instance tested + zero seed variance.
    # Default lowered from 20 → 1 in Option B tuning (May 2026). Bump back
    # up only if you find an instance class where it actually helps.
    multi_start_trials: int = 1
    # Random seed for reproducibility.
    seed: Optional[int] = 42

    # ---- v2 improvements (opt-in) -----------------------------------------
    # Block-building preprocessor (Eley 2002, Bortfeldt 2000): identify
    # identical SKUs with ≥ block_threshold units and pack them as solid
    # rectangular blocks (n_x × n_y × n_z grids).
    use_block_building: bool = False
    block_threshold: int = 4
    # True BRKGA replacement for the multi-start sweep (Gonçalves & Resende
    # 2013). Maintains a population of chromosomes, applies elite-biased
    # crossover each generation. Set use_brkga=True to enable.
    use_brkga: bool = False
    brkga_population_size: int = 30
    brkga_generations: int = 8
    brkga_elite_fraction: float = 0.20
    brkga_mutant_fraction: float = 0.15
    brkga_p_elite: float = 0.70
    # Cap on N above which BRKGA falls back to multi-start. The original v2
    # cutoff of 40 was conservative — BRKGA was given a deterministic decoder
    # and couldn't escape the same local optimum on larger instances. With
    # GRASP randomization (grasp_alpha > 1) the decoder is now non-deterministic
    # so BRKGA can usefully explore at higher N. Set to None to disable the
    # fallback entirely.
    brkga_n_threshold: Optional[int] = 200

    # ---- Phase 2 improvements (opt-in) -----------------------------------
    # SKU-consistent rotation pre-decision (Bortfeldt-Gehring 2001).
    # For each SKU with multiple units, score each allowed rotation by the
    # number of boxes that fit per pallet layer; lock in the winner before
    # search. Cuts search space, helps cases where mixed-rotation packings
    # would fragment a structured layout.
    sku_consistent_rotation: bool = False
    # GRASP randomization in placement (Parreño et al. 2008). Within a
    # placement step, pick uniformly among the top-`grasp_alpha` best
    # candidates instead of always taking the single best. Provides BRKGA
    # / multi-start with real per-decoder variation. 1 = current behavior.
    grasp_alpha: int = 1
    # Ejection chains (Crainic-Perboli-Tadei 2009 + Faroe-Pisinger-Zachariasen
    # 2003). After greedy packing, displace one or more placed items, try to
    # fit currently-unpacked items, then re-place the displaced items.
    # Iterate until no improvement or budget exhausted.
    use_ejection_chains: bool = False
    ejection_max_depth: int = 2          # number of items to remove in one chain
    ejection_max_iters: int = 200         # outer loop budget
    # Safety net: when GRASP / BRKGA randomization is enabled, also run
    # deterministic equivalents (multi_start, deterministic-BRKGA on
    # original boxes, block-building-without-GRASP) and pick the best
    # across all candidates. This guarantees Phase 2 features can only
    # add value, never destroy a v1- or v2-baseline win. Disable for
    # benchmarking when you want to measure raw Phase 2 contribution.
    use_safety_net: bool = True
    # Cap on the number of pallets the packer is allowed to open. When set
    # to 1, this turns the multi-pallet packer into a single-container
    # max-utilization packer (the Bischoff-Ratcliff objective). Items that
    # don't fit on the capped set of pallets end up in `result.unpacked`.
    # Default None = unlimited (the original multi-pallet behavior).
    max_pallets: Optional[int] = None
    # Candidate-selection metric. Controls how the candidate set's
    # min(quality) picks between competing packings. Options:
    #   "min_unpacked"   — (unpacked, pallets, -util_overall). Default.
    #                      Best for multi-pallet logistics: pack
    #                      everything first, then minimize pallet count.
    #   "max_util"       — (-util_overall, unpacked, pallets). Best for
    #                      single-container max-utilization (the BR
    #                      objective). Prioritizes density on the
    #                      assigned pallet(s) over fitting all items.
    optimize: str = "min_unpacked"
    # Phase 2e — layer-building decoder (George-Robinson 1980;
    # Bischoff-Ratcliff 1995). Builds horizontal layer-slabs along the
    # container's longest axis. Each layer's depth is set by a seed item;
    # the slab is filled by recursively packing on a virtual sub-pallet.
    # Produces qualitatively different packings than extreme-point —
    # especially on heterogeneous mixes (BR1-7, F13).
    use_layer_building: bool = False
    # Which axis to layer along ('x' = pallet length, 'y' = width,
    # 'z' = height, or 'all' = try all three).
    # Default 'all' tries every orientation (with 2 seed strategies per
    # axis = 6 candidate packings) and picks the best — cheap because
    # each layer-decoder is one decode, ~milliseconds at small N.
    layer_axis: str = "all"
    # Phase 4 — MIP polish (do Nascimento-Queiroz-Junqueira 2021, OR-Tools
    # CP-SAT). Solve the 3D-BPP exactly (or best-within-budget) for small
    # sub-problems. Provides a "provably optimal-or-close" candidate
    # alongside the heuristics.
    use_mip_polish: bool = False
    # Maximum N for which to attempt the MIP. With Q2 (quantitative
    # support) + Q3 (warm-starting from heuristic), CP-SAT can converge
    # to OPTIMAL on N=45 in ~15s and to FEASIBLE-better-than-v1 on
    # N=40 in 30s. The default 50 is calibrated for this regime; below
    # 50, MIP almost always converges to a useful solution within
    # mip_time_limit_s.
    mip_n_threshold: int = 50
    # Wall-clock budget for the CP-SAT solve in seconds.
    mip_time_limit_s: float = 30.0
    # CP-SAT parallel workers. 1 = single-threaded — empirically much
    # faster on our model (the parallel modes seem to interfere with the
    # warm-start hint, possibly because each worker re-explores from
    # scratch). On C1 (N=45) at default time-limit: 1 worker reaches
    # OPTIMAL in 12s; 4 workers don't converge in 30s.
    mip_num_workers: int = 1

    # ---- Realism layer (opt-in; the API service turns these on) -----------
    # All three default OFF so library/benchmark callers keep bit-identical
    # behavior. See postprocess.py and docs/reports/34_realism_layer.md.
    # Rigid per-pallet x/y translation of the finished layout so the load is
    # centred on the deck (weighted CoG when weights exist, else bbox midpoint).
    recenter_layout: bool = False
    # Post-pass that re-rotates same-SKU boxes to the dominant orientation of
    # their (SKU, z-level) when the swap is feasibility-preserving.
    align_orientations: bool = False
    # Weight of the epsilon-scaled secondary "realism" fitness term
    # (heavy-low height moment + max-height + orientation consistency).
    # A dial in [0, 1]: 0.0 = off (fitness is pure volume, the historical
    # objective); values above 1.0 are clamped by build_realism_context so
    # the bounded-loss guarantee holds (the term can never cost more than
    # half the smallest box's volume, so it never causes a box to be
    # dropped).
    realism_weight: float = 0.0
    # Propagate each box's weight transitively down the whole support chain
    # when accumulating loads against max_load_on_top (the v2 engine's
    # _propagate_load model: check-direct / commit-transitive). Default OFF
    # keeps the historical direct-supporter-only accumulation in the BRKGA
    # decoders — a tall stack of individually-legal links could load the
    # bottom box far past its rated limit (hardening round 2, F19). The API
    # service turns this on.
    transitive_load_bearing: bool = False

