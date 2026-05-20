"""
pallet_packer.py
=================
3D bin packing / pallet loading with realistic physical constraints.

Algorithm
---------
- Extreme-Point placement heuristic (Crainic, Perboli & Tadei, INFORMS JoC 2008)
- Pluggable constraint stack: support / no-floating, weight limit, CoG envelope,
  restricted rotations, overhang, fragility / load-bearing, stacking compatibility.
- Multi-start randomized search (BRKGA-lite) for quality improvement.
- First-Fit-Decreasing across pallets to minimize pallet count.

References
----------
- Crainic, T.G., Perboli, G. & Tadei, R. (2008). Extreme Point-Based Heuristics
  for Three-Dimensional Bin Packing. INFORMS J. Computing 20(3), 368-384.
- Junqueira, L., Morabito, R. & Yamashita, D.S. (2012). Three-dimensional
  container loading models with cargo stability and load bearing constraints.
  Computers & Operations Research 39(1), 74-85.
- Bischoff, E.E. (2006). Three-dimensional packing of items with limited load
  bearing strength. EJOR 168(3), 952-966.
- Ramos, A.G., Silva, E. & Oliveira, J.F. (2018). A new load balance methodology
  for container loading problem in road transportation. EJOR 266(3), 1140-1152.
- Gonçalves, J.F. & Resende, M.G.C. (2013). A biased random key genetic
  algorithm for 2D and 3D bin packing problems. IJPE 145, 500-510.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, List, Optional, Tuple

EPS = 1e-6


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
    multi_start_trials: int = 20
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


# ---------------------------------------------------------------------------
# Pallet state and packing logic
# ---------------------------------------------------------------------------
class PalletState:
    """State of one pallet during packing."""

    def __init__(self, pallet: Pallet, pallet_id: str, config: PackerConfig):
        self.pallet = pallet
        self.pallet_id = pallet_id
        self.config = config
        self.placements: List[Placement] = []
        # Candidate positions for the next box's back-left-bottom corner.
        # Seeded with the floor origin; `try_place` additionally explores
        # the 3 far-corner anchors per rotation (these depend on the box's
        # current dimensions and so cannot be pre-stored in the EP list).
        self.extreme_points: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        self.total_weight: float = 0.0
        # Cached accumulated weight resting on each placement's top face.
        # Updated incrementally on _commit() so feasibility is O(N) not O(N²).
        self._top_load: dict = {}

    # ----- geometry helpers ------------------------------------------------
    def _base_overlap_area(
        self, cand: Placement, supporter: Placement
    ) -> float:
        """Area where `cand`'s bottom face rests on `supporter`'s top face."""
        if abs(cand.z - supporter.z2) > EPS:
            return 0.0
        x_overlap = max(0.0, min(cand.x2, supporter.x2) - max(cand.x, supporter.x))
        y_overlap = max(0.0, min(cand.y2, supporter.y2) - max(cand.y, supporter.y))
        return x_overlap * y_overlap

    def _supporters_of(self, cand: Placement) -> List[Tuple[Placement, float]]:
        """Return [(supporter, contact_area)] for all items cand rests on."""
        if cand.z <= EPS:
            return []  # resting on floor
        out: List[Tuple[Placement, float]] = []
        for p in self.placements:
            area = self._base_overlap_area(cand, p)
            if area > EPS:
                out.append((p, area))
        return out

    def _centroid_supported(
        self, cand: Placement, supporters: List[Tuple[Placement, float]]
    ) -> bool:
        """The footprint centroid must lie over (or touch) some supporter."""
        if cand.z <= EPS:
            return True
        cx = cand.x + cand.dx / 2.0
        cy = cand.y + cand.dy / 2.0
        for s, _ in supporters:
            if (s.x - EPS <= cx <= s.x2 + EPS and
                    s.y - EPS <= cy <= s.y2 + EPS):
                return True
        return False

    def _within_pallet(self, cand: Placement) -> bool:
        ov = self.pallet.max_overhang if self.config.allow_pallet_overhang else 0.0
        return (cand.x >= -EPS and
                cand.y >= -EPS and
                cand.z >= -EPS and
                cand.x2 <= self.pallet.length + ov + EPS and
                cand.y2 <= self.pallet.width + ov + EPS and
                cand.z2 <= self.pallet.height + EPS)

    def _collides(self, cand: Placement) -> bool:
        return any(cand.overlaps(p) for p in self.placements)

    def _load_bearing_ok(
        self, cand: Placement, supporters: List[Tuple[Placement, float]]
    ) -> bool:
        """O(N) check using the cached _top_load on each placement.

        Each placement i tracks how much weight currently rests on its top
        (the sum of (supported_weight × contact_share) from items above).
        The candidate would add `cand.box.weight × (a/total_area)` to each
        supporter; we reject if any supporter would exceed its limit.
        """
        if not supporters:
            return True
        total_area = sum(a for _, a in supporters)
        if total_area <= 0:
            return True
        for s, a in supporters:
            added = cand.box.weight * (a / total_area)
            current = self._top_load.get(id(s), 0.0)
            if current + added > s.box.max_load_on_top + EPS:
                return False
        return True

    def _cog_after(self, cand: Placement) -> Tuple[float, float]:
        """Pallet (x, y) CoG if `cand` is added."""
        tot_w = self.total_weight + cand.box.weight
        if tot_w <= 0:
            return self.pallet.length / 2.0, self.pallet.width / 2.0
        sx = sum(p.box.weight * (p.x + p.dx / 2.0) for p in self.placements)
        sy = sum(p.box.weight * (p.y + p.dy / 2.0) for p in self.placements)
        sx += cand.box.weight * (cand.x + cand.dx / 2.0)
        sy += cand.box.weight * (cand.y + cand.dy / 2.0)
        return sx / tot_w, sy / tot_w

    def _cog_ok(self, cand: Placement) -> bool:
        if cand.box.weight <= 0 and self.total_weight <= 0:
            return True
        # CoG check only makes sense after we have prior items to balance
        # against; with a single box in a corner the check is unsatisfiable.
        if not self.placements:
            return True
        new_total = self.total_weight + cand.box.weight
        max_w = self.pallet.max_weight if self.pallet.max_weight < float("inf") else None
        if max_w is not None:
            if new_total < self.config.cog_check_min_load_fraction * max_w:
                return True
        cx, cy = self._cog_after(cand)
        L, W = self.pallet.length, self.pallet.width
        frac = self.config.cog_envelope_fraction
        x_range = self.pallet.cog_x_range or (
            L / 2.0 - frac * L, L / 2.0 + frac * L
        )
        y_range = self.pallet.cog_y_range or (
            W / 2.0 - frac * W, W / 2.0 + frac * W
        )
        return (x_range[0] - EPS <= cx <= x_range[1] + EPS and
                y_range[0] - EPS <= cy <= y_range[1] + EPS)

    # ----- feasibility ----------------------------------------------------
    def feasible(self, cand: Placement) -> bool:
        """Run the full constraint stack on a candidate placement."""
        # 1. Geometry
        if not self._within_pallet(cand):
            return False
        if self._collides(cand):
            return False
        # 2. Weight budget
        if self.total_weight + cand.box.weight > self.pallet.max_weight + EPS:
            return False
        # 3. Support / no-floating
        supporters = self._supporters_of(cand)
        if cand.z > EPS:
            footprint = cand.dx * cand.dy
            supported = sum(a for _, a in supporters)
            # Super-blocks demand full support so that internal decomposition
            # doesn't strand individual sub-boxes with insufficient base contact.
            min_support = 1.0 if cand.box.requires_full_support else self.config.support_ratio
            if footprint <= 0 or supported / footprint < min_support - EPS:
                return False
            if self.config.require_centroid_supported and not self._centroid_supported(cand, supporters):
                return False
        # 4. Load bearing
        if self.config.enforce_load_bearing and not self._load_bearing_ok(cand, supporters):
            return False
        # 5. CoG envelope
        if not self._cog_ok(cand):
            return False
        return True

    # ----- placement ------------------------------------------------------
    # Placement scoring strategies (the inner loop's tie-breaker over EPs).
    # 'blb'  - back-left-bottom: sort by (z, y, x)        (default)
    # 'bbl'  - bottom-back-left: sort by (z, x, y)
    # 'max_touch' - prefer placements with the largest support contact
    # 'corner_fit' - prefer placements where the box hugs a corner / wall
    SCORING_STRATEGIES = ("blb", "bbl", "max_touch", "corner_fit")

    def try_place(self, box: Box, strategy: str = "blb") -> bool:
        """Try each (extreme point, rotation) pair; place at the best feasible.

        In addition to the discovered extreme points, we always try the four
        floor-corner anchors per rotation. This is essential for balanced
        loading: pure EP heuristics can only place adjacent to existing boxes
        and so will never reach diagonally-opposite placements on their own,
        which CoG envelope constraints often require.
        """
        best: Optional[Tuple[Placement, Tuple]] = None
        candidates: List[Tuple[float, float, float]] = list(self.extreme_points)
        L, W = self.pallet.length, self.pallet.width

        for rot in box.allowed_rotations:
            dx, dy, _dz = box.dims_for(rot)
            # Floor-level corner anchors for this rotation.
            corner_anchors = [
                (0.0, 0.0, 0.0),
                (max(0.0, L - dx), 0.0, 0.0),
                (0.0, max(0.0, W - dy), 0.0),
                (max(0.0, L - dx), max(0.0, W - dy), 0.0),
            ]
            for ep in candidates + corner_anchors:
                cand = Placement(box=box, rotation=rot, x=ep[0], y=ep[1], z=ep[2])
                if not self.feasible(cand):
                    continue
                score = self._score_placement(cand, strategy)
                if best is None or score < best[1]:
                    best = (cand, score)
        if best is None:
            return False
        self._commit(best[0])
        return True

    def _score_placement(self, cand: Placement, strategy: str) -> Tuple:
        """Lower score = better. Strategies emphasise different geometric goals."""
        if strategy == "blb":
            return (cand.z, cand.y, cand.x)
        if strategy == "bbl":
            return (cand.z, cand.x, cand.y)
        if strategy == "max_touch":
            # Touch = contact area with supporters + walls (negative so larger=better).
            touch = 0.0
            sups = self._supporters_of(cand)
            touch += sum(a for _, a in sups)
            # Add wall touches.
            if cand.x <= EPS: touch += cand.dy * cand.dz
            if cand.y <= EPS: touch += cand.dx * cand.dz
            if abs(cand.x2 - self.pallet.length) <= EPS: touch += cand.dy * cand.dz
            if abs(cand.y2 - self.pallet.width) <= EPS: touch += cand.dx * cand.dz
            return (-touch, cand.z, cand.y, cand.x)
        if strategy == "corner_fit":
            # Count walls/floors the box hugs.
            hugs = 0
            if cand.x <= EPS: hugs += 1
            if cand.y <= EPS: hugs += 1
            if cand.z <= EPS: hugs += 1
            if abs(cand.x2 - self.pallet.length) <= EPS: hugs += 1
            if abs(cand.y2 - self.pallet.width) <= EPS: hugs += 1
            return (-hugs, cand.z, cand.y, cand.x)
        # Fallback
        return (cand.z, cand.y, cand.x)

    def _commit(self, cand: Placement) -> None:
        # Update the cached top-load on supporters BEFORE appending,
        # so we don't accidentally include the new box as its own supporter.
        sups = self._supporters_of(cand)
        sup_area = sum(a for _, a in sups)
        if sup_area > 0:
            # Direct supporters carry cand's own weight, in proportion.
            for s, a in sups:
                share = cand.box.weight * (a / sup_area)
                self._top_load[id(s)] = self._top_load.get(id(s), 0.0) + share
                # Transitive: cand's weight also flows down through whatever
                # the supporters themselves rest on. A simple recursive walk
                # handles arbitrarily-stacked towers; the visit set keeps it
                # linear in the support DAG.
                self._propagate_load(s, share, set())

        self.placements.append(cand)
        self.total_weight += cand.box.weight
        self._top_load.setdefault(id(cand), 0.0)
        # Remove the consumed EP if exactly at cand's origin.
        used = (cand.x, cand.y, cand.z)
        self.extreme_points = [
            ep for ep in self.extreme_points
            if not (abs(ep[0] - used[0]) < EPS and
                    abs(ep[1] - used[1]) < EPS and
                    abs(ep[2] - used[2]) < EPS)
        ]
        # Generate three new EPs at the placed box's "open" corners.
        new_eps = [
            (cand.x2, cand.y, cand.z),  # right of box
            (cand.x, cand.y2, cand.z),  # behind box
            (cand.x, cand.y, cand.z2),  # on top of box
        ]
        for ep in new_eps:
            ep_dropped = self._gravity_project(ep)
            if ep_dropped not in self.extreme_points:
                self.extreme_points.append(ep_dropped)
        # Prune EPs strictly inside placed boxes — they can never be used.
        self.extreme_points = [
            ep for ep in self.extreme_points
            if not self._point_inside_any_placement(ep)
        ]

    def _point_inside_any_placement(
        self, ep: Tuple[float, float, float]
    ) -> bool:
        x, y, z = ep
        for p in self.placements:
            if (p.x - EPS < x < p.x2 - EPS and
                    p.y - EPS < y < p.y2 - EPS and
                    p.z - EPS < z < p.z2 - EPS):
                return True
        return False

    def _propagate_load(self, placement: Placement, weight: float,
                        visited: set) -> None:
        """Distribute `weight` recursively to whatever `placement` rests on."""
        key = id(placement)
        if key in visited:
            return
        visited.add(key)
        sups = self._supporters_of(placement)
        total_area = sum(a for _, a in sups)
        if total_area <= 0:
            return
        for s, a in sups:
            share = weight * (a / total_area)
            self._top_load[id(s)] = self._top_load.get(id(s), 0.0) + share
            self._propagate_load(s, share, visited)

    def _gravity_project(
        self, ep: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        """Drop an EP straight down until it sits on something or the floor.

        Uses a tiny test footprint (a point) so we don't lose granularity.
        """
        x, y, z = ep
        best_z = 0.0
        for p in self.placements:
            if (p.x - EPS <= x <= p.x2 + EPS and
                    p.y - EPS <= y <= p.y2 + EPS and
                    p.z2 <= z + EPS):
                if p.z2 > best_z:
                    best_z = p.z2
        return (x, y, best_z)


# ---------------------------------------------------------------------------
# Multi-pallet orchestration + multi-start search
# ---------------------------------------------------------------------------
@dataclass
class PackResult:
    pallets: List[PalletState]
    unpacked: List[Box]

    @property
    def num_pallets(self) -> int:
        return len(self.pallets)

    @property
    def total_volume_utilisation(self) -> float:
        used = sum(p.box.volume for st in self.pallets for p in st.placements)
        cap = sum(st.pallet.length * st.pallet.width * st.pallet.height
                  for st in self.pallets)
        return used / cap if cap > 0 else 0.0


class PalletPacker:
    """Top-level packer: multi-start search over box orderings."""

    def __init__(self, pallet: Pallet, config: Optional[PackerConfig] = None):
        self.pallet = pallet
        self.config = config or PackerConfig()
        self._rng = random.Random(self.config.seed)

    # -------- block-building (Eley 2002 / Bortfeldt 2000) -----------------
    def _group_identical_boxes(self, boxes: List[Box]) -> dict:
        """Group boxes that are interchangeable (same dims, weight, properties)."""
        groups: dict = {}
        for b in boxes:
            key = (b.length, b.width, b.height, b.weight, b.max_load_on_top,
                   tuple(sorted(r.name for r in b.allowed_rotations)),
                   b.group)
            groups.setdefault(key, []).append(b)
        return groups

    def _build_blocks(
        self, boxes: List[Box], strategy: str = "max"
    ) -> Tuple[List[Box], List[Tuple]]:
        """Greedy: form super-blocks from identical groups; remainders pass through.

        strategy:
            'max'    - one large block per group (max-units, non-spanning preferred)
            'column' - tall thin columns (1×1×n_z) — smallest footprint per block
            'layer'  - flat layers (n_x×n_y×1) — minimal vertical footprint

        Returns (items_for_packing, block_specs).
        """
        groups = self._group_identical_boxes(boxes)
        items: List[Box] = []
        specs: List[Tuple] = []
        counter = 0
        for _key, group_boxes in groups.items():
            remaining = list(group_boxes)
            while len(remaining) >= self.config.block_threshold:
                best = self._find_best_block(remaining[0], len(remaining),
                                             strategy=strategy)
                if best is None:
                    break
                _score, rot, nx, ny, nz = best
                size = nx * ny * nz
                if size < self.config.block_threshold:
                    break
                members = remaining[:size]
                remaining = remaining[size:]
                tmpl = members[0]
                dx, dy, dz = tmpl.dims_for(rot)
                block_id = f"__BLOCK_{counter:03d}"
                counter += 1
                super_box = Box(
                    id=block_id,
                    length=nx * dx,
                    width=ny * dy,
                    height=nz * dz,
                    weight=sum(b.weight for b in members),
                    max_load_on_top=tmpl.max_load_on_top,
                    allowed_rotations=NO_ROTATION,
                    requires_full_support=True,
                )
                items.append(super_box)
                specs.append((super_box, tmpl, rot, nx, ny, nz,
                              [b.id for b in members]))
            items.extend(remaining)
        return items, specs

    def _find_best_block(
        self, template: Box, K: int, strategy: str = "max"
    ) -> Optional[Tuple[Tuple, Rotation, int, int, int]]:
        """Find the largest valid (rotation, n_x, n_y, n_z) block.

        Scoring varies by strategy — each gives different residual-space shapes.
        Blocks are also constrained so total block weight ≤ pallet weight
        capacity; an over-weight block could never be placed and would silently
        strand its member boxes as unpacked.
        """
        best = None
        L, W = self.pallet.length, self.pallet.width
        # Cap on units per block by total weight (any block must fit on a pallet).
        max_units_by_weight = K
        if template.weight > 0 and self.pallet.max_weight < float("inf"):
            max_units_by_weight = max(1, int(self.pallet.max_weight // template.weight))
            if max_units_by_weight < self.config.block_threshold:
                # Even a minimum-size block would be over-weight — skip blocks
                # for this group entirely.
                return None
        for rot in template.allowed_rotations:
            dx, dy, dz = template.dims_for(rot)
            if dx <= 0 or dy <= 0 or dz <= 0:
                continue
            max_nx = int(L // dx)
            max_ny = int(W // dy)
            max_nz = int(self.pallet.height // dz)
            if template.weight > 0 and template.max_load_on_top < float("inf"):
                max_stack_load = int(template.max_load_on_top / template.weight) + 1
                max_nz = min(max_nz, max_stack_load)
            max_nx = min(max_nx, K, max_units_by_weight)
            max_ny = min(max_ny, K, max_units_by_weight)
            max_nz = min(max_nz, K, max_units_by_weight)
            if max_nx < 1 or max_ny < 1 or max_nz < 1:
                continue

            if strategy == "column":
                # Pure vertical column: (1, 1, max_nz). Smallest footprint.
                if max_nz >= 2:
                    total = max_nz  # nx=1, ny=1, nz=max_nz
                    if total <= K and total <= max_units_by_weight:
                        score = (total, max_nz, -dx * dy)
                        if best is None or score > best[0]:
                            best = (score, rot, 1, 1, max_nz)
                continue

            if strategy == "layer":
                # Floor layer: (max_nx, max_ny, 1). Smallest vertical footprint.
                for nx in range(max_nx, 0, -1):
                    for ny in range(max_ny, 0, -1):
                        t = nx * ny
                        if t <= K and t <= max_units_by_weight:
                            score = (t, -dx * nx * dy * ny)
                            if best is None or score > best[0]:
                                best = (score, rot, nx, ny, 1)
                            break
                continue

            # strategy == 'max': largest block with non-spanning preference.
            for nx in range(max_nx, 0, -1):
                for ny in range(max_ny, 0, -1):
                    for nz in range(max_nz, 0, -1):
                        total = nx * ny * nz
                        if total <= K and total <= max_units_by_weight:
                            block_l = nx * dx
                            block_w = ny * dy
                            block_fp = block_l * block_w
                            spans = (block_l >= L - EPS and block_w >= W - EPS)
                            score = (total, 0 if spans else 1, nz, -block_fp)
                            if best is None or score > best[0]:
                                best = (score, rot, nx, ny, nz)
                            break
        return best

    def _expand_blocks(
        self, result: PackResult, specs: List[Tuple]
    ) -> PackResult:
        """Replace each super-block placement with its grid of individual placements."""
        spec_by_id = {s[0].id: s for s in specs}
        new_pallets: List[PalletState] = []
        for st in result.pallets:
            expanded: List[Placement] = []
            for p in st.placements:
                if p.box.id not in spec_by_id:
                    expanded.append(p)
                    continue
                _sb, tmpl, rot, nx, ny, nz, member_ids = spec_by_id[p.box.id]
                dx, dy, dz = tmpl.dims_for(rot)
                idx = 0
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            original = Box(
                                id=member_ids[idx],
                                length=tmpl.length,
                                width=tmpl.width,
                                height=tmpl.height,
                                weight=tmpl.weight,
                                max_load_on_top=tmpl.max_load_on_top,
                                allowed_rotations=tmpl.allowed_rotations,
                            )
                            expanded.append(Placement(
                                box=original,
                                rotation=rot,
                                x=p.x + i * dx,
                                y=p.y + j * dy,
                                z=p.z + k * dz,
                            ))
                            idx += 1
            new_st = PalletState(st.pallet, st.pallet_id, st.config)
            new_st.placements = expanded
            new_st.total_weight = st.total_weight
            new_pallets.append(new_st)
        # Note: unpacked super-blocks would mean the whole block didn't fit;
        # in that case we expand those too so the user sees individual ids.
        unpacked_expanded: List[Box] = []
        for b in result.unpacked:
            if b.id in spec_by_id:
                _sb, tmpl, _rot, _nx, _ny, _nz, member_ids = spec_by_id[b.id]
                for mid in member_ids:
                    unpacked_expanded.append(Box(
                        id=mid,
                        length=tmpl.length, width=tmpl.width, height=tmpl.height,
                        weight=tmpl.weight,
                        max_load_on_top=tmpl.max_load_on_top,
                        allowed_rotations=tmpl.allowed_rotations,
                    ))
            else:
                unpacked_expanded.append(b)
        return PackResult(pallets=new_pallets, unpacked=unpacked_expanded)

    # -------- single trial -----------------------------------------------
    def _pack_once(
        self,
        boxes: List[Box],
        order: List[int],
        strategy: str = "blb",
        pallet_selection: str = "best_fit",
    ) -> PackResult:
        """Pack boxes (in the given order) onto as few pallets as possible.

        pallet_selection:
            'first_fit' - place into the first pallet where the box fits.
            'best_fit'  - try every open pallet, pick the one with the highest
                          current utilisation that still accepts the box.
        """
        ordered = [boxes[i] for i in order]
        pallets: List[PalletState] = []
        unpacked: List[Box] = []

        def used_volume(st: PalletState) -> float:
            return sum(p.box.volume for p in st.placements)

        for box in ordered:
            placed = False
            if pallet_selection == "first_fit":
                for st in pallets:
                    if st.try_place(box, strategy=strategy):
                        placed = True
                        break
            else:  # best_fit: pack into the fullest pallet that accepts it.
                candidates = sorted(pallets, key=used_volume, reverse=True)
                for st in candidates:
                    if st.try_place(box, strategy=strategy):
                        placed = True
                        break
            if not placed:
                st = PalletState(self.pallet,
                                 pallet_id=f"P{len(pallets) + 1:03d}",
                                 config=self.config)
                if st.try_place(box, strategy=strategy):
                    pallets.append(st)
                    placed = True
                else:
                    unpacked.append(box)
        return PackResult(pallets=pallets, unpacked=unpacked)

    # -------- multi-start orchestrator ------------------------------------
    def pack(self, boxes: List[Box]) -> PackResult:
        """Pack with optional block-building and BRKGA improvements.

        When block-building is enabled, we try multiple decomposition
        strategies (no-blocks, max-blocks, column-blocks, layer-blocks)
        and return the best across all of them.
        """
        def search(items: List[Box]) -> PackResult:
            if self.config.use_brkga:
                return self._brkga_search(items)
            return self._multi_start(items)

        def quality(res: PackResult) -> Tuple:
            # Unpacked items are the worst outcome — rank them first.
            # A solution that packs everything in 5 pallets must beat a
            # solution that leaves items unpacked in 0 pallets (a "0 pallets"
            # result with unpacked items is a failed packing, not a win).
            return (len(res.unpacked), res.num_pallets,
                    -res.total_volume_utilisation)

        candidates: List[PackResult] = []
        # Baseline: search without block-building (always tried).
        candidates.append(search(boxes))

        # Optional: try each block-building strategy.
        if self.config.use_block_building:
            for strategy in ("max", "column", "layer"):
                items, specs = self._build_blocks(boxes, strategy=strategy)
                if specs:
                    raw = search(items)
                    candidates.append(self._expand_blocks(raw, specs))

        return min(candidates, key=quality)

    # -------- multi-start search (v1) -------------------------------------
    def _multi_start(self, boxes: List[Box]) -> PackResult:
        # Seed ordering: heaviest first, then largest volume, then longest side.
        def base_key(b: Box) -> Tuple[float, float, float]:
            w = -b.weight if self.config.heavy_on_bottom else 0.0
            return (w, -b.volume, -max(b.length, b.width, b.height))

        n = len(boxes)
        indices = list(range(n))
        seed_order = sorted(indices, key=lambda i: base_key(boxes[i]))

        trials: List[Tuple[List[int], str, str]] = []
        strategies = PalletState.SCORING_STRATEGIES
        selections = ("best_fit", "first_fit")

        for strat in strategies:
            for sel in selections:
                trials.append((list(seed_order), strat, sel))

        random_trials_needed = max(0, self.config.multi_start_trials - len(trials))
        for _ in range(random_trials_needed):
            perm = list(seed_order)
            if n >= 2:
                for _ in range(max(2, n // 5)):
                    a, b = self._rng.sample(range(n), 2)
                    perm[a], perm[b] = perm[b], perm[a]
            strat = self._rng.choice(strategies)
            sel = self._rng.choice(selections)
            trials.append((perm, strat, sel))

        best_result: Optional[PackResult] = None

        def quality_score(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets, -res.total_volume_utilisation)

        for order, strat, sel in trials:
            res = self._pack_once(boxes, order, strategy=strat, pallet_selection=sel)
            if best_result is None or quality_score(res) < quality_score(best_result):
                best_result = res

        assert best_result is not None
        return best_result

    # -------- BRKGA search (v2) -------------------------------------------
    def _brkga_search(self, boxes: List[Box]) -> PackResult:
        """True BRKGA with population recombination (Gonçalves & Resende 2013).

        Chromosome: (N + 2) random keys.
            keys[0:N] — box order keys (sort indices by these)
            keys[N]   — selects a placement strategy
            keys[N+1] — selects pallet-selection mode (best/first fit)
        """
        n = len(boxes)
        if n == 0:
            return PackResult(pallets=[], unpacked=[])

        strategies = PalletState.SCORING_STRATEGIES
        selections = ("best_fit", "first_fit")
        P = max(4, self.config.brkga_population_size)
        n_elite = max(1, int(P * self.config.brkga_elite_fraction))
        n_mutant = max(1, int(P * self.config.brkga_mutant_fraction))
        n_cross = P - n_elite - n_mutant
        p_e = self.config.brkga_p_elite

        def random_chromosome() -> List[float]:
            return [self._rng.random() for _ in range(n + 2)]

        def decode(chrom: List[float]) -> PackResult:
            order = sorted(range(n), key=lambda i: chrom[i])
            strat = strategies[int(chrom[n] * len(strategies)) % len(strategies)]
            sel = selections[int(chrom[n + 1] * len(selections)) % len(selections)]
            return self._pack_once(boxes, order, strategy=strat, pallet_selection=sel)

        def fitness(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets, -res.total_volume_utilisation)

        # Seed the population with the deterministic heavy/big-first ordering
        # encoded as monotonic keys (so the BRKGA at worst matches multi-start).
        def base_key(b: Box) -> Tuple[float, float, float]:
            w = -b.weight if self.config.heavy_on_bottom else 0.0
            return (w, -b.volume, -max(b.length, b.width, b.height))

        seed_order = sorted(range(n), key=lambda i: base_key(boxes[i]))
        seed_chrom = [0.0] * (n + 2)
        for rank, idx in enumerate(seed_order):
            seed_chrom[idx] = rank / max(1, n - 1)
        seed_chrom[n] = 0.0      # 'blb' strategy
        seed_chrom[n + 1] = 0.0  # 'best_fit' selection

        population: List[List[float]] = [seed_chrom]
        for _ in range(P - 1):
            population.append(random_chromosome())

        best_result: Optional[PackResult] = None
        for gen in range(self.config.brkga_generations):
            scored = []
            for c in population:
                r = decode(c)
                scored.append((fitness(r), c, r))
            scored.sort(key=lambda t: t[0])

            # Track best.
            if best_result is None or scored[0][0] < fitness(best_result):
                best_result = scored[0][2]

            # Build next generation.
            new_pop: List[List[float]] = [t[1] for t in scored[:n_elite]]
            non_elites = [t[1] for t in scored[n_elite:]]
            # Crossover offspring (elite × non-elite, biased to elite).
            for _ in range(n_cross):
                if not non_elites:
                    new_pop.append(random_chromosome())
                    continue
                pa = self._rng.choice(new_pop[:n_elite])
                pb = self._rng.choice(non_elites)
                child = [pa[i] if self._rng.random() < p_e else pb[i]
                         for i in range(n + 2)]
                new_pop.append(child)
            # Fresh mutants.
            for _ in range(n_mutant):
                new_pop.append(random_chromosome())
            population = new_pop

        assert best_result is not None
        return best_result

    # -------- single trial -----------------------------------------------


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------
def to_json(result: PackResult, pallet: Pallet) -> dict:
    """Serialize a PackResult to the JSON schema agreed in the design report."""
    pallets_out = []
    for st in result.pallets:
        # Identify supporters for each placement (for downstream visualisers).
        items_out = []
        for p in st.placements:
            sups = st._supporters_of(p)
            supported_by = [s.box.id for s, _ in sups] if sups else ["floor"]
            supports = [
                q.box.id for q in st.placements
                if any(abs(q.z - p.z2) < EPS and
                       max(0, min(q.x2, p.x2) - max(q.x, p.x)) *
                       max(0, min(q.y2, p.y2) - max(q.y, p.y)) > EPS
                       for _ in [None])
            ]
            footprint = p.dx * p.dy
            support_ratio = (
                sum(a for _, a in sups) / footprint
                if footprint > 0 and p.z > EPS else 1.0
            )
            items_out.append({
                "item_id": p.box.id,
                "position": {"x": round(p.x, 3),
                             "y": round(p.y, 3),
                             "z": round(p.z, 3)},
                "dimensions": {"L": round(p.dx, 3),
                               "W": round(p.dy, 3),
                               "H": round(p.dz, 3)},
                "orientation": {"perm": list(p.rotation.value),
                                "name": p.rotation.name},
                "weight": p.box.weight,
                "support_ratio": round(support_ratio, 4),
                "supported_by": supported_by,
                "supports": supports,
            })
        # Pallet CoG.
        if st.total_weight > 0:
            cx = sum(p.box.weight * (p.x + p.dx / 2.0) for p in st.placements) / st.total_weight
            cy = sum(p.box.weight * (p.y + p.dy / 2.0) for p in st.placements) / st.total_weight
            cz = sum(p.box.weight * (p.z + p.dz / 2.0) for p in st.placements) / st.total_weight
        else:
            cx = cy = cz = 0.0
        used_volume = sum(p.box.volume for p in st.placements)
        capacity = pallet.length * pallet.width * pallet.height
        pallets_out.append({
            "pallet_id": st.pallet_id,
            "dimensions": {"L": pallet.length, "W": pallet.width,
                           "H": pallet.height, "max_weight": pallet.max_weight},
            "utilisation": round(used_volume / capacity if capacity > 0 else 0.0, 4),
            "cog": {"x": round(cx, 3), "y": round(cy, 3), "z": round(cz, 3)},
            "total_weight": round(st.total_weight, 3),
            "items": items_out,
        })
    return {
        "input_summary": {
            "items_packed": sum(len(p["items"]) for p in pallets_out),
            "items_unpacked": len(result.unpacked),
            "pallets_used": result.num_pallets,
            "total_volume_utilisation": round(result.total_volume_utilisation, 4),
        },
        "pallets": pallets_out,
        "unpacked_items": [
            {"item_id": b.id,
             "dimensions": {"L": b.length, "W": b.width, "H": b.height},
             "weight": b.weight,
             "reason": "no_feasible_placement"}
            for b in result.unpacked
        ],
    }


def save_json(result: PackResult, pallet: Pallet, path: str) -> None:
    with open(path, "w") as f:
        json.dump(to_json(result, pallet), f, indent=2)


# ---------------------------------------------------------------------------
# Validator: independent sanity check of a packing result
# ---------------------------------------------------------------------------
def validate(result: PackResult, pallet: Pallet,
             config: Optional[PackerConfig] = None) -> List[str]:
    """Return a list of constraint violations (empty list means all good).

    Re-checks the produced packing against geometry, weight, support, and
    rotation constraints — independent of the placement engine, so any bug
    in the engine should produce a non-empty list here.
    """
    cfg = config or PackerConfig()
    errors: List[str] = []
    for st in result.pallets:
        # 1. Pallet boundaries
        for p in st.placements:
            ov = pallet.max_overhang if cfg.allow_pallet_overhang else 0.0
            if (p.x < -EPS or p.y < -EPS or p.z < -EPS or
                    p.x2 > pallet.length + ov + EPS or
                    p.y2 > pallet.width + ov + EPS or
                    p.z2 > pallet.height + EPS):
                errors.append(
                    f"{st.pallet_id}: {p.box.id} outside pallet bounds "
                    f"at ({p.x},{p.y},{p.z})+({p.dx},{p.dy},{p.dz})"
                )
        # 2. No pairwise overlap
        for i, a in enumerate(st.placements):
            for b in st.placements[i + 1:]:
                if a.overlaps(b):
                    errors.append(
                        f"{st.pallet_id}: {a.box.id} overlaps {b.box.id}"
                    )
        # 3. Weight budget
        total = sum(p.box.weight for p in st.placements)
        if total > pallet.max_weight + EPS:
            errors.append(
                f"{st.pallet_id}: total weight {total} > limit {pallet.max_weight}"
            )
        # 4. Support / no-floating
        for p in st.placements:
            if p.z <= EPS:
                continue
            footprint = p.dx * p.dy
            supported = 0.0
            for q in st.placements:
                if q is p:
                    continue
                if abs(p.z - q.z2) > EPS:
                    continue
                ox = max(0.0, min(p.x2, q.x2) - max(p.x, q.x))
                oy = max(0.0, min(p.y2, q.y2) - max(p.y, q.y))
                supported += ox * oy
            if footprint > 0 and supported / footprint < cfg.support_ratio - EPS:
                errors.append(
                    f"{st.pallet_id}: {p.box.id} is floating "
                    f"(support ratio {supported / footprint:.2f} < {cfg.support_ratio})"
                )
        # 5. Rotation allowed
        for p in st.placements:
            if p.rotation not in p.box.allowed_rotations:
                errors.append(
                    f"{st.pallet_id}: {p.box.id} uses disallowed rotation "
                    f"{p.rotation.name}"
                )
        # 6. Load bearing (per direct supporter, weighted by contact area)
        if cfg.enforce_load_bearing:
            load_on: dict = {id(p): 0.0 for p in st.placements}
            for placed in st.placements:
                sups = []
                for q in st.placements:
                    if q is placed:
                        continue
                    if abs(placed.z - q.z2) > EPS:
                        continue
                    ox = max(0.0, min(placed.x2, q.x2) - max(placed.x, q.x))
                    oy = max(0.0, min(placed.y2, q.y2) - max(placed.y, q.y))
                    if ox * oy > EPS:
                        sups.append((q, ox * oy))
                sup_area = sum(a for _, a in sups)
                if sup_area <= 0:
                    continue
                for q, a in sups:
                    load_on[id(q)] += placed.box.weight * (a / sup_area)
            for q in st.placements:
                if load_on[id(q)] > q.box.max_load_on_top + EPS:
                    errors.append(
                        f"{st.pallet_id}: {q.box.id} carries "
                        f"{load_on[id(q)]:.2f} kg > max_load_on_top "
                        f"{q.box.max_load_on_top}"
                    )
    return errors
