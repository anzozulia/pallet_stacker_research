"""
pallet_packer.packer — the core algorithm.

Contains:
  - PalletState: per-pallet engine with extreme-point placement, the full
    constraint stack (geometry, support, weight, CoG, fragility, rotation,
    overhang), and the GRASP-aware try_place primitive.
  - PackResult: the multi-pallet packing result returned from pack().
  - PalletPacker: the top-level orchestrator. Runs multi-start search,
    BRKGA, block-building, layer-building, ejection chains, MIP polish, and
    the safety-net candidate generation. Final answer is min(candidates)
    under the configured quality function.

References live in the original module docstring; key ones:
  - Crainic, Perboli & Tadei 2008 (extreme points)
  - Gonçalves & Resende 2013 (BRKGA)
  - Bischoff & Ratcliff 1995 (layer building)
  - Eley 2002 / Bortfeldt 2000 (block building)
  - do Nascimento, Queiroz & Junqueira 2021 (MIP polish)
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from .models import (
    EPS,
    load_tol,
    Box,
    Pallet,
    Placement,
    PackerConfig,
    Rotation,
    ALL_ROTATIONS,
    THIS_SIDE_UP,
    NO_ROTATION,
)


def _adaptive_mip_budget(n_items: int, default_budget: float) -> float:
    """Scale MIP time budget by problem size.

    Small problems converge to OPTIMAL in well under a second; giving them
    the full 30s default is pure waste in batch settings. Large problems
    (near the mip_n_threshold) get the full budget. Never returns more than
    `default_budget` — respects user overrides for hard instances.

    Empirical calibration from the post-Phase-4 evaluation (with 1-worker
    CP-SAT + warm-start):
      N < 10  → typically <1s; 5s budget is more than enough.
      N < 25  → typically 1-10s; 15s budget covers the long tail.
      N ≥ 25  → use the full default budget.
    """
    if n_items < 10:
        return min(5.0, default_budget)
    if n_items < 25:
        return min(15.0, default_budget)
    return default_budget


# ---------------------------------------------------------------------------
# Per-pallet state and packing logic
# ---------------------------------------------------------------------------
class PalletState:
    """State of one pallet during packing."""

    def __init__(self, pallet: Pallet, pallet_id: str, config: PackerConfig,
                 rng: Optional[random.Random] = None):
        self.pallet = pallet
        self.pallet_id = pallet_id
        self.config = config
        # RNG for GRASP randomization. If None, a fresh seeded RNG is used —
        # but for reproducibility callers should pass in the packer's RNG.
        self._rng = rng if rng is not None else random.Random(config.seed)
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

    def _rider_inflow(self, cand: Placement) -> float:
        """Load the candidate would INHERIT by becoming a new supporter of
        already-placed boxes (round 3, F21 "under-fill").

        A box placed with its top exactly at an existing box's bottom,
        overlapping it in XY, physically takes a contact-share of that
        rider's outflow — the checks below the candidate never see it, so
        it must be checked (feasible) and booked (commit) explicitly.
        Per rider R: T_old = R's existing supporter contact, a = contact
        with cand, share = out_R * a / (T_old + a) with out_R = R's weight
        plus the load already resting on R (the _top_load cache is
        transitive — v2 has always committed transitively — so the
        inherited amount uses the same semantics as the cache it lands in).
        Old supporters are deliberately NOT debited: strictly conservative,
        no negative propagation, no float dust.
        """
        inherited = 0.0
        for r in self.placements:
            if abs(r.z - cand.z2) > EPS:
                continue
            ox = min(cand.x2, r.x2) - max(cand.x, r.x)
            oy = min(cand.y2, r.y2) - max(cand.y, r.y)
            if ox <= EPS or oy <= EPS:
                continue
            a = ox * oy
            t_old = sum(ar for _, ar in self._supporters_of(r))
            out_r = r.box.weight + self._top_load.get(id(r), 0.0)
            if out_r <= 0.0:
                continue
            inherited += out_r * (a / (t_old + a))
        return inherited

    def _load_bearing_ok(
        self, cand: Placement, supporters: List[Tuple[Placement, float]],
        extra_load: float = 0.0,
    ) -> bool:
        """O(N) check using the cached _top_load on each placement.

        Each placement i tracks how much weight currently rests on its top
        (the sum of (supported_weight × contact_share) from items above).
        The candidate would add `flow_w × (a/total_area)` to each
        supporter; we reject if any supporter would exceed its limit.
        `extra_load` is the rider inflow the candidate inherits by becoming
        a new supporter of existing boxes (F21) — it flows down with the
        candidate's own weight.
        """
        if not supporters:
            return True
        total_area = sum(a for _, a in supporters)
        if total_area <= 0:
            return True
        flow_w = cand.box.weight + extra_load
        for s, a in supporters:
            added = flow_w * (a / total_area)
            current = self._top_load.get(id(s), 0.0)
            if current + added > s.box.max_load_on_top + load_tol(s.box.max_load_on_top):
                return False
        if getattr(self.config, "transitive_load_bearing", False):
            # F19 (hardening round 2): the direct check above never rejects
            # a FRESH pure column — each new box's direct supporter carries
            # only its immediate rider. Dry-run the transitive flow the
            # commit (_propagate_load) would push and test every box in the
            # downward chain against its limit.
            inflow: dict = {}
            for s, a in supporters:
                inflow[id(s)] = (inflow.get(id(s), 0.0)
                                 + flow_w * (a / total_area))
            frontier = sorted((s for s, _ in supporters),
                              key=lambda p: (-p.z, p.x, p.y))
            seen = {id(s) for s, _ in supporters}
            while frontier:
                p = frontier.pop(0)
                inc = inflow.get(id(p), 0.0)
                if (self._top_load.get(id(p), 0.0) + inc
                        > p.box.max_load_on_top
                        + load_tol(p.box.max_load_on_top)):
                    return False
                if inc <= 0 or p.z <= EPS:
                    continue
                subs = self._supporters_of(p)
                sub_area = sum(a for _, a in subs)
                if sub_area <= 0:
                    continue
                for s2, a2 in subs:
                    inflow[id(s2)] = (inflow.get(id(s2), 0.0)
                                      + inc * (a2 / sub_area))
                    if id(s2) not in seen:
                        seen.add(id(s2))
                        frontier.append(s2)
                frontier.sort(key=lambda p2: (-p2.z, p2.x, p2.y))
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
        if (self.total_weight + cand.box.weight
                > self.pallet.max_weight + load_tol(self.pallet.max_weight)):
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
        elif self.config.allow_pallet_overhang:
            # Floor placement under overhang (F17): the box must still rest
            # ON the deck — deck-contact area >= the effective support ratio
            # of its footprint. Without this, overhang-inflated bounds let a
            # floor box sit fully off the deck, floating in air. Inert when
            # overhang is off (every floor box is then 100% on the deck).
            footprint = cand.dx * cand.dy
            deck_x = min(cand.x2, float(self.pallet.length)) - max(cand.x, 0.0)
            deck_y = min(cand.y2, float(self.pallet.width)) - max(cand.y, 0.0)
            contact = deck_x * deck_y if (deck_x > 0 and deck_y > 0) else 0.0
            if contact <= EPS:
                return False
            min_support = 1.0 if cand.box.requires_full_support else self.config.support_ratio
            if min_support > 0.0 and (
                    footprint <= 0 or contact / footprint < min_support - EPS):
                return False
        # 4. Load bearing (incl. F21: load inherited by under-filling
        #    beneath already-placed boxes — the candidate becomes their
        #    supporter and must be able to carry its share, and that share
        #    flows on down through the candidate's own chain).
        if self.config.enforce_load_bearing:
            inherited = self._rider_inflow(cand)
            if (inherited > cand.box.max_load_on_top
                    + load_tol(cand.box.max_load_on_top)):
                return False
            if not self._load_bearing_ok(cand, supporters,
                                         extra_load=inherited):
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
    #
    # NOTE: Tried 3 additional strategies in a Path 1 attempt — tight_bbox,
    # back_first, min_overhang. Together added +52% to baseline runtime
    # via 14 trials vs the original 8, but yielded ZERO new wins in the
    # candidate set across the 41-case suite. The existing 4 strategies
    # already cover the search space well enough that more scoring variants
    # are redundant. The 3 strategy implementations remain in
    # `_score_placement` (cheap dead code) in case future work wants to
    # re-enable them by extending this tuple. See 12_path_1_decoder.md.
    SCORING_STRATEGIES = ("blb", "bbl", "max_touch", "corner_fit")

    def try_place(self, box: Box, strategy: str = "blb") -> bool:
        """Try each (extreme point, rotation) pair; place at the best feasible.

        In addition to the discovered extreme points, we always try the four
        floor-corner anchors per rotation. This is essential for balanced
        loading: pure EP heuristics can only place adjacent to existing boxes
        and so will never reach diagonally-opposite placements on their own,
        which CoG envelope constraints often require.

        When `config.grasp_alpha > 1`, instead of always picking the single
        best-scoring placement we pick uniformly among the top-α (Parreño
        et al. 2008). This gives BRKGA / multi-start non-trivial per-decode
        variation so they can actually explore the solution space — the
        deterministic best-only decoder converged every chromosome to the
        same packing (the cause of the v2 ablation finding).
        """
        feasible: List[Tuple[Tuple, Placement]] = []
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
                feasible.append((score, cand))
        if not feasible:
            return False
        feasible.sort(key=lambda t: t[0])
        alpha = max(1, self.config.grasp_alpha)
        if alpha == 1:
            chosen = feasible[0][1]
        else:
            # Pick uniformly among the top-α; if fewer feasible than α, the
            # pool is just everything we have.
            pool = feasible[:alpha]
            chosen = self._rng.choice(pool)[1]
        self._commit(chosen)
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
        # F21 (round 3): load inherited by under-filling beneath existing
        # boxes — booked onto the candidate and flowed down with its own
        # weight. Computed BEFORE appending (the rider scan must not see
        # the candidate itself). Old supporters keep their full shares
        # (conservative over-booking; see _rider_inflow).
        inherited = self._rider_inflow(cand)
        if sups:
            # Direct supporters carry cand's own weight (plus its inherited
            # rider load) in proportion to contact, and the flow continues
            # down the support DAG (historical v2 semantics; exact diamond
            # handling since round 3 — see _apply_load_flow).
            self._apply_load_flow(sups, cand.box.weight + inherited)

        self.placements.append(cand)
        self.total_weight += cand.box.weight
        self._top_load[id(cand)] = (self._top_load.get(id(cand), 0.0)
                                    + inherited)
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

    def _apply_load_flow(self, sups: List[Tuple[Placement, float]],
                         weight: float) -> None:
        """Distribute `weight` over direct supporters and on down the
        support DAG, updating the _top_load cache.

        Round 3 (F22): the old recursive `_propagate_load` used a visited
        set that ADDED a re-converged diamond node's second share but
        BLOCKED its onward distribution — everything below the junction
        permanently undercounted, so the check (`_load_bearing_ok`, which
        is exact) and the commit disagreed with each other. This is the
        same accumulate-then-distribute worklist as the dry-run: highest
        bottom-z first, a node's inflow is complete before it is applied
        and forwarded, exact on arbitrary DAGs.
        """
        total_area = sum(a for _, a in sups)
        if total_area <= 0 or weight <= 0:
            return
        inflow: dict = {}
        for s, a in sups:
            inflow[id(s)] = (inflow.get(id(s), 0.0)
                             + weight * (a / total_area))
        frontier = sorted((s for s, _ in sups),
                          key=lambda p: (-p.z, p.x, p.y))
        seen = {id(s) for s, _ in sups}
        while frontier:
            p = frontier.pop(0)
            inc = inflow.get(id(p), 0.0)
            if inc <= 0:
                continue
            self._top_load[id(p)] = self._top_load.get(id(p), 0.0) + inc
            if p.z <= EPS:
                continue
            subs = self._supporters_of(p)
            sub_area = sum(a for _, a in subs)
            if sub_area <= 0:
                continue
            for s2, a2 in subs:
                inflow[id(s2)] = (inflow.get(id(s2), 0.0)
                                  + inc * (a2 / sub_area))
                if id(s2) not in seen:
                    seen.add(id(s2))
                    frontier.append(s2)
            frontier.sort(key=lambda p2: (-p2.z, p2.x, p2.y))

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


def regen_top_load(st: "PalletState") -> dict:
    """Rebuild the per-placement top-load cache from current geometry.

    Used after a removal (PalletPacker) and after postprocess.py's
    orientation swaps — any edit that changes contact areas.
    """
    st._top_load = {id(p): 0.0 for p in st.placements}
    # For every placement, distribute its weight onto its supporters using
    # the same model as _commit (exact worklist flow since round 3, F22).
    # Each box's flow is independent and additive, so iteration order does
    # not matter; keep bottom-up z for determinism of float accumulation.
    for p in sorted(st.placements, key=lambda pl: pl.z):
        sups = st._supporters_of(p)
        if sups:
            st._apply_load_flow(sups, p.box.weight)
    return st._top_load


class PalletPacker:
    """Top-level packer: multi-start search over box orderings."""

    def __init__(self, pallet: Pallet, config: Optional[PackerConfig] = None):
        self.pallet = pallet
        self.config = config or PackerConfig()
        # Wall-clock deadline for pack(time_limit_s=...). None = unbounded
        # (the historical behavior). See _expired().
        self._deadline: Optional[float] = None
        self._rng = random.Random(self.config.seed)

    # -------- SKU-consistent rotation (Bortfeldt-Gehring 2001) ------------
    def _lock_sku_rotation(self, boxes: List[Box]) -> List[Box]:
        """For each SKU with multiple units, lock in the most space-efficient
        rotation as the only allowed one.

        Algorithm: for each group of interchangeable boxes (same dims, weight,
        same rotation set), compute per allowed rotation the units that fit
        a single pallet via grid: `floor(L_eff/dx) * floor(W_eff/dy) *
        floor(H/dz)`. With overhang enabled, L_eff = L + max_overhang and
        W_eff similarly (matching the asymmetric overhang in
        _within_pallet — boxes extend off the +x and +y edges only).
        Without overhang, L_eff = L. The rotation with the highest grid count
        is the layout-most-efficient for that SKU; drop all other rotations
        for those boxes.

        Returns a NEW list of Box objects with restricted `allowed_rotations`.
        Original objects are not mutated; this keeps the caller's data safe
        if they reuse the same Box list across multiple PalletPacker calls.

        Tie-breaking: when multiple rotations tie on grid count, the one
        with the smallest "wasted edge" (sum of L_eff%dx and W_eff%dy) wins
        — this minimises footprint slack and tends to favour aspect ratios
        that interlock well with other SKUs.

        Skipped for:
          - SKU groups with only one unit (rotation freedom is fine).
          - Boxes whose `allowed_rotations` is already a single rotation.
          - Boxes where no rotation fits — leave alone so packing surfaces
            the infeasibility via the unpacked list.
        """
        groups = self._group_identical_boxes(boxes)
        rebuilt: List[Box] = []
        ov = self.pallet.max_overhang if self.config.allow_pallet_overhang else 0.0
        L_eff = self.pallet.length + ov
        W_eff = self.pallet.width + ov
        H = self.pallet.height
        for _key, members in groups.items():
            if len(members) <= 1:
                rebuilt.extend(members)
                continue
            template = members[0]
            if len(template.allowed_rotations) <= 1:
                rebuilt.extend(members)
                continue
            best_rot = None
            best_score: Optional[Tuple[int, float]] = None
            for rot in template.allowed_rotations:
                dx, dy, dz = template.dims_for(rot)
                if dx <= 0 or dy <= 0 or dz <= 0:
                    continue
                if dx > L_eff + EPS or dy > W_eff + EPS or dz > H + EPS:
                    continue
                nx = int(L_eff // dx)
                ny = int(W_eff // dy)
                nz = int(H // dz)
                grid = nx * ny * nz
                if grid <= 0:
                    continue
                slack = (L_eff - nx * dx) + (W_eff - ny * dy)
                score = (grid, -slack)
                if best_score is None or score > best_score:
                    best_score = score
                    best_rot = rot
            if best_rot is None:
                rebuilt.extend(members)
                continue
            locked = [best_rot]
            for b in members:
                rebuilt.append(Box(
                    id=b.id,
                    length=b.length, width=b.width, height=b.height,
                    weight=b.weight,
                    max_load_on_top=b.max_load_on_top,
                    allowed_rotations=locked,
                    group=b.group,
                    requires_full_support=b.requires_full_support,
                ))
        return rebuilt

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
            new_st = PalletState(st.pallet, st.pallet_id, st.config, rng=self._rng)
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

        for bi, box in enumerate(ordered):
            if self._expired():
                unpacked.extend(ordered[bi:])
                break
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
                # Respect max_pallets cap (BR-style single-container mode).
                if (self.config.max_pallets is not None and
                        len(pallets) >= self.config.max_pallets):
                    unpacked.append(box)
                    continue
                st = PalletState(self.pallet,
                                 pallet_id=f"P{len(pallets) + 1:03d}",
                                 config=self.config,
                                 rng=self._rng)
                if st.try_place(box, strategy=strategy):
                    pallets.append(st)
                    placed = True
                else:
                    unpacked.append(box)
        return PackResult(pallets=pallets, unpacked=unpacked)

    # -------- multi-start orchestrator ------------------------------------
    def _expired(self) -> bool:
        return (self._deadline is not None
                and time.monotonic() > self._deadline)

    def _clamp_to_deadline(self, budget_s: float) -> float:
        """Clamp a stage's own wall budget to the time left before the
        deadline — stage gates only check _expired() on ENTRY, so an
        un-clamped stage could run its full budget past the deadline
        (hardening round 2, A3-3). No deadline -> unchanged."""
        if self._deadline is None:
            return budget_s
        return max(0.0, min(budget_s, self._deadline - time.monotonic()))

    def pack(self, boxes: List[Box],
             time_limit_s: Optional[float] = None) -> PackResult:
        """Pack with optional block-building and BRKGA improvements.

        When block-building is enabled, we try multiple decomposition
        strategies (no-blocks, max-blocks, column-blocks, layer-blocks)
        and return the best across all of them.

        time_limit_s (hardening plan B1): optional wall-clock deadline.
        None (default) = unbounded, bit-identical to the historical
        behavior. With a deadline, candidate-generation stages are skipped
        once it expires and the greedy per-box loops stop placing (the
        remainder becomes unpacked) — pack() always returns the best
        candidate found so far. Needed because the v2 warm-start is
        superlinear on homogeneous loads (N=400 identical constrained
        boxes ~175 s) and used to run to completion regardless of the
        caller's budget (finding F2).
        """
        self._deadline = (time.monotonic() + float(time_limit_s)
                          if time_limit_s is not None else None)
        # Phase 2a: SKU-lock candidate set. We keep BOTH locked and unlocked
        # box lists so the search can take the best across both — SKU lock
        # is greedy per-SKU and can break joint-SKU interlock layouts (F3),
        # so it's not safe to apply unconditionally. The candidate-set
        # approach is the simplest fix: try both, keep the best.
        box_variants: List[List[Box]] = [boxes]
        if self.config.sku_consistent_rotation:
            locked = self._lock_sku_rotation(boxes)
            # Only add if SKU lock actually changed something.
            changed = any(
                set(r.name for r in a.allowed_rotations) !=
                set(r.name for r in b.allowed_rotations)
                for a, b in zip(boxes, locked)
            )
            if changed:
                box_variants.append(locked)

        def search(items: List[Box]) -> PackResult:
            # BRKGA's exploration value drops with N when the decoder is
            # deterministic (every chromosome converges to the same packing).
            # With GRASP randomization (grasp_alpha > 1) the decoder is
            # now non-deterministic, so BRKGA can usefully explore at higher N.
            threshold = self.config.brkga_n_threshold
            if (self.config.use_brkga and
                    (threshold is None or len(items) <= threshold)):
                return self._brkga_search(items)
            return self._multi_start(items)

        # Realism tie-break key (D14, gated on realism_weight > 0 — the
        # tuple is unchanged when the flag is off). The candidate set often
        # holds several packings that TIE on (unpacked, pallets, util) —
        # e.g. every candidate that packs all boxes — and the historical
        # first-wins tie left the pick to candidate-generation order, which
        # is how heavy-on-top winners survived. The 4th key prefers the
        # heavy-low / flat / orientation-consistent candidate among ties.
        _realism_ctx = None
        if float(getattr(self.config, "realism_weight", 0.0) or 0.0) > 0.0:
            try:
                from ._brkga_core.realism import (
                    build_realism_context, realism_scalar)
                _realism_ctx = build_realism_context(
                    boxes, self.pallet, self.config)
            except Exception:                      # noqa: BLE001 — never fail a solve
                _realism_ctx = None

        def _realism_key(res: PackResult) -> float:
            if _realism_ctx is None or not res.pallets:
                return 0.0
            vals = [realism_scalar(PackResult(pallets=[st], unpacked=[]),
                                   _realism_ctx) for st in res.pallets]
            return sum(vals) / len(vals)

        def quality(res: PackResult) -> Tuple:
            # Two ranking modes:
            #   "min_unpacked": Unpacked items are the worst outcome.
            #                  Then minimize pallets, then maximize util.
            #                  Best for multi-pallet logistics.
            #   "max_util":    Maximize total packed volume / pallet
            #                  capacity first. Best for single-container
            #                  (BR-style) where leaving 1 small item out
            #                  to gain 2pp density is worth it.
            if self.config.optimize == "max_util":
                # Higher util wins; unpacked count and pallet count are
                # tiebreakers.
                base: Tuple = (-res.total_volume_utilisation,
                               len(res.unpacked), res.num_pallets)
            else:
                base = (len(res.unpacked), res.num_pallets,
                        -res.total_volume_utilisation)
            if _realism_ctx is not None:
                return base + (_realism_key(res),)
            return base

        candidates: List[PackResult] = []
        # Baseline search for each box variant (original + optionally locked).
        for variant in box_variants:
            candidates.append(search(variant))

        # Safety net: guarantee result is ≥ v1 quality AND ≥ original-v2
        # (block-building + BRKGA without GRASP) quality. Without this,
        # Phase 2 randomization (GRASP) and per-SKU rotation locking can
        # destroy specific wins that depend on deterministic decoding or
        # SKU-flexible rotation (e.g. F3's interlock layout).
        #
        # Implementation: re-run a minimal set of candidate paths with
        # GRASP disabled, on the ORIGINAL boxes only (locked-variant is
        # already covered above by the Phase 2 candidates).
        need_safety = (
            self.config.use_safety_net and
            (self.config.grasp_alpha > 1 or self.config.use_brkga)
        )
        if need_safety and self._expired():
            need_safety = False
        if need_safety:
            saved_alpha = self.config.grasp_alpha
            saved_brkga = self.config.use_brkga
            saved_rng = self._rng
            # Fresh RNG so safety-net BRKGA explores the canonical chromosome
            # subspace regardless of how much RNG the Phase 2 path consumed.
            self._rng = random.Random(self.config.seed)
            self.config.grasp_alpha = 1
            try:
                # (1) Deterministic search on the original boxes — this is
                #     the path that finds F3's 1p/90% (BRKGA on individual
                #     boxes without GRASP picks the interlock layout).
                if saved_brkga:
                    self.config.use_brkga = True
                    candidates.append(search(boxes))
                # (2) Pure v1 multi-start (no BRKGA, no GRASP) — absolute v1
                #     guarantee.
                self.config.use_brkga = False
                candidates.append(self._multi_start(boxes))
                # (3) Deterministic BRKGA + block-building — orig-v2 guarantee
                #     for cases where blocks help.
                if saved_brkga and self.config.use_block_building:
                    self.config.use_brkga = True
                    for strategy in ("max", "column", "layer"):
                        items, specs = self._build_blocks(boxes, strategy=strategy)
                        if specs:
                            raw = search(items)
                            candidates.append(self._expand_blocks(raw, specs))
            finally:
                self.config.grasp_alpha = saved_alpha
                self.config.use_brkga = saved_brkga
                self._rng = saved_rng

        # Phase 2 features (GRASP-randomized): try each block-building
        # strategy across all box variants.
        if self.config.use_block_building and not self._expired():
            for variant in box_variants:
                for strategy in ("max", "column", "layer"):
                    items, specs = self._build_blocks(variant, strategy=strategy)
                    if specs:
                        raw = search(items)
                        candidates.append(self._expand_blocks(raw, specs))

        # Phase 4: MIP polish (CP-SAT). Adds a provably-optimal candidate
        # when N is small enough. Lazy-imports the solver so installations
        # without ortools still work. Uses the best heuristic result
        # achieved so far to (a) set a reasonable pallet-count budget and
        # (b) warm-start the MIP with the heuristic's placements.
        #
        # Skips when the heuristic already achieves the volume LB —
        # MIP can't improve on that, and running it would waste budget.
        if (self.config.use_mip_polish
                and len(boxes) <= self.config.mip_n_threshold
                and not self._expired()):
            try:
                from .mip import mip_polish as _mip_polish
                # Estimate a sensible P budget from heuristic candidates,
                # AND get the warm-start hint.
                if candidates:
                    best_so_far = min(candidates, key=quality)
                    P_budget = max(1, best_so_far.num_pallets)
                    warm = best_so_far
                else:
                    P_budget = 1
                    warm = None
                # Respect the user's explicit cap when set.
                if self.config.max_pallets is not None:
                    P_budget = min(P_budget, self.config.max_pallets)
                # Compute the volume LB. If the heuristic is already at LB
                # AND all items are packed, skip MIP — it can't help.
                pallet_vol = self.pallet.length * self.pallet.width * self.pallet.height
                total_vol = sum(b.volume for b in boxes)
                vol_lb = max(1, int(total_vol / pallet_vol + 0.9999))
                if (warm is not None and not warm.unpacked and
                        warm.num_pallets <= vol_lb):
                    pass  # skip — already at LB
                else:
                    mip_budget = self._clamp_to_deadline(_adaptive_mip_budget(
                        len(boxes), self.config.mip_time_limit_s))
                    mip_result = _mip_polish(
                        boxes, self.pallet, self.config,
                        time_limit_s=mip_budget,
                        num_workers=self.config.mip_num_workers,
                        max_pallets=P_budget,
                        warm_start=warm,
                    )
                    if mip_result is not None:
                        candidates.append(mip_result)
                    # Also try one fewer pallet — gives MIP a chance to BEAT v1.
                    if P_budget > 1 and self.config.max_pallets is None:
                        mip_aggressive = _mip_polish(
                            boxes, self.pallet, self.config,
                            time_limit_s=mip_budget,
                            num_workers=self.config.mip_num_workers,
                            max_pallets=P_budget - 1,
                            warm_start=warm,
                        )
                        if mip_aggressive is not None:
                            candidates.append(mip_aggressive)
            except ImportError:
                pass

        # Phase 2e: layer-building decoder candidates. Tries 3 axes × 2 seed
        # strategies per box variant — each combination produces a
        # qualitatively different packing structure.
        if self.config.use_layer_building and not self._expired():
            axes = [self.config.layer_axis] if self.config.layer_axis else ["x", "y", "z"]
            if axes == ["all"]:
                axes = ["x", "y", "z"]
            elif len(axes) == 1 and axes[0] not in ("x", "y", "z"):
                axes = ["x", "y", "z"]
            for variant in box_variants:
                for ax in axes:
                    for strat in ("cross_section", "depth", "min_depth", "sku_volume"):
                        for fill in ("ep", "maxrects", "sku_grid"):
                            if self._expired():
                                break
                            candidates.append(self._layer_pack(
                                variant, axis=ax, seed_strategy=strat, fill=fill,
                            ))

        best = min(candidates, key=quality)

        # Phase 2d: ejection chains. Only fires when (a) opted in and (b)
        # there ARE unpacked items — the standard Crainic-Perboli-Tadei
        # formulation.
        if (self.config.use_ejection_chains and best.unpacked
                and not self._expired()):
            improved = self._ejection_chains(best)
            if quality(improved) < quality(best):
                best = improved

        # Q1: multi-pallet leftover consolidation. Fires when (a) opted in
        # via use_ejection_chains AND (b) the result has 2+ pallets AND
        # (c) no unpacked items left. Targets the "first-fit waste" case
        # where greedy packs everything but uses one pallet too many.
        if (self.config.use_ejection_chains and not self._expired() and
                not best.unpacked and best.num_pallets >= 2):
            # First: try fast item-by-item displacement.
            improved = self._consolidate_leftover_pallet(best)
            if quality(improved) < quality(best):
                best = improved
            # Second: try re-packing everything onto N-1 pallets from scratch
            # with multi_start. Explores fundamentally different orderings.
            if not best.unpacked and best.num_pallets >= 2:
                rebuilt = self._consolidate_via_rebuild(best)
                if rebuilt is not None and quality(rebuilt) < quality(best):
                    best = rebuilt

        # MIP-based last-pallet polish for large N. Activated when (a) MIP
        # polish is enabled, (b) the full-problem MIP wasn't run (N too
        # large), and (c) result still has multiple pallets.
        if (self.config.use_mip_polish and len(boxes) > self.config.mip_n_threshold
                and not self._expired()
                and not best.unpacked and best.num_pallets >= 2):
            polished = self._mip_last_pallet_polish(best)
            if polished is not None and quality(polished) < quality(best):
                best = polished

        return best

    # -------- Phase 2d: ejection chains -----------------------------------
    def _ejection_chains(self, result: PackResult) -> PackResult:
        """Local-search post-process: try to absorb unpacked items by
        displacing 1 (or up to `ejection_max_depth`) placed items, then
        re-placing the displaced items.

        Standard Crainic-Perboli-Tadei 2009 / Faroe-Pisinger-Zachariasen 2003
        formulation: ONLY fires when there are unpacked items. Multi-pallet
        rebalancing is Phase 3 territory, not this method.

        Termination: stops when no swap improves quality, or
        `ejection_max_iters` reached, or wall-clock budget exhausted.

        Doesn't touch the validator — every move calls PalletState.feasible
        via try_place, so the constraint stack is honored.
        """
        if not result.unpacked:
            return result
        depth = max(1, self.config.ejection_max_depth)
        budget = self.config.ejection_max_iters
        wall_budget_s = self._clamp_to_deadline(5.0)  # hard wall-clock cap so we don't stall

        def quality(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets,
                    -res.total_volume_utilisation)

        current = result
        start = time.time()
        for _ in range(budget):
            if time.time() - start > wall_budget_s:
                break
            improved = self._try_one_ejection_pass(current, depth)
            if improved is None or quality(improved) >= quality(current):
                break
            current = improved
        return current

    # -------- MIP-based last-pallet polish for large N --------
    def _mip_last_pallet_polish(self, result: "PackResult") -> Optional["PackResult"]:
        """For large-N cases where MIP can't solve the full problem,
        extract a sub-problem from the weakest pallet + top-z items
        from a neighbor and invoke MIP with the rest of items as
        fixed obstacles.

        If MIP packs all sub-items on (sub_pallet_count - 1) pallets,
        we save a pallet. Otherwise no change.

        Returns the improved PackResult, or None.
        """
        try:
            from .mip import mip_polish as _mip_polish
        except ImportError:
            return None
        if result.num_pallets < 2 or result.unpacked:
            return None
        # Sort pallets by util ascending; weakest first.
        sorted_pallets = sorted(
            result.pallets,
            key=lambda st: sum(p.box.volume for p in st.placements),
        )
        weakest = sorted_pallets[0]
        # Donor candidate = pallet with next-most slack (second-weakest by util).
        donor = sorted_pallets[1] if len(sorted_pallets) > 1 else None
        fixed_pallets = sorted_pallets[2:]  # untouched

        # Build the sub-problem: weakest's items + top-z items from donor.
        sub_boxes: List[Box] = [p.box for p in weakest.placements]
        # From donor, extract top-K items by z-position (most accessible).
        donor_pls_sorted = sorted(
            donor.placements, key=lambda p: -p.z,
        ) if donor else []
        # Pick K items from donor such that sub-problem N stays within
        # MIP's tractable range (~30).
        budget = max(0, 30 - len(sub_boxes))
        donor_movable = donor_pls_sorted[:budget]
        donor_fixed = donor_pls_sorted[budget:] if donor else []
        for pl in donor_movable:
            sub_boxes.append(pl.box)

        if len(sub_boxes) == 0:
            return None

        # Fixed obstacles: untouched pallets' items, plus donor's bottom-z.
        # Pallet indices: weakest=0 (will be empty post-polish),
        # donor=1 (gets repacked), then fixed_pallets at 2..
        obstacles: List[Tuple[int, Placement]] = []
        # Donor's bottom-z items are obstacles on pallet 1 (the donor).
        donor_pallet_idx = 1
        for pl in donor_fixed:
            obstacles.append((donor_pallet_idx, pl))
        # Untouched pallets' items are obstacles on pallets 2, 3, ...
        for fp_idx, st in enumerate(fixed_pallets):
            mip_pallet_idx = 2 + fp_idx
            for pl in st.placements:
                obstacles.append((mip_pallet_idx, pl))

        # Available pallet count for the sub-problem MIP = number of
        # pallets in the result minus 1 (we're trying to drop the weakest).
        target_pallets = result.num_pallets - 1
        if target_pallets < 1:
            return None

        # Build a "warm-start" for the sub-items based on their existing
        # positions (when they exist).
        warm_pallets: List[PalletState] = []
        # Pallet 0 corresponds to the dropped weakest — skip.
        # Pallet 1 is the donor.
        if donor:
            warm_pallets.append(donor)
        # Then the fixed_pallets follow.
        warm_pallets.extend(fixed_pallets)
        warm = PackResult(pallets=warm_pallets, unpacked=[])

        sub_result = _mip_polish(
            sub_boxes, self.pallet, self.config,
            time_limit_s=self._clamp_to_deadline(_adaptive_mip_budget(
                len(sub_boxes), self.config.mip_time_limit_s)),
            num_workers=self.config.mip_num_workers,
            max_pallets=target_pallets,
            warm_start=warm,
            fixed_obstacles=obstacles,
        )
        if sub_result is None or sub_result.unpacked:
            return None

        # Splice: build the final PackResult.
        # The MIP returned a packing of sub_boxes on `target_pallets`
        # pallets. Combine these with the fixed_obstacles' "implicit"
        # pallets.
        # Build pallet groupings.
        new_pallets: List[PalletState] = []
        # First: the MIP-packed sub-pallets get re-built with the
        # fixed obstacles added back in.
        for mip_p_idx, mip_st in enumerate(sub_result.pallets):
            new_st = PalletState(self.pallet, f"P{mip_p_idx+1:03d}",
                                 self.config, rng=self._rng)
            new_st.placements = list(mip_st.placements)
            # Add back obstacles assigned to this same MIP pallet idx.
            for obs_p_idx, obs_pl in obstacles:
                if obs_p_idx == mip_p_idx:
                    new_st.placements.append(obs_pl)
            new_st.total_weight = sum(p.box.weight for p in new_st.placements)
            new_pallets.append(new_st)

        return PackResult(pallets=new_pallets, unpacked=list(result.unpacked))

    # -------- Q1: rebuild with N-1 pallet cap (most-likely-to-succeed) ---
    def _consolidate_via_rebuild(
        self, result: PackResult,
    ) -> Optional[PackResult]:
        """Try to fit everything onto N-1 pallets by re-packing from
        scratch with `max_pallets` constrained. Explores a broader range
        of orderings than the post-hoc ejection approach.

        Returns the rebuilt PackResult if it strictly improves, else None.
        """
        if result.num_pallets < 2:
            return None
        all_boxes: List[Box] = []
        for st in result.pallets:
            for p in st.placements:
                all_boxes.append(p.box)
        if result.unpacked:
            all_boxes.extend(result.unpacked)
        target_count = result.num_pallets - 1

        def quality(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets,
                    -res.total_volume_utilisation)

        # Snapshot config for restore.
        saved = (
            self.config.max_pallets,
            self.config.use_safety_net,
            self.config.use_ejection_chains,
            self.config.use_mip_polish,
        )
        self.config.max_pallets = target_count
        # Disable nested ejection/MIP/safety to avoid recursion and stalls.
        self.config.use_safety_net = False
        self.config.use_ejection_chains = False
        self.config.use_mip_polish = False
        saved_rng = self._rng
        self._rng = random.Random(self.config.seed)
        try:
            # Try several diverse orderings via multi_start.
            repacked = self._multi_start(all_boxes)
        finally:
            (self.config.max_pallets,
             self.config.use_safety_net,
             self.config.use_ejection_chains,
             self.config.use_mip_polish) = saved
            self._rng = saved_rng
        if quality(repacked) < quality(result):
            return repacked
        return None

    # -------- Q1: multi-pallet ejection (consolidate leftover pallet) ----
    def _consolidate_leftover_pallet(self, result: PackResult) -> PackResult:
        """Try to drop the lowest-utilized pallet by relocating its items
        onto the other pallets.

        This addresses the classical "first-fit waste" failure mode in
        bin-packing: greedy outer loops commit items permanently to a
        pallet, and the last (typically least-utilized) pallet ends up
        as a sparse leftover. The standard fix in the literature is
        ejection chains operating across pallets, not just on unpacked
        items.

        Algorithm:
          1. If there's only 1 pallet, nothing to consolidate. Skip.
          2. Find the lowest-volume pallet (the "victim").
          3. Build a fresh candidate from the OTHER pallets (preserving
             their layouts) and try to relocate each victim item onto
             them.
          4. For each victim item:
               a. Try direct placement on the most-utilized fitting pallet.
               b. If that fails, try depth-1 ejection: remove ONE item
                  from a candidate pallet, place the victim item, place
                  the displaced item somewhere else.
          5. If ALL victim items relocate successfully, the victim
             pallet is now empty — we've saved one pallet. Otherwise,
             revert to the original.

        Bounded by config.ejection_max_iters (outer attempts) and
        a 5-second wall-clock budget.
        """
        if len(result.pallets) < 2:
            return result
        if result.unpacked:
            # Unpacked items take precedence — handled by _ejection_chains.
            return result
        budget = self.config.ejection_max_iters
        wall_budget_s = self._clamp_to_deadline(5.0)

        def quality(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets,
                    -res.total_volume_utilisation)

        current = result
        start = time.time()
        for _ in range(budget):
            if time.time() - start > wall_budget_s:
                break
            improved = self._try_one_leftover_consolidation(current)
            if improved is None or quality(improved) >= quality(current):
                break
            current = improved
        return current

    def _try_one_leftover_consolidation(
        self, result: PackResult
    ) -> Optional[PackResult]:
        """One pass: pick the lowest-util pallet and try to redistribute
        its items onto the others. Returns the improved PackResult or
        None on no improvement.
        """
        if len(result.pallets) < 2:
            return None

        # Sort pallets by used volume ascending — the first is the victim.
        sorted_pallets = sorted(
            result.pallets,
            key=lambda st: sum(p.box.volume for p in st.placements),
        )
        victim = sorted_pallets[0]
        others = sorted_pallets[1:]
        victim_items = [pl.box for pl in victim.placements]
        if not victim_items:
            # Already empty — just drop it.
            return PackResult(pallets=list(others), unpacked=result.unpacked)

        # Build fresh candidate pallets from `others`, preserving layouts.
        candidate_pallets: List[PalletState] = []
        for st in others:
            new_st = PalletState(self.pallet, st.pallet_id, self.config,
                                 rng=self._rng)
            for p in st.placements:
                new_st.placements.append(p)
                new_st.total_weight += p.box.weight
                new_st._top_load.setdefault(id(p), 0.0)
            new_st.extreme_points = self._regen_extreme_points(new_st)
            new_st._top_load = self._regen_top_load(new_st)
            candidate_pallets.append(new_st)

        # Sort victim items by volume descending — the biggest are hardest
        # to relocate; try them first to fail fast.
        victim_items_sorted = sorted(victim_items, key=lambda b: -b.volume)

        unplaceable: List[Box] = []
        for item in victim_items_sorted:
            placed = False
            # Try direct placement on candidate pallets, sorted by
            # current fullness descending (best_fit-style).
            ordered = sorted(
                candidate_pallets,
                key=lambda st: -sum(p.box.volume for p in st.placements),
            )
            for st in ordered:
                if st.try_place(item, strategy="blb"):
                    placed = True
                    break
            if placed:
                continue
            # Direct placement failed. Try depth-1 ejection: remove ONE
            # item from a candidate pallet, place `item`, then try to
            # place the displaced item somewhere.
            placed_via_ejection = False
            for target_pallet in ordered:
                if placed_via_ejection:
                    break
                # Try removing each item from target_pallet and see if `item`
                # can replace it. Order placement-removal candidates by
                # smallest volume first (least disruptive to displace).
                candidates_to_remove = sorted(
                    target_pallet.placements, key=lambda p: p.box.volume,
                )
                for to_remove in candidates_to_remove:
                    # Build a hypothetical state of target_pallet without `to_remove`.
                    hypo = PalletState(self.pallet, target_pallet.pallet_id,
                                       self.config, rng=self._rng)
                    for p in target_pallet.placements:
                        if id(p) == id(to_remove):
                            continue
                        hypo.placements.append(p)
                        hypo.total_weight += p.box.weight
                    hypo.extreme_points = self._regen_extreme_points(hypo)
                    hypo._top_load = self._regen_top_load(hypo)
                    if not hypo.try_place(item, strategy="blb"):
                        continue
                    # `item` fits. Now we need to re-place `to_remove`
                    # on some OTHER candidate pallet (or back on `hypo`'s
                    # newly-modified state — but it was just removed
                    # because it didn't fit alongside `item`).
                    displaced_placed = False
                    for st2 in candidate_pallets:
                        if id(st2) == id(target_pallet):
                            continue
                        if st2.try_place(to_remove.box, strategy="blb"):
                            displaced_placed = True
                            break
                    if displaced_placed:
                        # Commit: replace target_pallet's contents with hypo.
                        target_pallet.placements = list(hypo.placements)
                        target_pallet.total_weight = hypo.total_weight
                        target_pallet.extreme_points = hypo.extreme_points
                        target_pallet._top_load = hypo._top_load
                        placed_via_ejection = True
                        break
                    # else: revert (no state changes were made yet).
            if not placed_via_ejection:
                unplaceable.append(item)

        if unplaceable:
            # Even with ejections we couldn't relocate everything.
            # Don't commit a partial result — return None.
            return None

        # Success — every victim item relocated. The victim pallet is
        # now empty; drop it.
        return PackResult(pallets=candidate_pallets, unpacked=list(result.unpacked))

    def _try_one_ejection_pass(
        self, result: PackResult, depth: int
    ) -> Optional[PackResult]:
        """One pass: for each unpacked item, try to fit it by displacing
        one (or up to `depth`) placed items. Returns the improved result,
        or None if nothing improved.
        """
        if not result.unpacked:
            return None
        all_placements: List[Tuple[int, Placement]] = []
        for pid, st in enumerate(result.pallets):
            for p in st.placements:
                all_placements.append((pid, p))

        def quality(res: PackResult) -> Tuple:
            return (len(res.unpacked), res.num_pallets,
                    -res.total_volume_utilisation)

        baseline_q = quality(result)

        for target in list(result.unpacked):
            for pid, placed in reversed(all_placements):
                if placed.box.id == target.id:
                    continue
                candidate_result = self._rebuild_after_swap(
                    result, [(pid, placed)], [target, placed.box]
                )
                if candidate_result is None:
                    continue
                if quality(candidate_result) < baseline_q:
                    return candidate_result
                if depth >= 2:
                    # Depth-2: try removing one more item from the same pallet.
                    for pid2, placed2 in reversed(all_placements):
                        if pid2 != pid:
                            break
                        if placed2.box.id in (placed.box.id, target.id):
                            continue
                        candidate2 = self._rebuild_after_swap(
                            result, [(pid, placed), (pid2, placed2)],
                            [target, placed.box, placed2.box],
                        )
                        if candidate2 is None:
                            continue
                        if quality(candidate2) < baseline_q:
                            return candidate2
        return None

    def _rebuild_after_swap(
        self,
        result: PackResult,
        remove: List[Tuple[int, Placement]],
        insert: List[Box],
    ) -> Optional[PackResult]:
        """Build a fresh PackResult that excludes `remove` and tries to insert
        the `insert` boxes (which include both new candidates and the removed
        items, in some order).

        Returns None on any infeasibility OR if pallet count would grow
        (we only accept ejection moves that don't increase pallet count).
        """
        # Build the set of placements that survive.
        removed_ids = {(pid, id(pl)) for pid, pl in remove}
        # Reconstruct fresh PalletStates with the surviving placements.
        new_pallets: List[PalletState] = []
        for pid, st in enumerate(result.pallets):
            new_st = PalletState(self.pallet, st.pallet_id, self.config,
                                 rng=self._rng)
            for p in st.placements:
                if (pid, id(p)) in removed_ids:
                    continue
                # Re-commit by calling _commit (preserves EPs and load cache).
                # The placement is known to be feasible since it was already
                # there; skip feasibility check for speed.
                new_st.placements.append(p)
                new_st.total_weight += p.box.weight
                new_st._top_load.setdefault(id(p), 0.0)
            # Rebuild EPs from scratch for the surviving placements.
            new_st.extreme_points = self._regen_extreme_points(new_st)
            # Rebuild _top_load from scratch for correctness after removal.
            new_st._top_load = self._regen_top_load(new_st)
            new_pallets.append(new_st)

        # Drop any pallet that's now empty.
        new_pallets = [st for st in new_pallets if st.placements]

        # Try to place each `insert` box in some pallet without opening
        # a new one. Order them largest-volume-first for stability.
        to_place = sorted(insert, key=lambda b: -b.volume)
        residual_unpacked: List[Box] = []

        for box in to_place:
            placed = False
            # best_fit-style: try the most-full pallet first.
            ordered_pallets = sorted(
                new_pallets,
                key=lambda st: -sum(p.box.volume for p in st.placements),
            )
            for st in ordered_pallets:
                if st.try_place(box, strategy="blb"):
                    placed = True
                    break
            if not placed:
                residual_unpacked.append(box)

        # Merge with the unpacked items that weren't part of `insert`.
        all_unpacked = list(residual_unpacked)
        original_inserted_ids = {b.id for b in insert}
        for u in result.unpacked:
            if u.id not in original_inserted_ids:
                all_unpacked.append(u)

        # Reject moves that would open a new pallet (pallet count grew).
        if len(new_pallets) > len(result.pallets):
            return None

        return PackResult(pallets=new_pallets, unpacked=all_unpacked)

    def _regen_extreme_points(self, st: "PalletState") -> List[Tuple[float, float, float]]:
        """Rebuild EP set from scratch for a state's current placements.

        Conservative reconstruction: floor origin + each placement's three
        "open" corners, gravity-projected onto whatever they land on.
        """
        eps: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
        for p in st.placements:
            for corner in [
                (p.x2, p.y, p.z),
                (p.x, p.y2, p.z),
                (p.x, p.y, p.z2),
            ]:
                projected = st._gravity_project(corner)
                if projected not in eps:
                    eps.append(projected)
        # Prune EPs that sit inside any placed box.
        eps = [ep for ep in eps if not st._point_inside_any_placement(ep)]
        return eps

    # -------- Phase 2e: 2D MaxRects (Jylanki 2010) ------------------------
    @staticmethod
    def _maxrects_prune(rects: List[Tuple[float, float, float, float]]) -> List[Tuple[float, float, float, float]]:
        """Remove free rectangles that are strictly contained in another.

        MaxRects's correctness relies on keeping only *maximal* free rects.
        Without pruning, the algorithm produces invalid placements (free
        rects with overlapping items inside).
        """
        result: List[Tuple[float, float, float, float]] = []
        for i, r in enumerate(rects):
            if r[2] <= 1e-9 or r[3] <= 1e-9:
                continue
            dominated = False
            for j, s in enumerate(rects):
                if i == j:
                    continue
                if (s[0] <= r[0] + 1e-9 and s[1] <= r[1] + 1e-9 and
                        s[0] + s[2] >= r[0] + r[2] - 1e-9 and
                        s[1] + s[3] >= r[1] + r[3] - 1e-9 and
                        (s[0] < r[0] - 1e-9 or s[1] < r[1] - 1e-9 or
                         s[0] + s[2] > r[0] + r[2] + 1e-9 or
                         s[1] + s[3] > r[1] + r[3] + 1e-9)):
                    dominated = True
                    break
            if not dominated:
                result.append(r)
        return result

    @staticmethod
    def _maxrects_split(
        free: List[Tuple[float, float, float, float]],
        px: float, py: float, pw: float, ph: float,
    ) -> List[Tuple[float, float, float, float]]:
        """Split free rectangles affected by a (px,py,pw,ph) placement.

        For each free rect that intersects the placement, replace it with up
        to 4 maximal sub-rects (left/right/below/above). Non-intersecting
        rects pass through unchanged. Result is then pruned of dominated
        rects by the caller.
        """
        new_free: List[Tuple[float, float, float, float]] = []
        for fx, fy, fw, fh in free:
            # No intersection.
            if (px + pw <= fx + 1e-9 or px >= fx + fw - 1e-9 or
                    py + ph <= fy + 1e-9 or py >= fy + fh - 1e-9):
                new_free.append((fx, fy, fw, fh))
                continue
            # Left of placement.
            if px > fx + 1e-9:
                new_free.append((fx, fy, px - fx, fh))
            # Right of placement.
            if px + pw < fx + fw - 1e-9:
                new_free.append((px + pw, fy, (fx + fw) - (px + pw), fh))
            # Below placement.
            if py > fy + 1e-9:
                new_free.append((fx, fy, fw, py - fy))
            # Above placement.
            if py + ph < fy + fh - 1e-9:
                new_free.append((fx, py + ph, fw, (fy + fh) - (py + ph)))
        return new_free

    def _fill_layer_sku_grid(
        self,
        remaining: List[Box],
        axis: str,
        layer_depth: float,
        layer_offset: float,
        face_w: float,
        face_h: float,
        global_placements: List[Placement],
    ) -> set:
        """SKU-grid pattern fill (Bischoff-Ratcliff 1995).

        Each layer is dominated by ONE SKU placed in a regular n_x × n_y
        grid where every grid item extends the FULL slab depth — no
        "depth shadows" wasted behind shorter items. The leftover
        L-shaped area is filled with secondary SKUs via 2D MaxRects.

        This is the actual algorithm that gets 83% on BR1, distinct from
        the generic 2D MaxRects fill (which packs items at the back of
        the slab with arbitrary depth, wasting volume behind shorter
        items).

        Algorithm:
          1. Group remaining items by SKU. For each SKU, find rotations
             where the SKU's depth-axis dim equals (or is within ε of)
             `layer_depth`. These rotations give "no-shadow" grid candidates.
          2. Among SKUs that have at least one no-shadow rotation, pick
             the one with the most total volume.
          3. Choose that SKU's rotation that gives the highest n_x*n_y
             grid count on the slab face.
          4. Place up to min(n_x*n_y, N_sku) copies in a regular grid.
          5. Build initial free-rect set = L-shaped leftover.
          6. Run 2D MaxRects on the leftover face area with the remaining
             items (any rotation valid, since they pack against the back
             wall).

        Returns the set of box IDs placed. Appends Placements to
        `global_placements` in REAL pallet coordinates.

        Falls back to returning empty set (caller's responsibility to
        handle) if no SKU offers a no-shadow grid candidate.
        """
        # axis_dims/to_global helpers — re-derive here for self-containment.
        def axis_dims(b: Box, rot: Rotation) -> Tuple[float, float, float]:
            dx, dy, dz = b.dims_for(rot)
            if axis == "x":
                return dx, dy, dz
            if axis == "y":
                return dy, dx, dz
            return dz, dx, dy

        def to_global(rot: Rotation, fx: float, fy: float) -> Tuple[float, float, float]:
            if axis == "x":
                return layer_offset, fx, fy
            if axis == "y":
                return fx, layer_offset, fy
            return fx, fy, layer_offset

        # Group items by SKU key.
        sku_buckets: Dict[tuple, Dict[str, list]] = {}
        for idx, b in enumerate(remaining):
            no_shadow_rots: List[Tuple[Rotation, float, float, float]] = []
            for rot in b.allowed_rotations:
                d, fw, fh = axis_dims(b, rot)
                # No-shadow ⟺ depth-dim equals slab depth.
                if abs(d - layer_depth) > 1e-6:
                    continue
                if fw > face_w + 1e-9 or fh > face_h + 1e-9:
                    continue
                no_shadow_rots.append((rot, d, fw, fh))
            if not no_shadow_rots:
                continue
            k = (b.length, b.width, b.height, b.weight,
                 tuple(sorted(r.name for r in b.allowed_rotations)))
            bucket = sku_buckets.setdefault(k, {"items": [], "rots": no_shadow_rots})
            bucket["items"].append((idx, b))

        if not sku_buckets:
            return set()

        # Pick the dominant SKU (most remaining volume).
        dom_key = max(
            sku_buckets.keys(),
            key=lambda k: sum(it[1].volume for it in sku_buckets[k]["items"]),
        )
        dom = sku_buckets[dom_key]

        # Pick best rotation: maximize grid count on slab face.
        best_rot = None
        best_score = (-1, 0.0)
        for rot, d, fw, fh in dom["rots"]:
            nx = int(face_w // fw)
            ny = int(face_h // fh)
            grid = nx * ny
            if grid == 0:
                continue
            slack = (face_w - nx * fw) + (face_h - ny * fh)
            score = (grid, -slack)
            if score > best_score:
                best_score = score
                best_rot = (rot, fw, fh, nx, ny)
        if best_rot is None:
            return set()
        rot, fw, fh, nx, ny = best_rot
        available_items = list(dom["items"])
        units_to_place = min(nx * ny, len(available_items))

        placed_ids: set = set()
        # Place grid items in row-major order.
        idx_in_bucket = 0
        for j in range(ny):
            for i in range(nx):
                if idx_in_bucket >= units_to_place:
                    break
                _bidx, item = available_items[idx_in_bucket]
                fx = i * fw
                fy = j * fh
                gx, gy, gz = to_global(rot, fx, fy)
                global_placements.append(Placement(
                    box=item, rotation=rot, x=gx, y=gy, z=gz,
                ))
                placed_ids.add(item.id)
                idx_in_bucket += 1

        # Build initial free-rect set = L-shaped leftover.
        grid_w = nx * fw
        grid_h = ny * fh
        free_rects: List[Tuple[float, float, float, float]] = []
        if face_w > grid_w + 1e-9:
            free_rects.append((grid_w, 0.0, face_w - grid_w, face_h))
        if face_h > grid_h + 1e-9:
            free_rects.append((0.0, grid_h, grid_w, face_h - grid_h))
        if not free_rects:
            return placed_ids

        # Fill leftover via 2D MaxRects with remaining items.
        items_to_fill: List[Tuple[Box, List[Tuple[float, float, Rotation, float]]]] = []
        for b in remaining:
            if b.id in placed_ids:
                continue
            opts: List[Tuple[float, float, Rotation, float]] = []
            for r in b.allowed_rotations:
                d, w, h = axis_dims(b, r)
                if d > layer_depth + 1e-9:
                    continue
                opts.append((w, h, r, d))
            if opts:
                items_to_fill.append((b, opts))
        items_to_fill.sort(key=lambda t: -max(w * h for w, h, _, _ in t[1]))

        for b, opts in items_to_fill:
            best_choice = None
            best_score_2d: Optional[Tuple[float, float]] = None
            for w, h, r, _d in opts:
                for fx, fy, rw, rh in free_rects:
                    if w <= rw + 1e-9 and h <= rh + 1e-9:
                        leftover_w = rw - w
                        leftover_h = rh - h
                        score = (rw * rh - w * h, min(leftover_w, leftover_h))
                        if best_score_2d is None or score < best_score_2d:
                            best_score_2d = score
                            best_choice = (fx, fy, w, h, r)
            if best_choice is None:
                continue
            fx, fy, pw, ph, r = best_choice
            gx, gy, gz = to_global(r, fx, fy)
            global_placements.append(Placement(box=b, rotation=r,
                                               x=gx, y=gy, z=gz))
            placed_ids.add(b.id)
            free_rects = self._maxrects_split(free_rects, fx, fy, pw, ph)
            free_rects = self._maxrects_prune(free_rects)

        return placed_ids

    def _fill_layer_maxrects(
        self,
        remaining: List[Box],
        axis: str,
        layer_depth: float,
        layer_offset: float,
        face_w: float,
        face_h: float,
        global_placements: List[Placement],
        seed_idx: int,
        seed_rot: Rotation,
    ) -> set:
        """2D MaxRects layer-fill (Jylanki 2010).

        Each item is reduced to its (depth, fw, fh) given the layering
        axis; an item is eligible if `depth <= layer_depth` and its
        cross-section (fw, fh) fits the slab face (face_w, face_h).

        For each eligible item we enumerate its (rotation, fw, fh, depth)
        options (multiple 3D rotations may give different 2D footprints),
        find the best-area-fit placement in the current free-rectangle
        set, and split affected rects per Jylanki's algorithm.

        Returns the set of box IDs that got placed; placements are appended
        to `global_placements` in real pallet coordinates.
        """
        L = self.pallet.length
        W = self.pallet.width
        H = self.pallet.height

        # Per-item face/depth options under each allowed rotation.
        def axis_dims(b: Box, rot: Rotation) -> Tuple[float, float, float]:
            dx, dy, dz = b.dims_for(rot)
            if axis == "x":
                return dx, dy, dz
            if axis == "y":
                return dy, dx, dz
            return dz, dx, dy

        def to_global(rot: Rotation, fx: float, fy: float) -> Tuple[float, float, float]:
            if axis == "x":
                return layer_offset, fx, fy
            if axis == "y":
                return fx, layer_offset, fy
            return fx, fy, layer_offset

        # Build option list per item.
        items_options: List[Tuple[int, Box, List[Tuple[float, float, Rotation, float]]]] = []
        for idx, b in enumerate(remaining):
            opts: List[Tuple[float, float, Rotation, float]] = []
            for rot in b.allowed_rotations:
                d, fw, fh = axis_dims(b, rot)
                if d > layer_depth + 1e-9:
                    continue
                if fw > face_w + 1e-9 or fh > face_h + 1e-9:
                    continue
                opts.append((fw, fh, rot, d))
            if opts:
                items_options.append((idx, b, opts))

        if not items_options:
            return set()

        # Place the seed first at (0, 0) to anchor the layer.
        free_rects: List[Tuple[float, float, float, float]] = [(0.0, 0.0, face_w, face_h)]
        placed_ids: set = set()
        seed_box = remaining[seed_idx]
        _depth, seed_fw, seed_fh = axis_dims(seed_box, seed_rot)
        # Sanity: seed's depth was already verified by caller.
        if seed_fw > face_w + 1e-9 or seed_fh > face_h + 1e-9:
            return set()
        gx, gy, gz = to_global(seed_rot, 0.0, 0.0)
        global_placements.append(Placement(box=seed_box, rotation=seed_rot,
                                           x=gx, y=gy, z=gz))
        placed_ids.add(seed_box.id)
        free_rects = self._maxrects_split(free_rects, 0.0, 0.0, seed_fw, seed_fh)
        free_rects = self._maxrects_prune(free_rects)

        # Sort remaining items by largest-area-option descending.
        items_options.sort(key=lambda t: -max(fw * fh for fw, fh, _, _ in t[2]))

        for idx, b, opts in items_options:
            if b.id in placed_ids:
                continue
            # Try every (rotation, free_rect, orientation) combo.
            best_score: Optional[Tuple[float, float]] = None
            best_choice = None  # (rx, ry, fw, fh, rot)
            for fw, fh, rot, _d in opts:
                for fx, fy, fw_avail, fh_avail in free_rects:
                    if fw <= fw_avail + 1e-9 and fh <= fh_avail + 1e-9:
                        # Best Area Fit + Best Short Side Fit tie-break.
                        leftover_w = fw_avail - fw
                        leftover_h = fh_avail - fh
                        score = (fw_avail * fh_avail - fw * fh,
                                 min(leftover_w, leftover_h))
                        if best_score is None or score < best_score:
                            best_score = score
                            best_choice = (fx, fy, fw, fh, rot)
            if best_choice is None:
                continue
            rx, ry, pw, ph, rot = best_choice
            gx, gy, gz = to_global(rot, rx, ry)
            global_placements.append(Placement(box=b, rotation=rot,
                                               x=gx, y=gy, z=gz))
            placed_ids.add(b.id)
            free_rects = self._maxrects_split(free_rects, rx, ry, pw, ph)
            free_rects = self._maxrects_prune(free_rects)

        return placed_ids

    # -------- Phase 2e: layer-building decoder ----------------------------
    def _layer_pack(self, boxes: List[Box], axis: str = "x",
                    seed_strategy: str = "cross_section",
                    fill: str = "ep") -> PackResult:
        """Layer-building decoder (George-Robinson 1980, Bischoff-Ratcliff 1995).

        Builds slab-shaped sub-problems along one pallet axis. Each slab's
        depth is set by a seed item; the slab is filled with smaller items
        via the extreme-point engine on a virtual sub-pallet whose
        dimensions match the slab.

        Three axis orientations:
          axis='z' — HORIZONTAL layers (Bischoff-Ratcliff 1995 layer-building).
                     Each layer is at a different height. Best when items
                     share a height (F13).
          axis='x' — WALL-building along pallet length (George-Robinson 1980).
                     Each wall extends full pallet height + width. Best
                     when cargo is heterogeneous and the container is
                     elongated (BR sets).
          axis='y' — same as 'x' but along pallet width.

        Seed-selection strategies (which item starts the next slab):
          'cross_section' — largest dy*dz (or analogous) cross-section.
                            Fills the slab densely with similar items.
          'depth'         — largest depth-dim. Thickest slab. Useful when
                            we want each slab to span more cargo at once.
          'volume'        — largest total volume. Standard "place big first."

        Currently a single-pallet decoder. Items that don't fit go to
        `.unpacked`. The multi-pallet outer loop in `pack()` will accept
        these and fall back to extreme-point candidates.
        """
        if not boxes:
            return PackResult(pallets=[], unpacked=[])

        pallet = self.pallet
        L, W, H = pallet.length, pallet.width, pallet.height
        axis = axis.lower()
        if axis not in ("x", "y", "z"):
            axis = "x"

        sub_cfg = PackerConfig(
            support_ratio=self.config.support_ratio,
            require_centroid_supported=self.config.require_centroid_supported,
            allow_pallet_overhang=False,
            heavy_on_bottom=False,
            cog_envelope_fraction=1.0,
            cog_check_min_load_fraction=1.0,
            enforce_load_bearing=False,
            multi_start_trials=1,
            seed=self.config.seed,
            grasp_alpha=self.config.grasp_alpha,
            use_safety_net=False,
            max_pallets=None,
        )

        # Each rotation produces dims = (dx, dy, dz). Pull out the dim along
        # the layering axis and the two cross-section dims.
        def axis_and_cross(b: Box, rot: Rotation) -> Tuple[float, float, float]:
            dx, dy, dz = b.dims_for(rot)
            if axis == "x":
                return dx, dy, dz       # depth, ca (=W's axis), cb (=H's axis)
            if axis == "y":
                return dy, dx, dz
            return dz, dx, dy            # axis == 'z'

        # Virtual pallet for a slab of given depth along the chosen axis.
        def virtual_pallet(depth: float) -> Pallet:
            if axis == "x":
                return Pallet(length=depth, width=W, height=H,
                              max_weight=float("inf"))
            if axis == "y":
                return Pallet(length=L, width=depth, height=H,
                              max_weight=float("inf"))
            return Pallet(length=L, width=W, height=depth,
                          max_weight=float("inf"))

        # Translate a sub-placement (relative to slab origin) into the
        # global pallet coordinates.
        def to_real(sp: Placement, offset: float) -> Placement:
            if axis == "x":
                return Placement(box=sp.box, rotation=sp.rotation,
                                 x=offset + sp.x, y=sp.y, z=sp.z)
            if axis == "y":
                return Placement(box=sp.box, rotation=sp.rotation,
                                 x=sp.x, y=offset + sp.y, z=sp.z)
            return Placement(box=sp.box, rotation=sp.rotation,
                             x=sp.x, y=sp.y, z=offset + sp.z)

        axis_total = L if axis == "x" else (W if axis == "y" else H)
        cross_a_total = W if axis == "x" else (L if axis == "y" else L)
        cross_b_total = H if axis == "x" else (H if axis == "y" else W)

        remaining = list(boxes)
        global_placements: List[Placement] = []
        layer_offset = 0.0

        while remaining and layer_offset < axis_total - 1e-6:
            if self._expired():
                break          # deadline: keep the slabs built so far (A3-3)
            rem_axis = axis_total - layer_offset

            # Build per-SKU volume table once per layer (cheap; constant).
            sku_vol: Dict[tuple, float] = {}
            if seed_strategy == "sku_volume":
                for b in remaining:
                    k = (b.length, b.width, b.height, b.weight,
                         tuple(sorted(r.name for r in b.allowed_rotations)))
                    sku_vol[k] = sku_vol.get(k, 0.0) + b.volume

            # Pick the best seed.
            best: Optional[Tuple[tuple, int, Rotation, Tuple[float, float, float]]] = None
            for i, b in enumerate(remaining):
                for rot in b.allowed_rotations:
                    d, ca, cb = axis_and_cross(b, rot)
                    if d > rem_axis + 1e-9:
                        continue
                    if ca > cross_a_total + 1e-9 or cb > cross_b_total + 1e-9:
                        continue
                    if seed_strategy == "cross_section":
                        # Thick layer with large cross-section seed.
                        score = (ca * cb, b.volume, d)
                    elif seed_strategy == "depth":
                        # Pick maximum-depth seed → thickest layers.
                        score = (d, ca * cb, b.volume)
                    elif seed_strategy == "min_depth":
                        # Pick minimum-depth seed → thinnest layers (more of
                        # them; matches BR-1995 layer-building style).
                        score = (-d, ca * cb, b.volume)
                    elif seed_strategy == "sku_volume":
                        # Bischoff-Ratcliff 1995 style: pick the SKU with
                        # most remaining volume, then the rotation that
                        # gives the smallest depth-dim for thinner layers.
                        k = (b.length, b.width, b.height, b.weight,
                             tuple(sorted(r.name for r in b.allowed_rotations)))
                        score = (sku_vol[k], -d, ca * cb)
                    else:  # 'volume'
                        score = (b.volume, ca * cb, d)
                    if best is None or score > best[0]:
                        best = (score, i, rot, (d, ca, cb))
            if best is None:
                break

            _score, seed_idx, seed_rot, (sd, _ca, _cb) = best
            layer_depth = sd

            if fill == "maxrects":
                # 2D MaxRects on the slab face. Items lie flat at the back
                # of the slab; their depth-axis dim must be ≤ layer_depth.
                placed_ids = self._fill_layer_maxrects(
                    remaining, axis, layer_depth, layer_offset,
                    cross_a_total, cross_b_total, global_placements,
                    seed_idx=seed_idx, seed_rot=seed_rot,
                )
                if not placed_ids:
                    remaining.pop(seed_idx)
                    continue
                remaining = [b for b in remaining if b.id not in placed_ids]
            elif fill == "sku_grid":
                # Q4: SKU-grid pattern fill (BR-1995). Depth-matched
                # rotation guarantees no depth shadows in the grid;
                # leftover L-shape filled by 2D MaxRects.
                placed_ids = self._fill_layer_sku_grid(
                    remaining, axis, layer_depth, layer_offset,
                    cross_a_total, cross_b_total, global_placements,
                )
                if not placed_ids:
                    # No SKU offered a no-shadow grid candidate at this
                    # slab depth. Skip this layer and let the next slab
                    # try a different depth.
                    remaining.pop(seed_idx)
                    continue
                remaining = [b for b in remaining if b.id not in placed_ids]
            else:
                # Extreme-point sub-fill (original).
                sub_pallet = virtual_pallet(layer_depth)
                sub_state = PalletState(sub_pallet, f"LAYER_{layer_offset:.0f}",
                                        sub_cfg, rng=self._rng)
                seed_box = remaining[seed_idx]
                placed_seed = sub_state.try_place(seed_box, strategy="blb")
                if not placed_seed:
                    remaining.pop(seed_idx)
                    continue
                candidates = list(remaining)
                candidates.pop(seed_idx)
                candidates.sort(key=lambda b: -b.volume)
                placed_ids = {seed_box.id}
                for c in candidates:
                    if sub_state.try_place(c, strategy="blb"):
                        placed_ids.add(c.id)
                for sp in sub_state.placements:
                    global_placements.append(to_real(sp, layer_offset))
                remaining = [b for b in remaining if b.id not in placed_ids]
            layer_offset += layer_depth

        if not global_placements:
            return PackResult(pallets=[], unpacked=remaining)

        # Replay-validate: layer-building uses a virtual sub-pallet whose
        # "floor" is the slab origin. For axis='z' (horizontal layers), the
        # slab's floor at layer_offset > 0 is NOT the real pallet floor —
        # items placed there must rest on items from the layer below. The
        # sub-pack didn't know that. We re-build the result against the
        # real PalletState in bottom-up order and demote any placement that
        # fails the full constraint stack to `.unpacked`. This is also a
        # cheap insurance against any other constraint that happens to bind
        # at the global level but not in the sub-pack (CoG, weight, etc.).
        sorted_placements = sorted(
            global_placements,
            key=lambda p: (p.z, p.y, p.x),
        )
        real_state = PalletState(self.pallet, "P001", self.config,
                                 rng=self._rng)
        demoted: List[Box] = []
        for p in sorted_placements:
            # Build a candidate Placement that re-feeds the real state's
            # feasibility check. We bypass try_place's EP-search because we
            # already have the position.
            if real_state.feasible(p):
                real_state._commit(p)
            else:
                demoted.append(p.box)
        if not real_state.placements:
            return PackResult(pallets=[], unpacked=remaining + demoted)
        return PackResult(pallets=[real_state], unpacked=remaining + demoted)

    def _regen_top_load(self, st: "PalletState") -> dict:
        """Rebuild the per-placement top-load cache after a removal."""
        return regen_top_load(st)

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
            if best_result is not None and self._expired():
                break
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

