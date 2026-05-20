"""
lower_bounds.py — Lower bounds on pallet count for 3D-MIBSBPP instances.

The point of this module: when our heuristic returns K pallets, we want to know
whether K is really the best achievable, or whether there's algorithmic
headroom. Phase 0 of the improvement roadmap.

Three bounds, all valid (each ≤ true pallet-count optimum):

  L_volume       ceil(sum(box.volume for FITTABLE boxes) / effective_pallet_vol).
                 Effective volume uses extended footprint when overhang is
                 allowed. Unfittable boxes (don't fit pallet in any rotation,
                 or exceed pallet weight) are excluded from the sum — they
                 stay unpacked, they don't drive pallet count.

  L_perSKU_max   max over SKUs s of ceil(N_s / M_s).
                 M_s = max units of SKU s on one pallet under its rotation
                 set (using effective footprint if overhang allowed), capped
                 by pallet weight budget. Valid because pallet count must be
                 ≥ ceil(N_s / M_s) for every SKU s individually.

  L_geometric    Martello-Pisinger-style combinatorial bounds.
                 (a) Items "big in every axis" (in every allowed rotation,
                     all three dims > pallet/2 + 1e-9) are pairwise mutually
                     exclusive: their 3D projections overlap on every face,
                     so they can't share a pallet. Count of such items is an
                     LB.
                 (b) For each axis pair (a, b), items "big in both a and b"
                     have overlapping (a, b) projections and must be
                     separated in the third axis c. Sum of their c-extents
                     on any pallet ≤ effective_P_c, giving a count LB.
                 With overhang, P[a] and P[b] use extended footprint; P[c]
                 uses height (which doesn't overhang).

  L_LP           LP relaxation in HiGHS via PuLP.
                 Volume-row-only relaxation. Closed form: ceil(sum_vol /
                 V_pallet). Kept as a real LP so Phase 4's MIP polish has
                 the same entry point — column-generation patterns can be
                 added later without restructuring.

A NOTE on what we tried and removed: a "per-SKU sum" bound — ceil(sum N_s/M_s)
— would be tempting but is INVALID in 3D-BPP. Per-SKU fractional shares are
NOT additive: in a feasible packing, sum_s n_{P,s}/M_s on a single pallet can
exceed 1 (because M_s is the max-when-alone bound; mixed packings can stack
density in non-obvious ways). The earlier version of this module included
that bound and immediately produced BELOW_LB rows (LB > true optimum) on
C1 / D1–D7 / F1 / F4. Removed in commit-of-record.

Best LB = max of the three valid bounds.

References
----------
- Martello, S., Pisinger, D. & Vigo, D. (2000). The Three-Dimensional Bin
  Packing Problem. Operations Research 48(2), 256–267.
- Boschetti, M. (2004). New lower bounds for the three-dimensional finite
  bin packing problem. Discrete Applied Mathematics 140, 241–258.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from pallet_packer import Box, Pallet, PackerConfig, Rotation


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LBReport:
    """All bounds for one instance, plus the consolidated best."""
    volume: int
    per_sku_max: int
    geometric: int
    lp: Optional[int]                       # None if PuLP/solver unavailable.
    best: int
    detail: Dict[str, object] = field(default_factory=dict)

    def __str__(self) -> str:
        lp = "n/a" if self.lp is None else str(self.lp)
        return (f"LB(vol={self.volume}, sku_max={self.per_sku_max}, "
                f"geom={self.geometric}, lp={lp}) → best={self.best}")


# ---------------------------------------------------------------------------
# Effective pallet dims given overhang
# ---------------------------------------------------------------------------
def _effective_pallet_dims(
    pallet: Pallet, config: Optional[PackerConfig]
) -> Tuple[float, float, float]:
    """Return the (L, W, H) the LB should use.

    With overhang allowed, the footprint extends by max_overhang on each
    side, so effective L = L + 2*max_overhang. Height does not overhang.

    Without overhang (default), the strict pallet dims are returned.
    """
    overhang = 0.0
    if (config is not None and config.allow_pallet_overhang
            and pallet.max_overhang > 0):
        overhang = pallet.max_overhang
    return (
        pallet.length + 2 * overhang,
        pallet.width + 2 * overhang,
        pallet.height,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _min_dim_over_rotations(box: Box, axis: int) -> float:
    """Smallest extent in `axis` over allowed rotations.

    `_min_dim_over_rotations(b, a) > P_a/2` ⟺ in every allowed rotation, the
    box's a-extent exceeds half the pallet — i.e. "this box is unavoidably
    big in axis a."
    """
    return min(box.dims_for(r)[axis] for r in box.allowed_rotations)


def _grid_fit(
    box: Box, eff: Tuple[float, float, float], rot: Rotation
) -> int:
    dx, dy, dz = box.dims_for(rot)
    if dx > eff[0] + 1e-9 or dy > eff[1] + 1e-9 or dz > eff[2] + 1e-9:
        return 0
    return (math.floor(eff[0] / dx)
            * math.floor(eff[1] / dy)
            * math.floor(eff[2] / dz))


def _max_units_per_pallet(
    box: Box, eff: Tuple[float, float, float], pallet: Pallet
) -> int:
    """Upper bound on units of one SKU per pallet."""
    best = 0
    for r in box.allowed_rotations:
        n = _grid_fit(box, eff, r)
        if n > best:
            best = n
    if best == 0:
        return 0
    if pallet.max_weight < float("inf") and box.weight > 0:
        by_weight = math.floor(pallet.max_weight / box.weight)
        if by_weight < best:
            best = max(0, by_weight)
    return best


def _box_fits_anywhere(
    box: Box, eff: Tuple[float, float, float], pallet: Pallet
) -> bool:
    """True iff at least one rotation fits inside the (effective) pallet AND
    the box's weight is ≤ pallet.max_weight."""
    if pallet.max_weight < float("inf") and box.weight > pallet.max_weight + 1e-9:
        return False
    for r in box.allowed_rotations:
        dx, dy, dz = box.dims_for(r)
        if dx <= eff[0] + 1e-9 and dy <= eff[1] + 1e-9 and dz <= eff[2] + 1e-9:
            return True
    return False


def _sku_key(b: Box) -> tuple:
    return (b.length, b.width, b.height, b.weight,
            tuple(sorted(r.name for r in b.allowed_rotations)),
            b.max_load_on_top)


def _group_skus(boxes: Iterable[Box]) -> Dict[tuple, List[Box]]:
    groups: Dict[tuple, List[Box]] = {}
    for b in boxes:
        groups.setdefault(_sku_key(b), []).append(b)
    return groups


# ---------------------------------------------------------------------------
# L_volume
# ---------------------------------------------------------------------------
def volume_lb(
    boxes: Iterable[Box],
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
) -> int:
    boxes = list(boxes)
    if not boxes:
        return 0
    eff = _effective_pallet_dims(pallet, config)
    eff_vol = eff[0] * eff[1] * eff[2]
    # Only items that fit contribute to required pallet count.
    fittable_vol = sum(b.volume for b in boxes if _box_fits_anywhere(b, eff, pallet))
    if fittable_vol <= 0:
        return 0
    return max(1, math.ceil(fittable_vol / eff_vol - 1e-9))


# ---------------------------------------------------------------------------
# L_perSKU_max
# ---------------------------------------------------------------------------
def per_sku_max_lb(
    boxes: Iterable[Box],
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
) -> int:
    """max over SKUs s of ceil(N_s / M_s).

    Each SKU independently requires ⌈N_s / M_s⌉ pallets because at most M_s
    fit on one pallet. Therefore total pallet count is at least the max
    across SKUs. (The sum is NOT a valid bound — see module docstring.)
    """
    skus = _group_skus(boxes)
    if not skus:
        return 0
    eff = _effective_pallet_dims(pallet, config)
    best = 0
    for items in skus.values():
        n = len(items)
        m = _max_units_per_pallet(items[0], eff, pallet)
        if m == 0:
            # Item doesn't fit (geometry or weight) — those go unpacked,
            # they don't drive pallet count.
            continue
        best = max(best, math.ceil(n / m))
    return best


# ---------------------------------------------------------------------------
# L_geometric — Martello-Pisinger style
# ---------------------------------------------------------------------------
def geometric_lb(
    boxes: Iterable[Box],
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
) -> Tuple[int, dict]:
    boxes = list(boxes)
    if not boxes:
        return 0, {}
    eff = _effective_pallet_dims(pallet, config)
    half = (eff[0] / 2.0, eff[1] / 2.0, eff[2] / 2.0)

    # Restrict to boxes that actually fit; an unfittable box can't drive an
    # LB on the count of pallets needed for fittable items.
    fittable = [b for b in boxes if _box_fits_anywhere(b, eff, pallet)]
    if not fittable:
        return 0, {}

    # (a) Items big in every axis: their 3D projections overlap on every
    # face, so they're pairwise mutually exclusive on any pallet.
    big_xyz = sum(
        1 for b in fittable
        if all(_min_dim_over_rotations(b, a) > half[a] + 1e-9
               for a in range(3))
    )

    # (b) For each axis pair (a, b): items big in BOTH a and b in every
    # rotation have overlapping (a, b) projections, so they must be
    # separated in the third axis c. Sum of their c-extents on any pallet
    # ≤ effective P_c. Note c is height for (0,1); for (0,2) and (1,2),
    # c is a footprint axis and uses effective length/width.
    pair_bounds: Dict[Tuple[int, int], int] = {}
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        c = 3 - a - b
        K = [
            it for it in fittable
            if _min_dim_over_rotations(it, a) > half[a] + 1e-9
            and _min_dim_over_rotations(it, b) > half[b] + 1e-9
        ]
        if not K:
            pair_bounds[(a, b)] = 0
            continue
        sum_c = sum(_min_dim_over_rotations(it, c) for it in K)
        pair_bounds[(a, b)] = max(1, math.ceil(sum_c / eff[c] - 1e-9))

    best = max([big_xyz] + list(pair_bounds.values()))
    return best, {
        "big_xyz_count": big_xyz,
        "pair_bounds": {f"{a},{b}": v for (a, b), v in pair_bounds.items()},
    }


# ---------------------------------------------------------------------------
# L_LP — HiGHS via PuLP (volume row only, valid)
# ---------------------------------------------------------------------------
def lp_lb(
    boxes: Iterable[Box],
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
    time_limit_s: float = 10.0,
) -> Optional[int]:
    """LP relaxation of bin-packing with the volume capacity row only.

    Single-pallet relaxation:
        minimize  K
        s.t.      sum_{i fittable} v_i ≤ V_eff · K
                  K ≥ 0

    Returns ceil(LP_opt) — exactly equal to volume_lb in the current
    formulation. Kept as a real LP so Phase 4 can add pattern columns or
    integrality without restructuring this module.

    Removed earlier per-SKU density row was mathematically unsound — see
    module docstring.

    Returns None if PuLP isn't installed.
    """
    try:
        import pulp
    except ImportError:
        return None

    boxes = list(boxes)
    if not boxes:
        return 0

    eff = _effective_pallet_dims(pallet, config)
    pallet_vol = eff[0] * eff[1] * eff[2]
    fittable_vol = sum(b.volume for b in boxes if _box_fits_anywhere(b, eff, pallet))

    if fittable_vol <= 0:
        return 0

    prob = pulp.LpProblem("lower_bound_relaxation", pulp.LpMinimize)
    K = pulp.LpVariable("K", lowBound=0)
    prob += K, "min_pallets"
    prob += fittable_vol <= pallet_vol * K, "volume_capacity"

    available = pulp.listSolvers(onlyAvailable=True)
    if "HiGHS" in available:
        solver = pulp.HiGHS(msg=False, timeLimit=time_limit_s)
    else:
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s)
    try:
        status = prob.solve(solver)
    except Exception:
        return None
    if status != 1:
        return None
    opt = pulp.value(prob.objective)
    if opt is None:
        return None
    return max(1, math.ceil(opt - 1e-6))


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------
def compute_lower_bounds(
    boxes: Iterable[Box],
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
    use_lp: bool = True,
) -> LBReport:
    """Compute every LB for one instance; return consolidated report."""
    boxes = list(boxes)
    vol = volume_lb(boxes, pallet, config)
    sku_max = per_sku_max_lb(boxes, pallet, config)
    geom, geom_detail = geometric_lb(boxes, pallet, config)
    lp = lp_lb(boxes, pallet, config) if use_lp else None
    bounds = [vol, sku_max, geom]
    if lp is not None:
        bounds.append(lp)
    return LBReport(
        volume=vol,
        per_sku_max=sku_max,
        geometric=geom,
        lp=lp,
        best=max(bounds) if bounds else 0,
        detail={"geometric": geom_detail},
    )


# ---------------------------------------------------------------------------
# CLI / self-tests
# ---------------------------------------------------------------------------
if __name__ == "__main__":  # pragma: no cover
    # Test 1: six 60×60×60 boxes in a 120×120×120 pallet → LB=1.
    pallet = Pallet(length=120, width=120, height=120)
    boxes = [Box(id=f"x{i}", length=60, width=60, height=60) for i in range(6)]
    print("Six 60^3 in 120^3:", compute_lower_bounds(boxes, pallet))

    # Test 2: three 70^3 boxes in 120^3 pallet → LB=3 (big_xyz).
    big_boxes = [Box(id=f"b{i}", length=70, width=70, height=70) for i in range(3)]
    print("Three 70^3 in 120^3:", compute_lower_bounds(big_boxes, pallet))

    # Test 3: single oversized box → LB=0 (doesn't fit, unpacked).
    too_big = [Box(id="too_big", length=200, width=50, height=50)]
    print("Single too-big box:", compute_lower_bounds(too_big, Pallet(120, 100, 100)))

    # Test 4: overhang case → LB=1.
    from pallet_packer import THIS_SIDE_UP
    overhang_pallet = Pallet(120, 100, 100, max_weight=200, max_overhang=20)
    overhang_cfg = PackerConfig(allow_pallet_overhang=True)
    overhang_boxes = [Box(id=f"O{i}", length=70, width=60, height=100, weight=4,
                          allowed_rotations=THIS_SIDE_UP) for i in range(4)]
    print("F10 overhang:", compute_lower_bounds(overhang_boxes, overhang_pallet, overhang_cfg))
