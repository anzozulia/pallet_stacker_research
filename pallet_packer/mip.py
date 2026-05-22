"""
mip_polish.py — CP-SAT 3D bin-packing solver for MIP polish.

Phase 4 of the algorithmic roadmap. Provides a constraint-programming
solver that takes a small set of boxes and a pallet, finds the optimum
single-pallet packing (or best within a time budget), and returns the
result in the same `PackResult` shape as the heuristic packer.

Use case: applied at the end of `PalletPacker.pack()` when the
sub-problem is small enough (N ≤ `mip_n_threshold`, default 25) AND
the user has set `use_mip_polish=True`. Acts as a "guaranteed-optimal-
or-near-it" candidate alongside the heuristic candidates.

Formulation
-----------
For N boxes, P pallets (default P = max_pallets):

  rotation[i][r]   bool   exactly one r ∈ allowed_rotations[i].
  x[i], y[i], z[i] int    placement in mm; bounded by pallet dims minus
                          box's chosen dim.
  assigned[i][p]   bool   item i is on pallet p.
  unpacked[i]      bool   1 - sum_p assigned[i][p].

Constraints:
  Exactly one rotation per item.
  dx[i] = sum_r rotation[i][r] * dim_x(r, box[i])  (linear via Add).
  x[i] + dx[i] ≤ L (if any pallet assignment), enforced via OnlyEnforceIf.
  Pairwise no-overlap: for each (i, j) and each pallet p, if both on p,
    at least one of 6 separation axes must hold.
  Pallet weight: sum of weights on each pallet ≤ max_weight.
  Pallet count cap respected by limiting P.

Objective (lex):
  1. minimize sum(unpacked[i])
  2. minimize total volume excluded = sum(box[i].volume * unpacked[i])
  These collapse into one objective: sum(volume * unpacked) — packing
  the items with most "volume per slot" first.

References
----------
- do Nascimento, O.X., Queiroz, T.A. & Junqueira, L. (2021). Practical
  constraints in the container loading problem. C&OR 128.
- OR-Tools CP-SAT no-overlap and interval primitives.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

try:
    from ortools.sat.python import cp_model
    _CP_SAT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CP_SAT_AVAILABLE = False
    cp_model = None  # type: ignore

from .models import Box, Pallet, PackerConfig, Placement, Rotation
from .packer import PackResult, PalletState


def _int(v: float) -> int:
    """Round to int — CP-SAT requires integer domains. Our test inputs are
    already mm-grained integers so this is exact."""
    return int(round(v))


def mip_polish(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    time_limit_s: float = 30.0,
    num_workers: int = 4,
    max_pallets: Optional[int] = None,
    warm_start: Optional[PackResult] = None,
    fixed_obstacles: Optional[List[Tuple[int, Placement]]] = None,
) -> Optional[PackResult]:
    """Solve the geometric 3D bin-packing problem exactly (or best within
    `time_limit_s`) using CP-SAT.

    Currently models:
      * Geometric fit and no-overlap.
      * Per-item rotation choice (from allowed set).
      * Per-pallet weight budget.
      * `max_pallets` cap (default config.max_pallets, else 1 — typical for
        the use-case where MIP polish takes a small slice of the input).

    Returns None if CP-SAT isn't installed, if no items fit at all, or if
    the solver times out without producing any feasible solution.
    """
    if not _CP_SAT_AVAILABLE:
        return None
    if not boxes:
        return PackResult(pallets=[], unpacked=[])

    if max_pallets is None:
        max_pallets = config.max_pallets if config.max_pallets is not None else 1
    n = len(boxes)
    P = max_pallets
    L = _int(pallet.length)
    W = _int(pallet.width)
    H = _int(pallet.height)
    max_w = pallet.max_weight if pallet.max_weight < float("inf") else None

    model = cp_model.CpModel()

    # --- decision variables ---
    rot_vars: List[List[cp_model.IntVar]] = []   # rot_vars[i][r] = BoolVar
    rotations: List[List[Rotation]] = []          # the actual Rotation list per item
    dx_var: List[cp_model.IntVar] = []
    dy_var: List[cp_model.IntVar] = []
    dz_var: List[cp_model.IntVar] = []

    for i, b in enumerate(boxes):
        rots = list(b.allowed_rotations)
        rotations.append(rots)
        rot_bools = [model.NewBoolVar(f"r{i}_{r.name}") for r in rots]
        rot_vars.append(rot_bools)
        # Exactly one rotation per item.
        model.AddExactlyOne(rot_bools)

        dims_per_rot = [b.dims_for(r) for r in rots]
        # dx[i] = sum_r rot_bools[r] * dx_for_rotation
        dx_choices = [_int(d[0]) for d in dims_per_rot]
        dy_choices = [_int(d[1]) for d in dims_per_rot]
        dz_choices = [_int(d[2]) for d in dims_per_rot]
        max_dx = max(dx_choices)
        max_dy = max(dy_choices)
        max_dz = max(dz_choices)
        dxv = model.NewIntVar(0, max_dx, f"dx{i}")
        dyv = model.NewIntVar(0, max_dy, f"dy{i}")
        dzv = model.NewIntVar(0, max_dz, f"dz{i}")
        # Channeling: dxv = sum_r rot_bools[r] * dx_choices[r].
        model.Add(dxv == sum(b_r * d for b_r, d in zip(rot_bools, dx_choices)))
        model.Add(dyv == sum(b_r * d for b_r, d in zip(rot_bools, dy_choices)))
        model.Add(dzv == sum(b_r * d for b_r, d in zip(rot_bools, dz_choices)))
        dx_var.append(dxv)
        dy_var.append(dyv)
        dz_var.append(dzv)

    # Placement coordinates.
    x_var = [model.NewIntVar(0, L, f"x{i}") for i in range(n)]
    y_var = [model.NewIntVar(0, W, f"y{i}") for i in range(n)]
    z_var = [model.NewIntVar(0, H, f"z{i}") for i in range(n)]

    # Pallet assignment.
    assigned = [[model.NewBoolVar(f"a{i}_p{p}") for p in range(P)] for i in range(n)]
    placed = [model.NewBoolVar(f"placed{i}") for i in range(n)]
    for i in range(n):
        model.Add(sum(assigned[i]) == placed[i])
    unpacked = [model.NewBoolVar(f"u{i}") for i in range(n)]
    for i in range(n):
        model.Add(placed[i] + unpacked[i] == 1)

    # Geometric containment, ONLY if placed.
    for i in range(n):
        model.Add(x_var[i] + dx_var[i] <= L).OnlyEnforceIf(placed[i])
        model.Add(y_var[i] + dy_var[i] <= W).OnlyEnforceIf(placed[i])
        model.Add(z_var[i] + dz_var[i] <= H).OnlyEnforceIf(placed[i])

    # Pairwise no-overlap: for each pair (i, j) and each pallet p where both
    # are assigned, at least one of 6 separation axes must hold.
    # We build the "same-pallet" boolean once per pair-pallet.
    for i in range(n):
        for j in range(i + 1, n):
            for p in range(P):
                # both_on_p ⟺ (assigned[i][p] ∧ assigned[j][p])
                both = model.NewBoolVar(f"both{i}_{j}_{p}")
                model.AddBoolAnd([assigned[i][p], assigned[j][p]]).OnlyEnforceIf(both)
                model.AddBoolOr([assigned[i][p].Not(), assigned[j][p].Not()]).OnlyEnforceIf(both.Not())

                # 6 separation literals.
                sep = [model.NewBoolVar(f"s{i}_{j}_{p}_{k}") for k in range(6)]
                model.Add(x_var[i] + dx_var[i] <= x_var[j]).OnlyEnforceIf(sep[0])
                model.Add(x_var[j] + dx_var[j] <= x_var[i]).OnlyEnforceIf(sep[1])
                model.Add(y_var[i] + dy_var[i] <= y_var[j]).OnlyEnforceIf(sep[2])
                model.Add(y_var[j] + dy_var[j] <= y_var[i]).OnlyEnforceIf(sep[3])
                model.Add(z_var[i] + dz_var[i] <= z_var[j]).OnlyEnforceIf(sep[4])
                model.Add(z_var[j] + dz_var[j] <= z_var[i]).OnlyEnforceIf(sep[5])
                # If both on same pallet, at least one separation must hold.
                model.AddBoolOr(sep).OnlyEnforceIf(both)

    # Fixed obstacles: pre-placed items at known positions on specific
    # pallets. These don't get re-arranged but the free items must
    # not overlap with them. Used for subproblem MIP polish where we
    # extract a slice of items and treat the rest as obstacles.
    if fixed_obstacles:
        for i in range(n):
            for (obs_p_idx, obs_pl) in fixed_obstacles:
                if obs_p_idx >= P:
                    continue
                # If free item i is on the same pallet as the obstacle,
                # it must not overlap.
                ox = _int(obs_pl.x); oy = _int(obs_pl.y); oz = _int(obs_pl.z)
                odx = _int(obs_pl.dx); ody = _int(obs_pl.dy); odz = _int(obs_pl.dz)
                # 6 separation literals for i vs the fixed obstacle.
                osep = [model.NewBoolVar(f"obs_s{i}_p{obs_p_idx}_k{k}_{id(obs_pl)}") for k in range(6)]
                model.Add(x_var[i] + dx_var[i] <= ox).OnlyEnforceIf(osep[0])
                model.Add(ox + odx <= x_var[i]).OnlyEnforceIf(osep[1])
                model.Add(y_var[i] + dy_var[i] <= oy).OnlyEnforceIf(osep[2])
                model.Add(oy + ody <= y_var[i]).OnlyEnforceIf(osep[3])
                model.Add(z_var[i] + dz_var[i] <= oz).OnlyEnforceIf(osep[4])
                model.Add(oz + odz <= z_var[i]).OnlyEnforceIf(osep[5])
                # If item i is on the obstacle's pallet, at least one
                # separation must hold.
                model.AddBoolOr(osep).OnlyEnforceIf(assigned[i][obs_p_idx])

    # Fragility constraint: if max_load_on_top == 0, nothing may be placed
    # on top of this item. For each fragile item i and each candidate j on
    # the same pallet, forbid j from sitting directly above i with xy
    # overlap. This captures the most common fragility pattern (a 0
    # tolerance "this side up / nothing on top" SKU). General-load tracking
    # would require flow constraints which we skip for now.
    for i in range(n):
        if boxes[i].max_load_on_top > 0.5:
            continue  # not fragile
        for j in range(n):
            if i == j:
                continue
            # If j is directly above i (z_j == z_i + dz_i) on the same
            # pallet AND their (x, y) projections overlap, that's
            # disallowed. We enforce: same_pallet AND z_above ⟹ no xy
            # overlap.
            for p in range(P):
                # Use existing pairwise overlap reasoning: we already
                # require no 3D-overlap (the pair-no-overlap constraint).
                # Above-and-overlapping in (x,y) is itself enforced by
                # the existing no-overlap constraint IF the items also
                # overlap in z. Here we want to forbid the specific
                # configuration where j sits at z_i + dz_i (touching the
                # top face of i) AND projects onto i.
                # Build: same_pallet_ij AND z_above_ji AND xo_ij AND yo_ij ⟹ FALSE.
                # i.e., we add the constraint: ¬(same ∧ z_above ∧ xo ∧ yo).
                same_p = model.NewBoolVar(f"frag_sp{i}_{j}_{p}")
                model.AddBoolAnd([assigned[i][p], assigned[j][p]]).OnlyEnforceIf(same_p)
                model.AddBoolOr([assigned[i][p].Not(), assigned[j][p].Not()]).OnlyEnforceIf(same_p.Not())
                z_above = model.NewBoolVar(f"frag_za{i}_{j}_{p}")
                model.Add(z_var[j] == z_var[i] + dz_var[i]).OnlyEnforceIf(z_above)
                model.Add(z_var[j] != z_var[i] + dz_var[i]).OnlyEnforceIf(z_above.Not())
                x_o1 = model.NewBoolVar(f"frag_xo1{i}_{j}_{p}")
                x_o2 = model.NewBoolVar(f"frag_xo2{i}_{j}_{p}")
                model.Add(x_var[i] + 1 <= x_var[j] + dx_var[j]).OnlyEnforceIf(x_o1)
                model.Add(x_var[j] + 1 <= x_var[i] + dx_var[i]).OnlyEnforceIf(x_o2)
                xo = model.NewBoolVar(f"frag_xo{i}_{j}_{p}")
                model.AddBoolAnd([x_o1, x_o2]).OnlyEnforceIf(xo)
                model.AddBoolOr([x_o1.Not(), x_o2.Not()]).OnlyEnforceIf(xo.Not())
                y_o1 = model.NewBoolVar(f"frag_yo1{i}_{j}_{p}")
                y_o2 = model.NewBoolVar(f"frag_yo2{i}_{j}_{p}")
                model.Add(y_var[i] + 1 <= y_var[j] + dy_var[j]).OnlyEnforceIf(y_o1)
                model.Add(y_var[j] + 1 <= y_var[i] + dy_var[i]).OnlyEnforceIf(y_o2)
                yo = model.NewBoolVar(f"frag_yo{i}_{j}_{p}")
                model.AddBoolAnd([y_o1, y_o2]).OnlyEnforceIf(yo)
                model.AddBoolOr([y_o1.Not(), y_o2.Not()]).OnlyEnforceIf(yo.Not())
                # If all four hold, that violates fragility. Forbid via OR.
                model.AddBoolOr([
                    same_p.Not(), z_above.Not(), xo.Not(), yo.Not(),
                ])

    # Pallet weight budget.
    if max_w is not None and max_w < float("inf"):
        max_w_int = _int(max_w)
        for p in range(P):
            model.Add(
                sum(_int(boxes[i].weight) * assigned[i][p] for i in range(n))
                <= max_w_int
            )
        # Also enforce items too heavy for the pallet are forced unpacked.
        for i, b in enumerate(boxes):
            if _int(b.weight) > max_w_int:
                model.Add(placed[i] == 0)

    # Items that don't fit in ANY rotation are forced unpacked.
    for i, b in enumerate(boxes):
        fits = False
        for r in b.allowed_rotations:
            dx, dy, dz = b.dims_for(r)
            if dx <= pallet.length and dy <= pallet.width and dz <= pallet.height:
                fits = True
                break
        if not fits:
            model.Add(placed[i] == 0)

    # Quantitative support constraint via corner sampling (Q2).
    # Approximates the heuristic's support_ratio constraint by requiring
    # every corner of an off-floor item to be inside the projection of
    # some supporter directly below it.
    #
    # With K=4 corners (top-left, top-right, bottom-left, bottom-right),
    # requiring ALL 4 covered is stricter than support_ratio=0.8 — any
    # MIP-found layout that passes this corner-coverage test will also
    # pass the heuristic's support_ratio=0.8 check. The cost is some
    # potentially valid 0.8-support layouts are rejected, but the MIP
    # candidate is just one of many in the candidate set, so this is OK.
    #
    # Falls back to the lighter "no floating" constraint when
    # support_ratio < 0.5 (rare in practice).
    use_strong_support = config.support_ratio >= 0.5
    # How many corners (of 4) must be covered. We match support_ratio
    # via ceil(4 * support_ratio): 0.8 → 4 corners (strict), 0.75 → 3,
    # 0.5 → 2. The K=4 corner sampling is a slight under-approximation
    # of full support_ratio area — we relax by 1 corner so the MIP can
    # find layouts that pass the validator's quantitative check.
    min_corners_covered = max(1, math.floor(4 * config.support_ratio))
    if use_strong_support:
        # For each item i with z[i] > 0: at least `min_corners_covered`
        # of the 4 corners must be covered by some supporter j with
        # z_j + dz_j = z_i, same pallet.
        for i in range(n):
            on_floor = model.NewBoolVar(f"floor{i}")
            model.Add(z_var[i] == 0).OnlyEnforceIf(on_floor)
            model.Add(z_var[i] >= 1).OnlyEnforceIf(on_floor.Not())
            # 4 corner positions: (x, y), (x+dx, y), (x, y+dy), (x+dx, y+dy).
            # We need to model "supporter j contains corner k" for each
            # (i, j, k). The constraint is:
            #   x_j ≤ corner_x ≤ x_j + dx_j   AND
            #   y_j ≤ corner_y ≤ y_j + dy_j
            # Note: top edge points are exclusive in standard convention,
            # so we use strict-or-equal with small adjustments — but since
            # we're modelling discrete integer coords with placement
            # semantics "extends to but not including x+dx", a corner at
            # (x_j + dx_j, *) is OUTSIDE supporter j's footprint. So we use
            # `corner_x < x_j + dx_j` (i.e., +1 ≤ x_j + dx_j).
            for k in range(4):
                covered_lits: List = []
                for j in range(n):
                    if i == j:
                        continue
                    same_pallet_lits = []
                    for p in range(P):
                        bp = model.NewBoolVar(f"sp_s{i}_{j}_{p}_k{k}")
                        model.AddBoolAnd([assigned[i][p], assigned[j][p]]).OnlyEnforceIf(bp)
                        model.AddBoolOr([assigned[i][p].Not(), assigned[j][p].Not()]).OnlyEnforceIf(bp.Not())
                        same_pallet_lits.append(bp)
                    same_pallet = model.NewBoolVar(f"smp{i}_{j}_k{k}")
                    model.AddBoolOr(same_pallet_lits).OnlyEnforceIf(same_pallet)
                    model.AddBoolAnd([s.Not() for s in same_pallet_lits]).OnlyEnforceIf(same_pallet.Not())

                    z_aligned = model.NewBoolVar(f"zal_s{i}_{j}_k{k}")
                    model.Add(z_var[j] + dz_var[j] == z_var[i]).OnlyEnforceIf(z_aligned)
                    model.Add(z_var[j] + dz_var[j] != z_var[i]).OnlyEnforceIf(z_aligned.Not())

                    # Corner x position depends on k.
                    if k in (1, 3):
                        # right corner
                        cx_lo = model.NewBoolVar(f"cxlo{i}_{j}_k{k}")
                        cx_hi = model.NewBoolVar(f"cxhi{i}_{j}_k{k}")
                        # Need x_i + dx_i - 1 >= x_j (corner is at x_i + dx_i - 1, the rightmost cell)
                        # And x_i + dx_i - 1 <= x_j + dx_j - 1 → x_i + dx_i ≤ x_j + dx_j
                        model.Add(x_var[i] + dx_var[i] >= x_var[j] + 1).OnlyEnforceIf(cx_lo)
                        model.Add(x_var[i] + dx_var[i] <= x_var[j] + dx_var[j]).OnlyEnforceIf(cx_hi)
                    else:
                        cx_lo = model.NewBoolVar(f"cxlo{i}_{j}_k{k}")
                        cx_hi = model.NewBoolVar(f"cxhi{i}_{j}_k{k}")
                        # left corner at x_i. Need x_j ≤ x_i and x_i < x_j + dx_j.
                        model.Add(x_var[i] >= x_var[j]).OnlyEnforceIf(cx_lo)
                        model.Add(x_var[i] + 1 <= x_var[j] + dx_var[j]).OnlyEnforceIf(cx_hi)
                    if k in (2, 3):
                        # bottom corner (y_i + dy_i - 1)
                        cy_lo = model.NewBoolVar(f"cylo{i}_{j}_k{k}")
                        cy_hi = model.NewBoolVar(f"cyhi{i}_{j}_k{k}")
                        model.Add(y_var[i] + dy_var[i] >= y_var[j] + 1).OnlyEnforceIf(cy_lo)
                        model.Add(y_var[i] + dy_var[i] <= y_var[j] + dy_var[j]).OnlyEnforceIf(cy_hi)
                    else:
                        cy_lo = model.NewBoolVar(f"cylo{i}_{j}_k{k}")
                        cy_hi = model.NewBoolVar(f"cyhi{i}_{j}_k{k}")
                        model.Add(y_var[i] >= y_var[j]).OnlyEnforceIf(cy_lo)
                        model.Add(y_var[i] + 1 <= y_var[j] + dy_var[j]).OnlyEnforceIf(cy_hi)

                    supports = model.NewBoolVar(f"sup_s{i}_{j}_k{k}")
                    model.AddBoolAnd([
                        placed[j], same_pallet, z_aligned,
                        cx_lo, cx_hi, cy_lo, cy_hi,
                    ]).OnlyEnforceIf(supports)
                    model.AddBoolOr([
                        placed[j].Not(), same_pallet.Not(), z_aligned.Not(),
                        cx_lo.Not(), cx_hi.Not(), cy_lo.Not(), cy_hi.Not(),
                    ]).OnlyEnforceIf(supports.Not())
                    covered_lits.append(supports)
                # Define a Bool "corner k of item i is covered" = OR(covered_lits).
                corner_is_covered = model.NewBoolVar(f"cov_{i}_k{k}")
                if covered_lits:
                    model.AddBoolOr(covered_lits).OnlyEnforceIf(corner_is_covered)
                    model.AddBoolAnd([c.Not() for c in covered_lits]).OnlyEnforceIf(corner_is_covered.Not())
                else:
                    # No supporter candidates — corner is always uncovered.
                    model.Add(corner_is_covered == 0)
                # Aggregate across the 4 corners of this item.
                if k == 0:
                    # First time we see item i: initialize sum tracker.
                    corner_covered_per_item = [corner_is_covered]
                else:
                    corner_covered_per_item.append(corner_is_covered)
                if k == 3:
                    # We have all 4 corners — apply the support constraint.
                    # If placed and not on floor, need at least
                    # `min_corners_covered` corners covered.
                    model.Add(
                        sum(corner_covered_per_item) >= min_corners_covered
                    ).OnlyEnforceIf([placed[i], on_floor.Not()])
    else:
        # Anti-floating constraint (light) — at least one supporter.
        for i in range(n):
            on_floor = model.NewBoolVar(f"floor{i}")
            model.Add(z_var[i] == 0).OnlyEnforceIf(on_floor)
            model.Add(z_var[i] >= 1).OnlyEnforceIf(on_floor.Not())
            supporters: List[cp_model.IntVar] = []
            for j in range(n):
                if j == i:
                    continue
                # j supports i: z_j + dz_j == z_i, footprints overlap on x AND y.
                same_pallet = model.NewBoolVar(f"sp{i}_{j}")
                sp_lits = []
                for p in range(P):
                    both_p = model.NewBoolVar(f"sp{i}_{j}_p{p}")
                    model.AddBoolAnd([assigned[i][p], assigned[j][p]]).OnlyEnforceIf(both_p)
                    model.AddBoolOr([assigned[i][p].Not(), assigned[j][p].Not()]).OnlyEnforceIf(both_p.Not())
                    sp_lits.append(both_p)
                model.AddBoolOr(sp_lits).OnlyEnforceIf(same_pallet)
                model.AddBoolAnd([s.Not() for s in sp_lits]).OnlyEnforceIf(same_pallet.Not())

                z_aligned = model.NewBoolVar(f"za{i}_{j}")
                model.Add(z_var[j] + dz_var[j] == z_var[i]).OnlyEnforceIf(z_aligned)
                model.Add(z_var[j] + dz_var[j] != z_var[i]).OnlyEnforceIf(z_aligned.Not())

                x_overlap = model.NewBoolVar(f"xo{i}_{j}")
                x_o_lit1 = model.NewBoolVar(f"xo1{i}_{j}")
                x_o_lit2 = model.NewBoolVar(f"xo2{i}_{j}")
                model.Add(x_var[i] + 1 <= x_var[j] + dx_var[j]).OnlyEnforceIf(x_o_lit1)
                model.Add(x_var[j] + 1 <= x_var[i] + dx_var[i]).OnlyEnforceIf(x_o_lit2)
                model.AddBoolAnd([x_o_lit1, x_o_lit2]).OnlyEnforceIf(x_overlap)
                model.AddBoolOr([x_o_lit1.Not(), x_o_lit2.Not()]).OnlyEnforceIf(x_overlap.Not())

                y_overlap = model.NewBoolVar(f"yo{i}_{j}")
                y_o_lit1 = model.NewBoolVar(f"yo1{i}_{j}")
                y_o_lit2 = model.NewBoolVar(f"yo2{i}_{j}")
                model.Add(y_var[i] + 1 <= y_var[j] + dy_var[j]).OnlyEnforceIf(y_o_lit1)
                model.Add(y_var[j] + 1 <= y_var[i] + dy_var[i]).OnlyEnforceIf(y_o_lit2)
                model.AddBoolAnd([y_o_lit1, y_o_lit2]).OnlyEnforceIf(y_overlap)
                model.AddBoolOr([y_o_lit1.Not(), y_o_lit2.Not()]).OnlyEnforceIf(y_overlap.Not())

                supports_ij = model.NewBoolVar(f"sup{i}_{j}")
                model.AddBoolAnd([
                    placed[j], same_pallet, z_aligned, x_overlap, y_overlap,
                ]).OnlyEnforceIf(supports_ij)
                model.AddBoolOr([
                    placed[j].Not(), same_pallet.Not(), z_aligned.Not(),
                    x_overlap.Not(), y_overlap.Not(),
                ]).OnlyEnforceIf(supports_ij.Not())
                supporters.append(supports_ij)

            # If placed[i], then on_floor OR at least one supporter.
            model.AddBoolOr([on_floor] + supporters).OnlyEnforceIf(placed[i])

    # Objective: minimize total volume excluded (= unpacked volume).
    # Equivalent to maximizing packed volume.
    model.Minimize(sum(_int(boxes[i].volume) * unpacked[i] for i in range(n)))

    # Q3: warm-start from heuristic result.
    # CP-SAT accepts solution hints via model.AddHint. A complete and
    # feasible hint substantially cuts the time-to-first-feasible-solution
    # (often by 10-100x) — the search starts in a known-good region
    # instead of from scratch.
    if warm_start is not None:
        # Build (box_id → (pallet_idx, placement)) mapping from the warm
        # start so we can pull placement coords / rotation by box id.
        placements_by_id: dict = {}
        for p_idx, st in enumerate(warm_start.pallets):
            if p_idx >= P:
                # Warm-start has more pallets than the MIP allows; skip
                # those placements (they'll be hinted as unpacked).
                continue
            for placement in st.placements:
                placements_by_id[placement.box.id] = (p_idx, placement)
        for i, b in enumerate(boxes):
            if b.id in placements_by_id:
                p_idx, placement = placements_by_id[b.id]
                model.AddHint(placed[i], 1)
                model.AddHint(unpacked[i], 0)
                for p in range(P):
                    model.AddHint(assigned[i][p], 1 if p == p_idx else 0)
                model.AddHint(x_var[i], _int(placement.x))
                model.AddHint(y_var[i], _int(placement.y))
                model.AddHint(z_var[i], _int(placement.z))
                # Find which rotation index matches.
                for k, r in enumerate(rotations[i]):
                    model.AddHint(rot_vars[i][k], 1 if r == placement.rotation else 0)
                dx_h = _int(placement.dx)
                dy_h = _int(placement.dy)
                dz_h = _int(placement.dz)
                model.AddHint(dx_var[i], dx_h)
                model.AddHint(dy_var[i], dy_h)
                model.AddHint(dz_var[i], dz_h)
            else:
                # In warm-start as unpacked (or not present at all).
                model.AddHint(placed[i], 0)
                model.AddHint(unpacked[i], 1)

    # Solve.
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_search_workers = num_workers
    status = solver.Solve(model)
    # Stash the status so callers can distinguish OPTIMAL (proven) from
    # FEASIBLE (best-found within budget).
    mip_polish.last_status = (
        "OPTIMAL" if status == cp_model.OPTIMAL else
        "FEASIBLE" if status == cp_model.FEASIBLE else
        "INFEASIBLE" if status == cp_model.INFEASIBLE else
        "UNKNOWN"
    )
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    # Decode solution.
    pallets_state: List[PalletState] = []
    state_by_p: List[PalletState] = []
    for p in range(P):
        st = PalletState(pallet, f"P{p+1:03d}", config)
        state_by_p.append(st)
    unpacked_boxes: List[Box] = []
    for i, b in enumerate(boxes):
        if solver.Value(placed[i]) == 0:
            unpacked_boxes.append(b)
            continue
        # Find which pallet.
        for p in range(P):
            if solver.Value(assigned[i][p]) == 1:
                # Find chosen rotation.
                chosen_rot = None
                for k, r in enumerate(rotations[i]):
                    if solver.Value(rot_vars[i][k]) == 1:
                        chosen_rot = r
                        break
                if chosen_rot is None:
                    chosen_rot = rotations[i][0]
                pl = Placement(
                    box=b, rotation=chosen_rot,
                    x=solver.Value(x_var[i]),
                    y=solver.Value(y_var[i]),
                    z=solver.Value(z_var[i]),
                )
                state_by_p[p].placements.append(pl)
                state_by_p[p].total_weight += b.weight
                break

    # Drop empty pallets.
    for st in state_by_p:
        if st.placements:
            pallets_state.append(st)

    raw_result = PackResult(pallets=pallets_state, unpacked=unpacked_boxes)

    # The MIP doesn't currently model support_ratio, CoG, or fragility
    # (modelling those in CP-SAT inflates the model size considerably).
    # Replay-validate each pallet through the real PalletState's full
    # constraint stack: items that fail get demoted to .unpacked. This
    # keeps the MIP candidate validator-clean and trustable in the
    # candidate-set min(quality) pick. The cost is that we may lose some
    # of the MIP's "raw" gain on cases where support is tight — but the
    # candidate-set design means we never regress below v1 anyway.
    rebuilt_pallets: List[PalletState] = []
    demoted: List[Box] = []
    for st in raw_result.pallets:
        new_st = PalletState(pallet, st.pallet_id, config)
        # Replay in bottom-up z order so supporters are committed first.
        for p in sorted(st.placements, key=lambda pl: (pl.z, pl.y, pl.x)):
            if new_st.feasible(p):
                new_st._commit(p)
            else:
                demoted.append(p.box)
        if new_st.placements:
            rebuilt_pallets.append(new_st)

    return PackResult(pallets=rebuilt_pallets,
                      unpacked=unpacked_boxes + demoted)
