"""
pallet_packer.validate — independent constraint check for a PackResult.

By design this lives outside the placement engine. If any bug in the
engine produces an invalid placement (overlap, floating box, weight
overrun, etc.), the validator catches it. Every entry point in the
algorithm returns a result that has been replay-validated through this
function — empty error list is non-negotiable.
"""
from __future__ import annotations

from typing import List, Optional

from .models import EPS, Pallet, PackerConfig
from .packer import PackResult


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
