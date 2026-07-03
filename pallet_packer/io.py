"""
pallet_packer.io — serialize a PackResult to the agreed JSON schema.

Read by the matplotlib visualizers and any downstream consumer that needs
the geometry + supporter graph + per-pallet CoG / utilization summary.
"""
from __future__ import annotations

import json

from .models import EPS, Pallet
from .packer import PackResult


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
            # Round 5 (R5): "floor" only for actual floor boxes. A z>0 box
            # with no supporters (a residual floater in a plan served with
            # warnings) reports [] — it used to claim ["floor"] while its
            # own support_ratio said 0.0.
            if sups:
                supported_by = [s.box.id for s, _ in sups]
            elif p.z <= EPS:
                supported_by = ["floor"]
            else:
                supported_by = []
            supports = [
                q.box.id for q in st.placements
                if any(abs(q.z - p.z2) < EPS and
                       max(0, min(q.x2, p.x2) - max(q.x, p.x)) *
                       max(0, min(q.y2, p.y2) - max(q.y, p.y)) > EPS
                       for _ in [None])
            ]
            footprint = p.dx * p.dy
            if footprint > 0 and p.z > EPS:
                support_ratio = sum(a for _, a in sups) / footprint
            elif p.z <= EPS and pallet.max_overhang > 0 and footprint > 0:
                # Round 5 (R5): under overhang a floor box may legally sit
                # at partial deck contact — report the ACTUAL contact
                # fraction (the quantity the engine/validator enforce)
                # instead of a flattering hardcoded 1.0.
                dcx = max(0.0, min(p.x2, float(pallet.length)) - max(p.x, 0.0))
                dcy = max(0.0, min(p.y2, float(pallet.width)) - max(p.y, 0.0))
                support_ratio = min(1.0, (dcx * dcy) / footprint)
            else:
                support_ratio = 1.0
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

