"""
pallet_packer.repair — deterministic post-hoc repair of load violations.

Policy (hardening round 3, F23 / ADR D17): a result that fails validate()
must never ship silently. After the round-3 engine fixes the engines are
believed correct; this module is the LAST-RESORT backstop for residual or
unknown bugs. It deterministically strips the placements feeding an
overloaded carrier — topmost first, so nothing is ever left hovering —
into `unpacked`, until no load violation remains.

Deliberately independent of the placement engines: it recomputes loads
from FINAL geometry with the same model as validate.py (direct or
transitive per config), so repair and validate always agree. Pure Python,
no decoder twins involved.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import EPS, load_tol, Pallet, PackerConfig, Placement
from .packer import PackResult, PalletState


def _contact_area(top: Placement, below: Placement) -> float:
    """Area where `top`'s bottom face rests on `below`'s top face
    (mirrors validate._contact_area)."""
    if abs(top.z - below.z2) > EPS:
        return 0.0
    ox = max(0.0, min(top.x2, below.x2) - max(top.x, below.x))
    oy = max(0.0, min(top.y2, below.y2) - max(top.y, below.y))
    return ox * oy


def _loads(placements: List[Placement], transitive: bool
           ) -> Dict[int, float]:
    """Final-geometry load resting on each placement (id(p) -> kg).

    Same accumulation as validate.py section 6: process boxes top-down
    (descending bottom z); each box's outflow (own weight, plus its inflow
    when transitive) is distributed over its supporters in proportion to
    contact area.
    """
    inflow: Dict[int, float] = {id(p): 0.0 for p in placements}
    for p in sorted(placements, key=lambda q: -q.z):
        out = p.box.weight + (inflow[id(p)] if transitive else 0.0)
        if p.z <= EPS or out <= 0.0:
            continue
        sups = [(q, _contact_area(p, q)) for q in placements if q is not p]
        sups = [(q, a) for q, a in sups if a > EPS]
        total = sum(a for _, a in sups)
        if total <= 0.0:
            continue  # hovering box — a geometry error, not repairable here
        for q, a in sups:
            inflow[id(q)] += out * (a / total)
    return inflow


def _feeders(carrier: Placement, placements: List[Placement]
             ) -> List[Placement]:
    """Transitive upward closure: every placement resting (directly or
    indirectly) on `carrier`. Any box in this set contributes load to the
    carrier under the transitive model, and the set is closed upward — its
    topmost member has nothing resting on it, so removing that member never
    leaves another box hovering."""
    feeders: List[Placement] = []
    seen = {id(carrier)}
    frontier = [carrier]
    while frontier:
        nxt: List[Placement] = []
        for base in frontier:
            for q in placements:
                if id(q) in seen:
                    continue
                if _contact_area(q, base) > EPS:
                    seen.add(id(q))
                    feeders.append(q)
                    nxt.append(q)
        frontier = nxt
    return feeders


def repair_load_violations(
    result: PackResult,
    pallet: Pallet,
    config: Optional[PackerConfig] = None,
) -> Tuple[PackResult, List[str]]:
    """Strip load-violating riders until no carrier exceeds its
    max_load_on_top. Mutates `result` in place and returns it with a list
    of human-readable action strings (empty when nothing was overloaded).

    Deterministic: the worst-overloaded carrier is fixed first (ties by
    pallet id, then carrier z/x/y/box id); the removed feeder is the
    topmost one (max bottom z, ties by x, y, box id). Each iteration
    removes exactly one placement, so the loop is bounded by the number
    of placements.

    Note (round 4, R8): the feeder set is always the TRANSITIVE upward
    closure, even under the direct load model — so the topmost-first order
    may strip a sibling tower's top before the box that actually feeds the
    overload. Deliberate: topmost-first is the only order that can never
    leave a box hovering, and determinism matters more than minimality
    here. Only load (max_load_on_top) violations are repairable; weight-cap
    or geometry violations are surfaced as warnings instead.
    """
    cfg = config or PackerConfig()
    transitive = bool(getattr(cfg, "transitive_load_bearing", False))
    actions: List[str] = []
    guard = sum(len(st.placements) for st in result.pallets) + 1

    for _ in range(guard):
        worst: Optional[Tuple[float, PalletState, Placement, float, float]] \
            = None
        for st in result.pallets:
            loads = _loads(st.placements, transitive)
            for p in st.placements:
                lim = p.box.max_load_on_top
                if lim is None or math.isinf(lim):
                    continue
                over = loads[id(p)] - (lim + load_tol(lim))
                if over <= 0.0:
                    continue
                key = (-over, st.pallet_id, p.z, p.x, p.y, p.box.id)
                if worst is None or key < worst[0]:
                    worst = (key, st, p, loads[id(p)], lim)
        if worst is None:
            break
        _, st, carrier, load, lim = worst
        feeders = _feeders(carrier, st.placements)
        if not feeders:
            # Overloaded with nothing resting on it — geometry too broken
            # for a load repair; leave it to the residual warnings.
            break
        # Topmost feeder; among equal z prefer the smallest (x, y, box id).
        top_z = max(q.z for q in feeders)
        candidates = sorted((q for q in feeders if q.z == top_z),
                            key=lambda q: (q.x, q.y, q.box.id))
        victim = candidates[0]
        st.placements.remove(victim)
        st.total_weight -= victim.box.weight
        result.unpacked.append(victim.box)
        actions.append(
            f"removed {victim.box.id} from {st.pallet_id}: it fed "
            f"{carrier.box.id}, which carried {load:.3f} > "
            f"max_load_on_top {lim:g}")
    # Drop pallets emptied by the repair.
    result.pallets[:] = [st for st in result.pallets if st.placements]
    return result, actions
