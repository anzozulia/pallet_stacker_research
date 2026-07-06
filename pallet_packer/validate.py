"""
pallet_packer.validate — independent constraint check for a PackResult.

By design this lives outside the placement engine. If any bug in the
engine produces an invalid placement (overlap, floating box, weight
overrun, etc.), the validator catches it. Every entry point in the
algorithm returns a result that has been replay-validated through this
function — empty error list is non-negotiable.

Hardened (2026-07, hardening plan A2) to mirror the ENGINE's feasibility
stack instead of a weaker approximation, because postprocess.py uses this
function as the safety gate for its geometry edits. Additions: per-box
requires_full_support, centroid-over-supporter, zero-contact-is-floating
(when stability semantics are active), the CoG envelope for pallets
with EXPLICIT cog ranges, and (round 6, F30) the floor centroid-over-deck
rule under overhang — mirroring the engines' toppling gate, active only
when cfg.require_centroid_supported. Round 8, F36 generalises F30 from a
single floor box to a connected sub-ASSEMBLY: under overhang each rigid
assembly's weighted CoG must project within the convex hull of its own
deck-contact region, else the stack topples even though every per-box
check and the whole-pallet CoG envelope pass. Load bearing follows the
config (round 2, F19): transitive accumulation when
cfg.transitive_load_bearing, else the historical direct-supporter bound.

Deliberate gating (so this stays no-stricter-than-the-engine for every
caller):
  * Support/centroid checks apply per box only when any stability semantic
    is active (config.support_ratio > 0, config.require_centroid_supported,
    or the box's own requires_full_support). Pure-geometric runs (BR
    benchmarks: sr=0, no centroid) legitimately float boxes — the engine
    does not enforce support there and neither do we.
  * The CoG check applies when the pallet carries EXPLICIT
    cog_x_range/cog_y_range, and (round 7, F35) when overhang is active with
    a finite config-fraction envelope — the regime where the engine's
    constraint decoders DO enforce the running-CoG bound. It is still NOT
    checked for the config-fraction default on a purely geometric (no
    overhang) packing: the geometric decoder path never enforces it, so
    validating it there would reject engine-legal geometric packings.
    (round 8, F37) It is also skipped for a single-placement pallet: the
    engines never CoG-check the first box on a pallet (v2 _cog_ok returns
    True when `not self.placements`; the JIT new-bin path commits the seed
    box with no envelope check), so flagging a lone corner box would reject
    an engine-legal placement.
  * The per-assembly toppling check (round 8, F36) applies only under
    overhang with cfg.require_centroid_supported — exactly F30's gate. It is
    provably inert without overhang: a floor box's deck-contact rectangle is
    then its full footprint, so the per-box centroid-over-supporter rule
    inductively keeps every assembly's weighted CoG inside its deck-contact
    hull. Assemblies are joined by the VERTICAL resting-on relation only —
    two stacks touching at a side face brace nothing against tipping.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .models import EPS, load_tol, Pallet, PackerConfig, Placement
from .packer import PackResult


def _contact_area(top: Placement, below: Placement) -> float:
    """Area where `top`'s bottom face rests on `below`'s top face."""
    if abs(top.z - below.z2) > EPS:
        return 0.0
    ox = max(0.0, min(top.x2, below.x2) - max(top.x, below.x))
    oy = max(0.0, min(top.y2, below.y2) - max(top.y, below.y))
    return ox * oy


def _supporters(p: Placement, placements: List[Placement]
                ) -> List[Tuple[Placement, float]]:
    out = []
    for q in placements:
        if q is p:
            continue
        a = _contact_area(p, q)
        if a > EPS:
            out.append((q, a))
    return out


def _assemblies(placements: List[Placement],
                sup_cache: Dict[int, List[Tuple[Placement, float]]]
                ) -> List[List[Placement]]:
    """Partition placements into rigid assemblies — connected components under
    the VERTICAL resting-on relation (``p`` rests on each supporter in
    ``sup_cache[id(p)]``). Boxes touching only at a side face are NOT joined:
    a vertical side face transmits no restraint against tipping outward, so
    two side-by-side stacks each stand (or topple) on their own (round 8,
    F36 — the corrected model; the naive face-adjacency union would wrongly
    merge a disjoint counterweight and mask a local tip)."""
    parent: Dict[int, int] = {id(p): id(p) for p in placements}

    def find(a: int) -> int:
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:          # path compression
            parent[a], a = root, parent[a]
        return root

    for p in placements:
        for s, _ in sup_cache[id(p)]:
            ra, rb = find(id(p)), find(id(s))
            if ra != rb:
                parent[ra] = rb
    comps: Dict[int, List[Placement]] = {}
    for p in placements:
        comps.setdefault(find(id(p)), []).append(p)
    return list(comps.values())


def _convex_hull(points: List[Tuple[float, float]]
                 ) -> List[Tuple[float, float]]:
    """Counter-clockwise convex hull (Andrew's monotone chain). Collinear
    points are dropped; 1-2 unique points return as a degenerate hull."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def _point_in_hull(pt: Tuple[float, float],
                   hull: List[Tuple[float, float]]) -> bool:
    """True if ``pt`` is inside or on a CCW convex ``hull`` (inclusive, with
    an EPS margin scaled per edge so the tolerance is a real distance, not an
    area). Degenerate hulls (point, segment) fall back to a bounding-box
    membership test."""
    x, y = pt
    if len(hull) < 3:
        xs = [h[0] for h in hull] or [0.0]
        ys = [h[1] for h in hull] or [0.0]
        return (min(xs) - EPS <= x <= max(xs) + EPS and
                min(ys) - EPS <= y <= max(ys) + EPS)
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        bx, by = hull[(i + 1) % n]
        # (B-A) x (P-A): >= 0 means P is left of / on the CCW edge (inside).
        c = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        edge = math.hypot(bx - ax, by - ay)
        if c < -EPS * max(edge, 1.0):
            return False
    return True


def validate(result: PackResult, pallet: Pallet,
             config: Optional[PackerConfig] = None) -> List[str]:
    """Return a list of constraint violations (empty list means all good).

    Re-checks the produced packing against geometry, weight, support,
    load-bearing, rotation, and (explicit) CoG constraints — independent of
    the placement engine, so any bug in the engine should produce a
    non-empty list here.
    """
    cfg = config or PackerConfig()
    errors: List[str] = []
    for st in result.pallets:
        placements = st.placements
        # 1. Pallet boundaries
        for p in placements:
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
        for i, a in enumerate(placements):
            for b in placements[i + 1:]:
                if a.overlaps(b):
                    errors.append(
                        f"{st.pallet_id}: {a.box.id} overlaps {b.box.id}"
                    )
        # 3. Weight budget
        total = sum(p.box.weight for p in placements)
        if total > pallet.max_weight + load_tol(pallet.max_weight):
            errors.append(
                f"{st.pallet_id}: total weight {total} > limit {pallet.max_weight}"
            )
        # 4. Support / no-floating / full-support / centroid
        #    (mirrors PalletState.feasible step 3, packer.py)
        sup_cache: Dict[int, List[Tuple[Placement, float]]] = {}
        for p in placements:
            sup_cache[id(p)] = _supporters(p, placements) if p.z > EPS else []
        for p in placements:
            if p.z <= EPS:
                # Floor placement. Under overhang the box can extend past
                # the deck — it must still rest ON it (F17): deck-contact
                # area >= the effective support ratio of its footprint.
                # Mirrors the engines; inert when overhang is off.
                if not cfg.allow_pallet_overhang:
                    continue
                stability_active = (cfg.support_ratio > 0.0
                                    or cfg.require_centroid_supported
                                    or p.box.requires_full_support)
                if not stability_active:
                    continue      # pure-geometric mode (BR): floats allowed
                fp = p.dx * p.dy
                x_hi = min(p.x2, float(pallet.length))
                y_hi = min(p.y2, float(pallet.width))
                dxc = x_hi - max(p.x, 0.0)
                dyc = y_hi - max(p.y, 0.0)
                contact = dxc * dyc if (dxc > 0 and dyc > 0) else 0.0
                min_support = (1.0 if p.box.requires_full_support
                               else cfg.support_ratio)
                if contact <= EPS or (
                        min_support > 0.0 and
                        (fp <= 0 or contact / fp < min_support - EPS)):
                    errors.append(
                        f"{st.pallet_id}: {p.box.id} floor placement has "
                        f"only {contact / fp if fp > 0 else 0.0:.0%} deck "
                        f"contact (min {min_support:.0%})"
                    )
                # F30 (round 6): centroid over the deck-contact rectangle,
                # mirroring the engines — a floor box whose centre of mass
                # projects past the deck edge tips over on placement.
                if cfg.require_centroid_supported:
                    cx = p.x + p.dx / 2.0
                    cy = p.y + p.dy / 2.0
                    if cx > x_hi + EPS or cy > y_hi + EPS:
                        errors.append(
                            f"{st.pallet_id}: {p.box.id} floor centroid "
                            f"({cx},{cy}) hangs past the deck edge under "
                            f"overhang — the box would topple"
                        )
                continue
            stability_active = (cfg.support_ratio > 0.0
                                or cfg.require_centroid_supported
                                or p.box.requires_full_support)
            if not stability_active:
                continue          # pure-geometric mode: engine doesn't enforce
            sups = sup_cache[id(p)]
            supported = sum(a for _, a in sups)
            footprint = p.dx * p.dy
            if supported <= EPS:
                errors.append(
                    f"{st.pallet_id}: {p.box.id} is floating (no contact)"
                )
                continue
            min_support = (1.0 if p.box.requires_full_support
                           else cfg.support_ratio)
            if footprint <= 0 or supported / footprint < min_support - EPS:
                errors.append(
                    f"{st.pallet_id}: {p.box.id} is floating "
                    f"(support ratio {supported / footprint:.2f} "
                    f"< {min_support})"
                )
            if cfg.require_centroid_supported:
                cx = p.x + p.dx / 2.0
                cy = p.y + p.dy / 2.0
                over = any(s.x - EPS <= cx <= s.x2 + EPS and
                           s.y - EPS <= cy <= s.y2 + EPS
                           for s, _ in sups)
                if not over:
                    errors.append(
                        f"{st.pallet_id}: {p.box.id} centroid ({cx},{cy}) "
                        f"is not over any supporter"
                    )
        # 5. Rotation allowed
        for p in placements:
            if p.rotation not in p.box.allowed_rotations:
                errors.append(
                    f"{st.pallet_id}: {p.box.id} uses disallowed rotation "
                    f"{p.rotation.name}"
                )
        # 6. Load bearing, weighted by contact area. Two accumulation models
        #    mirroring the engines (hardening round 2, F19):
        #    - transitive_load_bearing OFF (default): DIRECT supporters only —
        #      the BRKGA decoders' historical commit model
        #      (_apply_load_contribution_njit) adds each box's share to its
        #      direct supporters without propagating down, so decoder-legal
        #      results only guarantee the direct bound (was finding F15).
        #    - transitive_load_bearing ON: each box's outflow (own weight +
        #      everything that arrived on it) propagates down the whole
        #      support chain — the v2 engine's _propagate_load model, now
        #      shared by the decoders' transitive commit sibling. Boxes are
        #      processed top-down (descending bottom-z), so every box's
        #      arriving load is final before it distributes.
        if cfg.enforce_load_bearing:
            transitive = bool(getattr(cfg, "transitive_load_bearing", False))
            load_on: Dict[int, float] = {id(p): 0.0 for p in placements}
            for p in sorted(placements, key=lambda q: -q.z):
                sups = sup_cache[id(p)]
                sup_area = sum(a for _, a in sups)
                if sup_area <= 0:
                    continue
                outflow = p.box.weight + (load_on[id(p)] if transitive else 0.0)
                for s, a in sups:
                    load_on[id(s)] += outflow * (a / sup_area)
            for q in placements:
                if (load_on[id(q)] > q.box.max_load_on_top
                        + load_tol(q.box.max_load_on_top)):
                    errors.append(
                        f"{st.pallet_id}: {q.box.id} carries "
                        f"{load_on[id(q)]:.2f} kg > max_load_on_top "
                        f"{q.box.max_load_on_top}"
                    )
        # 7. CoG envelope — for EXPLICIT pallet cog ranges, and (round 7,
        #    F35 / ADR D20) for the config-fraction envelope UNDER OVERHANG.
        #    Mirrors PalletState._cog_ok on the final state, including its
        #    min-load gate for finite weight caps. The overhang gate keeps
        #    this no-stricter-than-the-engine: overhang forces the
        #    constraint-aware decoder path, which enforces the same running-CoG
        #    envelope (a fraction < 1.0 with cog_active); a purely geometric
        #    caller (no overhang) never has the engine enforce CoG, so we
        #    don't check it here either.
        cog_under_overhang = (cfg.allow_pallet_overhang
                              and cfg.cog_envelope_fraction < 1.0)
        # F37 (round 8): skip a single-placement pallet — the engines never
        # CoG-check the first box (v2 _cog_ok `not self.placements`; the JIT
        # new-bin path seeds the box with no envelope check), so flagging a
        # lone corner box would reject an engine-legal placement.
        if (pallet.cog_x_range or pallet.cog_y_range or cog_under_overhang) \
                and len(placements) > 1:
            if total > 0:
                max_w = (pallet.max_weight
                         if pallet.max_weight < float("inf") else None)
                gated = (max_w is not None and
                         total < cfg.cog_check_min_load_fraction * max_w)
                if not gated:
                    cx = sum(p.box.weight * (p.x + p.dx / 2.0)
                             for p in placements) / total
                    cy = sum(p.box.weight * (p.y + p.dy / 2.0)
                             for p in placements) / total
                    L, W = pallet.length, pallet.width
                    frac = cfg.cog_envelope_fraction
                    x_range = pallet.cog_x_range or (
                        L / 2.0 - frac * L, L / 2.0 + frac * L)
                    y_range = pallet.cog_y_range or (
                        W / 2.0 - frac * W, W / 2.0 + frac * W)
                    if not (x_range[0] - EPS <= cx <= x_range[1] + EPS and
                            y_range[0] - EPS <= cy <= y_range[1] + EPS):
                        errors.append(
                            f"{st.pallet_id}: CoG ({cx:.1f},{cy:.1f}) outside "
                            f"envelope x{x_range} y{y_range}"
                        )
        # 8. Per-assembly toppling (round 8, F36 / ADR D21). Under overhang a
        #    connected sub-assembly rooted on overhanging floor boxes can tip
        #    over the deck edge while a disjoint counterweight keeps the
        #    whole-pallet CoG (§7) central. Each rigid assembly's weighted CoG
        #    must project within the convex hull of ITS OWN deck-contact
        #    region. Gate == F30 (overhang + require_centroid_supported);
        #    provably inert otherwise (module docstring). Assemblies join by
        #    vertical support only, so a side-abutting counterweight stays a
        #    separate body.
        if cfg.allow_pallet_overhang and cfg.require_centroid_supported \
                and len(placements) > 1:
            for members in _assemblies(placements, sup_cache):
                total_w = sum(m.box.weight for m in members)
                if total_w <= 0:
                    continue                 # weightless: nothing to topple
                corners: List[Tuple[float, float]] = []
                for m in members:
                    if m.z > EPS:
                        continue             # only floor boxes touch the deck
                    x_lo, y_lo = max(m.x, 0.0), max(m.y, 0.0)
                    x_hi = min(m.x2, float(pallet.length))
                    y_hi = min(m.y2, float(pallet.width))
                    if x_hi > x_lo and y_hi > y_lo:
                        corners += [(x_lo, y_lo), (x_hi, y_lo),
                                    (x_hi, y_hi), (x_lo, y_hi)]
                if not corners:
                    continue                 # floating assembly — §4 owns it
                cx = sum(m.box.weight * (m.x + m.dx / 2.0)
                         for m in members) / total_w
                cy = sum(m.box.weight * (m.y + m.dy / 2.0)
                         for m in members) / total_w
                if not _point_in_hull((cx, cy), _convex_hull(corners)):
                    ids = ",".join(sorted(m.box.id for m in members))
                    errors.append(
                        f"{st.pallet_id}: sub-assembly [{ids}] CoG "
                        f"({cx:.1f},{cy:.1f}) projects past its deck-contact "
                        f"region under overhang — the stack topples"
                    )
    return errors
