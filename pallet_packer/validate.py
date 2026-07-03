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
(when stability semantics are active), and the CoG envelope for pallets
with EXPLICIT cog ranges. Load bearing follows the config (round 2, F19):
transitive accumulation when cfg.transitive_load_bearing, else the
historical direct-supporter bound.

Deliberate gating (so this stays no-stricter-than-the-engine for every
caller):
  * Support/centroid checks apply per box only when any stability semantic
    is active (config.support_ratio > 0, config.require_centroid_supported,
    or the box's own requires_full_support). Pure-geometric runs (BR
    benchmarks: sr=0, no centroid) legitimately float boxes — the engine
    does not enforce support there and neither do we.
  * The CoG check applies only when the pallet carries EXPLICIT
    cog_x_range/cog_y_range. The config-fraction envelope default (0.25)
    is NOT checked here: the geometric decoder path never enforces it, so
    validating it would reject engine-legal geometric packings.
"""
from __future__ import annotations

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
                dxc = min(p.x2, float(pallet.length)) - max(p.x, 0.0)
                dyc = min(p.y2, float(pallet.width)) - max(p.y, 0.0)
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
        # 7. CoG envelope — only for EXPLICIT pallet cog ranges (see module
        #    docstring). Mirrors PalletState._cog_ok on the final state,
        #    including its min-load gate for finite weight caps.
        if (pallet.cog_x_range or pallet.cog_y_range) and placements:
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
    return errors
