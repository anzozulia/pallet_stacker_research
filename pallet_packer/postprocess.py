"""
pallet_packer.postprocess — realism post-passes on a finished PackResult.

Both passes are rigid-geometry cleanups that run AFTER the solver has committed
to a layout. They never change which boxes are packed, never change z, and are
replay-validated (validate.py) with a full per-pallet revert on any violation.

  align_orientations_pass   Re-rotate same-SKU boxes to the dominant
      orientation of their (SKU, z-level) when the swap is
      feasibility-preserving. Fixes "identical cartons randomly rotated
      inside one layer" (the decoders re-derive rotation greedily per
      placement, so the winning rotation flips with the EMS landscape).
  recenter_pass             Rigid x/y translation of the whole layout so the
      load is centred on the deck (weighted CoG when the load has weight,
      else the footprint bbox midpoint). Fixes "everything jammed into the
      (0,0) corner": corner anchoring is what makes EMS packing dense, so
      the remedy is a final translation, not per-box centering. z is never
      touched; overhang is never created by the shift.

Gated by PackerConfig.align_orientations / PackerConfig.recenter_layout (both
default OFF so library/benchmark callers keep bit-identical output; the API
service enables them). Applied exactly once per solve by the brkga_pack_v35
wrapper in _brkga_core.driver — internal restart/group recursions call the
un-postprocessed _impl.
"""
from __future__ import annotations

import logging
import math
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from .models import EPS, Pallet, PackerConfig, Placement
from .packer import PackResult, PalletState, regen_top_load
from .validate import validate

logger = logging.getLogger(__name__)

# Wall-clock caps. The post-pass runs outside the solver's time_limit_s
# accounting, and the service's hard-kill margin above the soft budget is only
# ~30 s. apply_postprocess shares ONE global budget across all pallets
# (hardening plan C6 — a per-pallet cap times an unbounded pallet count could
# eat the whole margin); _ALIGN_BUDGET_S remains the default for direct
# single-pallet align_orientations_pass calls.
_POSTPROCESS_BUDGET_S = 4.0
_ALIGN_BUDGET_S = 2.0


def _sku_key(box) -> tuple:
    """SKU identity, attribute-based (never object identity — v2-origin
    results can contain Box copies). Mirrors the key used by
    _brkga_core.precompute.precompute_box_dims_and_sku so the post-pass and
    the fitness OI term agree on what "the same SKU" means."""
    rot_key = tuple(sorted(r.name for r in box.allowed_rotations))
    m = getattr(box, "max_load_on_top", math.inf)
    mlot_key = round(m, 6) if (m is not None and math.isfinite(m)) else math.inf
    rfs_key = 1 if getattr(box, "requires_full_support", False) else 0
    return (round(box.length, 6), round(box.width, 6), round(box.height, 6),
            round(box.weight, 6), mlot_key, rfs_key, rot_key)


def _dims_key(dims: Tuple[float, float, float]) -> tuple:
    return (round(dims[0], 6), round(dims[1], 6), round(dims[2], 6))


def _xy_overlap(a: Placement, b: Placement) -> float:
    ox = max(0.0, min(a.x2, b.x2) - max(a.x, b.x))
    oy = max(0.0, min(a.y2, b.y2) - max(a.y, b.y))
    return ox * oy


def align_orientations_pass(st: PalletState, pallet: Pallet,
                            config: PackerConfig,
                            time_budget_s: float = _ALIGN_BUDGET_S) -> int:
    """Re-rotate deviant same-SKU boxes to their (SKU, z-level) dominant
    orientation when the swap is feasibility-preserving. Returns the number
    of applied swaps.

    Dominance is per z-level (not per pallet) on purpose: it matches how
    humans stack and leaves deliberately interlocked alternating layers
    alone. Orientation identity is the post-rotation dims tuple, not the
    Rotation enum — two rotations can produce identical extents.
    """
    placements = st.placements
    if len(placements) < 2:
        return 0
    t0 = time.monotonic()

    groups: Dict[tuple, List[Placement]] = defaultdict(list)
    for p in placements:
        groups[(round(p.z, 6), _sku_key(p.box))].append(p)

    swaps = 0
    for key in sorted(groups.keys()):          # deterministic order
        members = groups[key]
        if len(members) < 2:
            continue
        counts = Counter(_dims_key(m.dims) for m in members)
        if len(counts) < 2:
            continue                           # already uniform at this level
        # Most common dims tuple; ties broken lexicographically smallest so
        # repeated application is a no-op (idempotence).
        dominant = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for m in sorted(members, key=lambda p: (p.x, p.y)):
            if _dims_key(m.dims) == dominant:
                continue
            if time.monotonic() - t0 > time_budget_s:
                logger.info("align_orientations: time box hit after %d swaps",
                            swaps)
                return swaps
            # Dependents gate (hardening plan A1): if anything rests on this
            # box, skip it entirely. ANY dims change under a dependent is
            # unrepairable locally — a dz change floats every dependent, and
            # a yaw change pulls the footprint out from under it (probe:
            # dependent stranded at support 0.0). A swap to identical dims
            # would be a no-op, and a deviant by definition has different
            # dims than the dominant.
            has_dependents = any(
                q is not m and abs(q.z - m.z2) <= EPS and _xy_overlap(q, m) > EPS
                for q in placements)
            if has_dependents:
                continue
            for rot in m.box.allowed_rotations:
                dims = m.box.dims_for(rot)
                if _dims_key(dims) != dominant:
                    continue
                ov = pallet.max_overhang if config.allow_pallet_overhang else 0.0
                if (m.x + dims[0] > pallet.length + ov + EPS or
                        m.y + dims[1] > pallet.width + ov + EPS or
                        m.z + dims[2] > pallet.height + EPS):
                    continue
                old_rot = m.rotation
                m.rotation = rot
                # Whole-pallet replay validation: covers overlap, own
                # support, dependents' support ratio, and the load-bearing
                # redistribution (validate recomputes it independently, so
                # the stale _top_load cache mid-pass is irrelevant).
                if validate(PackResult(pallets=[st], unpacked=[]),
                            pallet, config):
                    m.rotation = old_rot
                else:
                    swaps += 1
                break   # same dims == same geometry; trying more rotations
                        # with the dominant extents cannot change the verdict
    return swaps


def recenter_pass(st: PalletState, pallet: Pallet,
                  config: PackerConfig) -> Tuple[float, float]:
    """Rigid x/y translation centring the load on the deck. Returns (tx, ty).

    Target: the weighted CoG at the deck centre (bbox midpoint when the load
    is weightless), quantised with an integer floor to preserve the integer
    grid, and clamped so the load stays fully on the deck — the shift never
    creates overhang (a layout already wider than the deck stays flush at 0).

    Idempotent by construction: the target minimum coordinate is derived
    from translation-invariant internal offsets, so a second application
    computes a zero shift.
    """
    placements = st.placements
    if not placements:
        return 0.0, 0.0
    if pallet.cog_x_range is not None or pallet.cog_y_range is not None:
        # The caller owns CoG placement: an explicit range may be off-centre,
        # and shifting the load to the deck centre could violate it (C5/F7).
        # The config-fraction envelope needs no guard here — it is always
        # centred on the deck (L/2 ± frac·L), a rigid shift toward the centre
        # can only move the weight-CoG deeper into it, and weightless loads
        # short-circuit the engine's _cog_ok entirely.
        return 0.0, 0.0

    def _shift(lo, hi, centres_weights, deck):
        span = hi - lo
        total_w = sum(w for _, w in centres_weights)
        if total_w > EPS:
            cog = sum(c * w for c, w in centres_weights) / total_w
        else:
            cog = lo + span / 2.0
        internal = cog - lo                       # translation-invariant
        target_lo = math.floor(deck / 2.0 - internal)
        target_lo = min(max(0.0, float(target_lo)), max(0.0, deck - span))
        return target_lo - lo

    tx = _shift(min(p.x for p in placements),
                max(p.x2 for p in placements),
                [(p.x + p.dx / 2.0, p.box.weight) for p in placements],
                float(pallet.length))
    ty = _shift(min(p.y for p in placements),
                max(p.y2 for p in placements),
                [(p.y + p.dy / 2.0, p.box.weight) for p in placements],
                float(pallet.width))
    if tx or ty:
        for p in placements:
            p.x += tx
            p.y += ty
    return tx, ty


def apply_postprocess(result: PackResult, pallet: Pallet,
                      config: PackerConfig, verbose: bool = False,
                      time_budget_s: float = _POSTPROCESS_BUDGET_S) -> PackResult:
    """Run the enabled realism passes on every pallet, in place.

    Order matters: align first (swaps change the bounding box), then
    recenter. Safety net: a pallet whose layout fails replay validation
    before the passes is skipped; a pallet that fails it after the passes is
    reverted to its pre-pass snapshot. Never raises.

    time_budget_s is ONE wall-clock budget shared by all pallets' align
    passes (C6): once spent, later pallets skip aligning but still get the
    cheap O(N) recenter.
    """
    if not (config.recenter_layout or config.align_orientations):
        return result
    t0 = time.monotonic()
    for st in result.pallets:
        if not st.placements:
            continue
        if validate(PackResult(pallets=[st], unpacked=[]), pallet, config):
            # Engine handed us an already-invalid layout — don't touch it.
            logger.warning("postprocess: pallet %s failed pre-pass validation;"
                           " skipping", st.pallet_id)
            continue
        snapshot = [(p, p.rotation, p.x, p.y) for p in st.placements]
        swaps = 0
        try:
            if config.align_orientations:
                remaining = time_budget_s - (time.monotonic() - t0)
                if remaining > 0.0:
                    swaps = align_orientations_pass(
                        st, pallet, config, time_budget_s=remaining)
                else:
                    logger.info("postprocess: global budget spent; skipping "
                                "align for %s", st.pallet_id)
            if config.recenter_layout:
                recenter_pass(st, pallet, config)
            errors = validate(PackResult(pallets=[st], unpacked=[]),
                              pallet, config)
        except Exception:                          # noqa: BLE001 — never fail a solve
            logger.exception("postprocess: pallet %s raised; reverting",
                             st.pallet_id)
            errors = ["postprocess raised"]
        if errors:
            for p, rot, x, y in snapshot:
                p.rotation, p.x, p.y = rot, x, y
        elif swaps:
            # Rotation swaps changed contact areas; refresh the engine's
            # incremental top-load cache from the final geometry.
            regen_top_load(st)
        if verbose and not errors:
            print(f"  [postprocess] {st.pallet_id}: {swaps} orientation "
                  f"swap(s), recentred")
    return result
