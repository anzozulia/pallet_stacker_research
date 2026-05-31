"""
pallet_packer._brkga_core.precompute — per-box and per-pallet array
preparation.

These functions run once per `brkga_pack_v35` invocation and translate
the Python `Box` / `Pallet` / `PackerConfig` objects into typed numpy
arrays that the JIT decoders can consume.

  precompute_box_dims_and_sku       SKU detection + rotation dims
  precompute_constraint_arrays      weights, max_load_on_top, RFS
  precompute_cog_envelope           CoG bounds derived from pallet/config
"""
from __future__ import annotations

import math
from typing import List

import numpy as np

from ..models import Box, Pallet, PackerConfig
from ..brkga_v3_fast import precompute_box_dims


# Sentinel: "no limit" stored as a large finite number so JIT comparisons
# work without nan/inf-aware paths.
_NO_LIMIT = 1e18


def precompute_box_dims_and_sku(boxes: List[Box]) -> tuple:
    """Extended precompute: also returns sku_id_per_box for block extension.

    SKU is keyed by (length, width, height, weight, allowed-rotation set).
    Boxes with the same SKU can be packed as a composite block.
    """
    n = len(boxes)
    n_rots_arr, dims_all = precompute_box_dims(boxes)
    sku_id_per_box = np.zeros(n, dtype=np.int64)
    sku_to_id: dict = {}
    for i, b in enumerate(boxes):
        # Key ignores rotation flags - same SKU = same dimensions + weight
        # plus same rotation set (so rotations match in the block).
        #
        # Also key on max_load_on_top + requires_full_support: block-building
        # stacks same-SKU units into a composite grid using a SINGLE (seed)
        # box's load limit. If a fragile box (max_load_on_top=0) shared a SKU
        # with a sturdy box of identical size, the block would stack onto the
        # fragile one and crush it (verified: docs/reports/30_verification.md).
        # Keying on mlot/rfs guarantees every block has a uniform load limit,
        # which is exactly what the block decoder's stack-height cap assumes.
        # Geometric/weightless workloads have mlot=inf + rfs=False for every
        # box, so the SKU partition is unchanged -> bit-identical there.
        rot_key = tuple(sorted(r.name for r in b.allowed_rotations))
        m = getattr(b, 'max_load_on_top', float('inf'))
        mlot_key = round(m, 6) if (m is not None and math.isfinite(m)) else float('inf')
        rfs_key = 1 if getattr(b, 'requires_full_support', False) else 0
        key = (round(b.length, 6), round(b.width, 6), round(b.height, 6),
               round(b.weight, 6), mlot_key, rfs_key, rot_key)
        if key not in sku_to_id:
            sku_to_id[key] = len(sku_to_id)
        sku_id_per_box[i] = sku_to_id[key]
    return n_rots_arr, dims_all, sku_id_per_box


def precompute_constraint_arrays(boxes: List[Box], pallet: Pallet) -> tuple:
    """Return (weights, max_load_on_top, requires_full_support,
                pallet_max_weight, has_constraints).

    weights:               float64[n]  per-box weight (0 if none)
    max_load_on_top:       float64[n]  per-box load capacity on top
                                       (_NO_LIMIT if infinite or unset)
    requires_full_support: int64[n]    1 if box demands full support
                                       (overrides config.support_ratio with
                                       1.0 for this box); 0 otherwise
    pallet_max_weight:     float64     pallet weight cap (_NO_LIMIT if inf)
    has_constraints:       bool        True iff any constraint is finite —
                                       JIT decoders short-circuit if False
    """
    n = len(boxes)
    weights = np.zeros(n, dtype=np.float64)
    mlot = np.full(n, _NO_LIMIT, dtype=np.float64)
    rfs = np.zeros(n, dtype=np.int64)
    has_constraints = False
    for i, b in enumerate(boxes):
        weights[i] = float(b.weight) if b.weight else 0.0
        m = getattr(b, 'max_load_on_top', float('inf'))
        if m is not None and not math.isinf(m):
            mlot[i] = float(m)
            has_constraints = True
        if getattr(b, 'requires_full_support', False):
            rfs[i] = 1
            has_constraints = True
    pmw = getattr(pallet, 'max_weight', float('inf'))
    if pmw is None or math.isinf(pmw):
        pallet_max_weight = _NO_LIMIT
    else:
        pallet_max_weight = float(pmw)
        has_constraints = True
    return weights, mlot, rfs, pallet_max_weight, has_constraints


def precompute_cog_envelope(pallet: Pallet, config: PackerConfig) -> tuple:
    """Return (cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                cog_min_load_frac, cog_active).

    cog_active = True iff the envelope is finite (config.cog_envelope_fraction
    < 1.0 OR the pallet supplies explicit cog_x_range / cog_y_range bounds).
    When inactive, the JIT decoders skip the CoG check entirely.
    """
    L = float(pallet.length)
    W = float(pallet.width)
    frac = float(config.cog_envelope_fraction)
    min_load_frac = float(config.cog_check_min_load_fraction)
    if pallet.cog_x_range is not None:
        cx_min, cx_max = float(pallet.cog_x_range[0]), float(pallet.cog_x_range[1])
    else:
        cx_min = L / 2.0 - frac * L
        cx_max = L / 2.0 + frac * L
    if pallet.cog_y_range is not None:
        cy_min, cy_max = float(pallet.cog_y_range[0]), float(pallet.cog_y_range[1])
    else:
        cy_min = W / 2.0 - frac * W
        cy_max = W / 2.0 + frac * W
    full_envelope = (
        pallet.cog_x_range is None
        and pallet.cog_y_range is None
        and frac >= 1.0)
    cog_active = not full_envelope
    return cx_min, cx_max, cy_min, cy_max, min_load_frac, cog_active
