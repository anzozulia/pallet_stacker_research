"""
pallet_packer._brkga_core.realism — the secondary "realism" fitness term.

The historical fitness (1 - volume_on_pallet0 / capacity) ties for EVERY
feasible layout once all boxes fit, so evolution, local search, and seeds
stop doing anything and layout quality is decided entirely by decoder
tie-breaks. This module adds an epsilon-scaled secondary term that gives the
search a gradient toward layouts a warehouse would accept:

  HM  height moment    sum(m_i * z_center_i) / (sum(m_i) * H)   heavy-low + flat
  MH  max height       max(z2_i) / H                            anti-tower
  OI  orientation      1 - sum_sku(dominant height-class count) / N
      inconsistency    Variants are keyed by dz (the vertical extent), NOT
                       the full dims tuple: yaw variation within a layer is
                       legitimate interlocking, while a TIPPED box (different
                       dz) breaks the layer's top surface — that is the
                       "randomly rotated" defect worth optimizing away.

  R = 0.5*HM + 0.4*MH + 0.1*OI      in [0, 1)
  fitness' = base + eps * R

Term weights: the two PHYSICAL terms dominate. HM is crush/stability safety
and MH is silhouette; OI is largely cosmetic AND the align post-pass already
fixes feasible orientation deviants after the fact — measured on the
28-box warehouse_order_mix scenario, an equal-weight OI outvoted HM and made
the search prefer a heavy-on-top layout over the heavy-low alternative
purely because the latter mixed more orientations.

Epsilon guarantee (bounded loss, NOT lexicographic dominance): with
eps = 0.5 * min_box_volume / capacity * min(1.0, realism_weight), the realism
term can never cost more than half the smallest box's volume, so it can never
cause a box to be dropped — but it may prefer a marginally-lower-volume
arrangement. realism_weight is a dial in [0, 1]; values above 1 are CLAMPED
(an unclamped weight > 2 would make dropping the smallest box profitable and
silently void the guarantee — hardening plan A4). When the computed eps falls
under 1e-7 (degenerate dust-box instances) the term is disabled for that
solve rather than floored (flooring would break the bounded-loss budget).

No CoG-offset-from-center term on purpose: all decoders corner-anchor at the
origin, so pre-recenter that offset measures fill level, not arrangement
quality; postprocess.recenter_pass handles centring after the fact.

Scalar/batch parity: driver best-updates use the batch path while the polish
phases use the scalar path, and both feed the same 1e-9 strict-improvement
comparisons — so both paths compute the SAME formulas from the SAME integer
dims_all table (never Placement.dims floats). Exact bit-identity is not
attainable (different summation trees); the divergence is bounded well under
1e-12, six orders below the acceptance band. Guarded by a parity test.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ..models import Box, Pallet, PackerConfig

logger = logging.getLogger(__name__)

# Below this effective epsilon the term is meaningless next to the 1e-9
# acceptance bands — disable rather than floor (see module docstring).
_MIN_EPS = 1e-7

# Term weights (sum to 1). Physical terms dominate the cosmetic one — see
# the module docstring for the measured failure that motivated this split.
_W_HM = 0.5
_W_MH = 0.4
_W_OI = 0.1


@dataclass
class RealismContext:
    """Everything both fitness paths need, precomputed once per solve."""
    eps: float                    # effective epsilon (includes realism_weight)
    weights: np.ndarray           # (n,) float64 per-box weight
    sku_id_per_box: np.ndarray    # (n,) int64
    variant_id: np.ndarray        # (n, max_rots) int64 canonical dims-variant
    n_skus: int
    n_variants: int               # max height-class id + 1 (<= 3)
    dims_all: np.ndarray          # (n, max_rots, 3) int64 (shared, not copied)
    index_by_id: dict             # box.id -> row index (attribute-based:
                                  # v2-origin results contain Box copies)
    rot_row: list                 # per box: {Rotation: dims_all row} built from
                                  # the ORIGINAL rotation order (correct even for
                                  # rotation-locked box variants whose
                                  # allowed_rotations list is a subset)
    pallet_h: float


def build_realism_context(
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    *,
    dims_all: Optional[np.ndarray] = None,
    sku_id_per_box: Optional[np.ndarray] = None,
    weights_arr: Optional[np.ndarray] = None,
    n_rots_arr: Optional[np.ndarray] = None,
) -> Optional[RealismContext]:
    """Build the realism context, or None when the term is disabled
    (realism_weight <= 0, empty input, or a degenerate epsilon)."""
    if float(config.realism_weight) <= 0.0 or not boxes:
        return None
    ids = [str(b.id) for b in boxes]
    if len(set(ids)) != len(ids):
        # The scalar path maps placements back to rows BY ID while the batch
        # path is positional — duplicate ids would silently desync the two
        # and corrupt the search (C7/F9). The service's input gate enforces
        # unique ids; this protects un-gated library callers.
        logger.info("realism disabled for this solve: duplicate box ids "
                    "(scalar fitness is id-keyed; parity would break)")
        return None
    if dims_all is None or sku_id_per_box is None or n_rots_arr is None:
        from .precompute import precompute_box_dims_and_sku
        n_rots_arr, dims_all, sku_id_per_box = \
            precompute_box_dims_and_sku(boxes)
    if weights_arr is None:
        weights_arr = np.array(
            [float(b.weight) if b.weight else 0.0 for b in boxes],
            dtype=np.float64)

    cap = float(pallet.length * pallet.width * pallet.height)
    if cap <= 0.0:
        return None
    # Rotation 0 always exists; volume is rotation-invariant.
    vols0 = (dims_all[:, 0, 0].astype(np.float64)
             * dims_all[:, 0, 1] * dims_all[:, 0, 2])
    min_vol = float(vols0.min())
    # min(1.0, w): the bounded-loss guarantee holds only for eps <=
    # 0.5*min_vol/cap — see module docstring.
    eps = 0.5 * (min_vol / cap) * min(1.0, float(config.realism_weight))
    if eps < _MIN_EPS:
        logger.info("realism disabled for this solve: eps=%.3g < %.0e",
                    eps, _MIN_EPS)
        return None

    # Canonical height-class table: the OI variant of a rotation is its dz
    # (vertical extent), NOT the full dims tuple — yaw rotations that keep
    # dz share a variant (interlocking is legitimate), tipped rotations get
    # their own. Boxes of one SKU may list allowed rotations in different
    # orders, so canonicalise per SKU over the union of dz values, then map
    # each box's own rotation rows.
    n, max_rots = dims_all.shape[0], dims_all.shape[1]
    n_skus = int(sku_id_per_box.max()) + 1
    sku_variants: dict = {}
    for i in range(n):
        s = int(sku_id_per_box[i])
        heights = sku_variants.setdefault(s, set())
        for r in range(int(n_rots_arr[i])):
            heights.add(int(dims_all[i, r, 2]))
    sku_variant_index = {
        s: {dz: k for k, dz in enumerate(sorted(heights))}
        for s, heights in sku_variants.items()}
    variant_id = np.zeros((n, max_rots), dtype=np.int64)
    for i in range(n):
        vmap = sku_variant_index[int(sku_id_per_box[i])]
        for r in range(int(n_rots_arr[i])):
            variant_id[i, r] = vmap[int(dims_all[i, r, 2])]
    n_variants = max(len(v) for v in sku_variant_index.values())

    index_by_id = {}
    for i, b in enumerate(boxes):
        index_by_id.setdefault(str(b.id), i)
    rot_row = [{r: k for k, r in enumerate(b.allowed_rotations)}
               for b in boxes]

    return RealismContext(
        eps=eps, weights=weights_arr, sku_id_per_box=sku_id_per_box,
        variant_id=variant_id, n_skus=n_skus, n_variants=n_variants,
        dims_all=dims_all, index_by_id=index_by_id, rot_row=rot_row,
        pallet_h=float(pallet.height))


def _terms(z: np.ndarray, dz: np.ndarray, vols: np.ndarray, w: np.ndarray,
           sku: np.ndarray, var: np.ndarray, ctx: RealismContext) -> float:
    """R for ONE layout given aligned per-placed-box arrays. Shared by the
    scalar path; the batch path vectorises the identical formulas."""
    n_placed = z.shape[0]
    if n_placed == 0:
        return 0.0
    h = ctx.pallet_h
    m = w if float(w.sum()) > 0.0 else vols
    m_tot = float(m.sum())
    hm = float((m * (z + dz / 2.0)).sum()) / (m_tot * h) if m_tot > 0 else 0.0
    mh = float((z + dz).max()) / h
    counts = np.zeros((ctx.n_skus, ctx.n_variants), dtype=np.float64)
    np.add.at(counts, (sku, var), 1.0)
    oi = 1.0 - float(counts.max(axis=1).sum()) / n_placed
    return _W_HM * hm + _W_MH * mh + _W_OI * oi


def realism_scalar(result, ctx: RealismContext) -> float:
    """R for a PackResult (pallet 0 only, matching _fitness_pallet1's scope).

    Dims come from the integer dims_all table via box.id lookup — NOT from
    Placement.dims floats — to stay in lockstep with the batch path.
    """
    if not result.pallets:
        return 0.0
    z_l, dz_l, vol_l, w_l, sku_l, var_l = [], [], [], [], [], []
    for p in result.pallets[0].placements:
        i = ctx.index_by_id.get(str(p.box.id))
        if i is None:
            # Unknown box (shouldn't happen: ids are unique per the input
            # gate). Fall back to float dims so the value stays sane.
            d = p.dims
            z_l.append(float(p.z))
            dz_l.append(float(d[2]))
            vol_l.append(float(d[0] * d[1] * d[2]))
            w_l.append(float(p.box.weight))
            sku_l.append(0)
            var_l.append(0)
            continue
        r = ctx.rot_row[i].get(p.rotation, 0)
        d = ctx.dims_all[i, r]
        z_l.append(float(p.z))
        dz_l.append(float(d[2]))
        vol_l.append(float(d[0]) * float(d[1]) * float(d[2]))
        w_l.append(float(ctx.weights[i]))
        sku_l.append(int(ctx.sku_id_per_box[i]))
        var_l.append(int(ctx.variant_id[i, r]))
    return _terms(np.asarray(z_l), np.asarray(dz_l), np.asarray(vol_l),
                  np.asarray(w_l), np.asarray(sku_l, dtype=np.int64),
                  np.asarray(var_l, dtype=np.int64), ctx)


def realism_batch(placements_out_all: np.ndarray, orders: np.ndarray,
                  ctx: RealismContext) -> np.ndarray:
    """R for a whole decoded population, vectorised. Same formulas as
    realism_scalar; parity guarded by tests at |delta| < 1e-12."""
    pop, n = placements_out_all.shape[0], placements_out_all.shape[1]
    mask = ((placements_out_all[:, :, 5] == 1)
            & (placements_out_all[:, :, 0] == 0))
    rot_idx = placements_out_all[:, :, 1]
    sel = ctx.dims_all[orders, rot_idx].astype(np.float64)   # (pop, n, 3)
    vols = sel[:, :, 0] * sel[:, :, 1] * sel[:, :, 2]
    z = placements_out_all[:, :, 4].astype(np.float64)
    dz = sel[:, :, 2]
    fm = mask.astype(np.float64)
    h = ctx.pallet_h

    w = ctx.weights[orders] * fm
    w_tot = w.sum(axis=1)
    m = np.where((w_tot > 0.0)[:, None], w, vols * fm)
    m_tot = m.sum(axis=1)
    hm = np.zeros(pop, dtype=np.float64)
    nz = m_tot > 0.0
    hm[nz] = ((m * (z + dz / 2.0)).sum(axis=1))[nz] / (m_tot[nz] * h)

    mh = np.where(mask, z + dz, 0.0).max(axis=1) / h if n else np.zeros(pop)

    sku = ctx.sku_id_per_box[orders]
    var = ctx.variant_id[orders, rot_idx]
    counts = np.zeros((pop, ctx.n_skus, ctx.n_variants), dtype=np.float64)
    rows = np.broadcast_to(np.arange(pop)[:, None], (pop, n))
    np.add.at(counts, (rows[mask], sku[mask], var[mask]), 1.0)
    n_placed = fm.sum(axis=1)
    oi = np.zeros(pop, dtype=np.float64)
    pp = n_placed > 0.0
    oi[pp] = 1.0 - counts.max(axis=2).sum(axis=1)[pp] / n_placed[pp]

    return _W_HM * hm + _W_MH * mh + _W_OI * oi
