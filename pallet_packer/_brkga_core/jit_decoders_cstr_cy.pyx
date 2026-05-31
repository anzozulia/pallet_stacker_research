# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_decoders_cstr_cy — Cython port of the
constraint-aware decoders.

Phase 4b (this file): decode_njit_mode_cstr (cstr modes 0/1/2).
Phase 4c (this file): decode_blocks_njit_mode_cstr (cstr mode 4) +
                      _max_block_under_constraints + _commit_block_placements.
Phase 4d (this file): decode_layer_njit_cstr (cstr mode 3 — Bischoff-Ratcliff
                      layers under constraints).

ALL constraint-aware decoders now Cython. Phase 4 complete.

All inner loops are nogil. Constraint helpers are cimported from
jit_constraints_cy via .pxd so the feasibility checks + bookkeeping
updates happen at C level with no Python boundary or GIL overhead.

Bit-identical to the Numba reference at fixed seed — see
scripts/ab_test_decoder_cstr_modes012.py.
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t
from cython.parallel cimport prange

from .v3fast_cy cimport _find_best_dftrc, _commit_ems, MAX_EMS_C
from .jit_primitives_cy cimport (
    _find_best_wall, _find_best_corner, _find_best_in_slab,
)
from .jit_constraints_cy cimport (
    _ck_load_on_top, _ck_cog_envelope,
    _ap_load_contribution, _ap_cog_contribution,
)
from .jit_decoders_geom_cy cimport _find_best_block_at_pos

ctypedef int64_t i64


def decode_njit_mode_cstr(
    bps_order, n_rots_per_box, dims_all,
    L, W, H, max_pallets, placements_out, mode,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
):
    """Cython implementation of decode_njit_mode_cstr (cstr modes 0/1/2).
    Matches the Numba reference signature + semantics exactly.
    """
    cdef i64 cL = L, cW = W, cH = H
    cdef int cmode = mode
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef i64[:, ::1] po_v = placements_out

    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs

    cdef i64 n = bo_v.shape[0]
    cdef double[::1] pal_w = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] ptl = np.zeros(n, dtype=np.float64)
    cdef double[::1] psx = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] psy = np.zeros(MAX_BINS, dtype=np.float64)

    return _decode_cstr_012_loop(
        bo_v, nr_v, da_v, cL, cW, cH, MAX_BINS, po_v, cmode,
        bin_emss, bin_ems_count, scratch,
        w_v, mlot_v, rfs_v, pmw, sr, rc,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
        pal_w, ptl, psx, psy, n,
    )


cdef i64 _decode_cstr_012_loop(
    const i64[::1] bps_order,
    const i64[::1] n_rots_per_box,
    const i64[:, :, ::1] dims_all,
    i64 L, i64 W, i64 H,
    i64 MAX_BINS,
    i64[:, ::1] placements_out,
    int mode,
    i64[:, :, :, ::1] bin_emss,
    i64[::1] bin_ems_count,
    i64[:, :, ::1] scratch,
    const double[::1] weights,
    const double[::1] mlot,
    const i64[::1] rfs,
    double pallet_max_weight,
    double support_ratio,
    int require_centroid,
    double cog_x_min, double cog_x_max,
    double cog_y_min, double cog_y_max,
    double cog_min_load_frac,
    int cog_active,
    double[::1] pallet_weights,
    double[::1] placement_top_loads,
    double[::1] pallet_sum_xw,
    double[::1] pallet_sum_yw,
    i64 n,
) noexcept nogil:
    """All-nogil inner driver — cstr modes 0/1/2."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, box_idx, n_rots
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, sc, yz, cand, BIG
    cdef i64 bin_best_rot, bin_best_score, bin_best_min
    cdef i64 bin_best_x, bin_best_y, bin_best_z
    cdef i64 best_rot_n, best_score_n, best_min_n, best_x_n, best_y_n, best_z_n
    cdef i64 new_count
    cdef double cand_weight
    cdef bint placed
    BIG = (W + H) * (W + H) + 1

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        cand_weight = weights[box_idx]
        placed = False

        for b in range(n_bins):
            # Pre-check: pallet weight cap.
            if pallet_weights[b] + cand_weight > pallet_max_weight + 1e-6:
                continue
            bin_best_rot = -1
            bin_best_score = -1
            bin_best_min = <i64>(1) << 62
            bin_best_x = 0
            bin_best_y = 0
            bin_best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if mode == 0:
                    _find_best_dftrc(
                        bin_emss[b], bin_ems_count[b],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    sc = ((L - x - dx) * (L - x - dx)
                          + (W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    if sc > bin_best_score:
                        bin_best_score = sc
                        bin_best_rot = r
                        bin_best_x = x; bin_best_y = y; bin_best_z = z
                elif mode == 1:
                    _find_best_wall(
                        bin_emss[b], bin_ems_count[b],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    yz = ((W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    cand = x * BIG - yz
                    if cand < bin_best_min:
                        bin_best_min = cand
                        bin_best_rot = r
                        bin_best_x = x; bin_best_y = y; bin_best_z = z
                else:
                    _find_best_corner(
                        bin_emss[b], bin_ems_count[b],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < bin_best_min:
                        bin_best_min = cand
                        bin_best_rot = r
                        bin_best_x = x; bin_best_y = y; bin_best_z = z

            if bin_best_rot < 0:
                continue
            dx = dims_all[box_idx, bin_best_rot, 0]
            dy = dims_all[box_idx, bin_best_rot, 1]
            dz = dims_all[box_idx, bin_best_rot, 2]

            # Load-on-top + support-ratio + centroid + per-box rfs.
            if not _ck_load_on_top(
                    placements_out, dims_all, bps_order, mlot,
                    placement_top_loads, n,
                    b, bin_best_x, bin_best_y, bin_best_z,
                    dx, dy, dz, cand_weight, support_ratio,
                    require_centroid, <int>rfs[box_idx]):
                continue

            # CoG envelope (only when cog_active).
            if cog_active != 0:
                if not _ck_cog_envelope(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac):
                    continue

            # Commit.
            new_count = _commit_ems(
                bin_emss[b], bin_ems_count[b],
                bin_best_x, bin_best_y, bin_best_z,
                bin_best_x + dx, bin_best_y + dy, bin_best_z + dz,
                scratch)
            for j in range(new_count):
                bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[b] = new_count
            placements_out[i, 0] = b
            placements_out[i, 1] = bin_best_rot
            placements_out[i, 2] = bin_best_x
            placements_out[i, 3] = bin_best_y
            placements_out[i, 4] = bin_best_z
            placements_out[i, 5] = 1
            pallet_weights[b] += cand_weight
            _ap_load_contribution(
                placements_out, dims_all, bps_order,
                placement_top_loads, n,
                b, bin_best_x, bin_best_y, bin_best_z,
                dx, dy, dz, cand_weight)
            _ap_cog_contribution(
                bin_best_x, bin_best_y, dx, dy,
                cand_weight, b, pallet_sum_xw, pallet_sum_yw)
            placed = True
            break

        if not placed:
            if n_bins >= MAX_BINS:
                placements_out[i, 5] = 0
                continue
            if cand_weight > pallet_max_weight + 1e-6:
                placements_out[i, 5] = 0
                continue
            bin_emss[n_bins, 0, 0, 0] = 0
            bin_emss[n_bins, 0, 0, 1] = 0
            bin_emss[n_bins, 0, 0, 2] = 0
            bin_emss[n_bins, 0, 1, 0] = L
            bin_emss[n_bins, 0, 1, 1] = W
            bin_emss[n_bins, 0, 1, 2] = H
            bin_ems_count[n_bins] = 1
            best_rot_n = -1
            best_score_n = -1
            best_min_n = <i64>(1) << 62
            best_x_n = 0
            best_y_n = 0
            best_z_n = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if mode == 0:
                    _find_best_dftrc(
                        bin_emss[n_bins], bin_ems_count[n_bins],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    sc = ((L - x - dx) * (L - x - dx)
                          + (W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    if sc > best_score_n:
                        best_score_n = sc
                        best_rot_n = r
                        best_x_n = x; best_y_n = y; best_z_n = z
                elif mode == 1:
                    _find_best_wall(
                        bin_emss[n_bins], bin_ems_count[n_bins],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    yz = ((W - y - dy) * (W - y - dy)
                          + (H - z - dz) * (H - z - dz))
                    cand = x * BIG - yz
                    if cand < best_min_n:
                        best_min_n = cand
                        best_rot_n = r
                        best_x_n = x; best_y_n = y; best_z_n = z
                else:
                    _find_best_corner(
                        bin_emss[n_bins], bin_ems_count[n_bins],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < best_min_n:
                        best_min_n = cand
                        best_rot_n = r
                        best_x_n = x; best_y_n = y; best_z_n = z
            if best_rot_n < 0:
                placements_out[i, 5] = 0
                continue
            r = best_rot_n
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            new_count = _commit_ems(
                bin_emss[n_bins], bin_ems_count[n_bins],
                best_x_n, best_y_n, best_z_n,
                best_x_n + dx, best_y_n + dy, best_z_n + dz,
                scratch)
            for j in range(new_count):
                bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[n_bins] = new_count
            placements_out[i, 0] = n_bins
            placements_out[i, 1] = best_rot_n
            placements_out[i, 2] = best_x_n
            placements_out[i, 3] = best_y_n
            placements_out[i, 4] = best_z_n
            placements_out[i, 5] = 1
            pallet_weights[n_bins] += cand_weight
            # First box at z=0; no top-load update needed.
            _ap_cog_contribution(
                best_x_n, best_y_n, dx, dy,
                cand_weight, n_bins, pallet_sum_xw, pallet_sum_yw)
            n_bins += 1

    return n_bins


# ===========================================================================
# Phase 4c: decode_blocks_njit_mode_cstr (cstr mode 4) + helpers.
#
# Mode-4 constraint-aware decoder. Like decode_blocks_njit_mode but also
# enforces:
#   - Pallet.max_weight  (per-pallet running cap)
#   - Box.max_load_on_top (internal stack height AND external supporters)
#   - support_ratio + per-box requires_full_support
#   - require_centroid + CoG envelope
# When a candidate block fails constraints, falls back to (1,1,1) at the
# same DFTRC position; if that fails, tries the next bin.
# ===========================================================================


cdef inline i64 _max_block_under_constraints(
    i64 k, i64 l, i64 m,
    double box_weight, double box_mlot, double pallet_weight_remaining,
    i64* out_k, i64* out_l, i64* out_m,
) noexcept nogil:
    """Reduce (k,l,m) to respect internal stack-height + pallet-weight caps.
    Returns 1 if a valid block remains, 0 otherwise.
    """
    cdef i64 max_m_stack, max_count
    # Internal stack cap.
    if box_weight > 0:
        max_m_stack = <i64>(box_mlot / box_weight) + 1
        if max_m_stack < 1:
            max_m_stack = 1
        if m > max_m_stack:
            m = max_m_stack
    if m < 1:
        m = 1
    # Pallet cap.
    if box_weight > 0 and pallet_weight_remaining < box_weight * k * l * m:
        max_count = <i64>(pallet_weight_remaining / box_weight)
        if max_count < 1:
            out_k[0] = 0; out_l[0] = 0; out_m[0] = 0
            return 0
        while k * l * m > max_count and m > 1:
            m -= 1
        while k * l * m > max_count and l > 1:
            l -= 1
        while k * l * m > max_count and k > 1:
            k -= 1
        if k * l * m > max_count:
            out_k[0] = 0; out_l[0] = 0; out_m[0] = 0
            return 0
    out_k[0] = k; out_l[0] = l; out_m[0] = m
    return 1


cdef i64 _commit_block_placements(
    i64[:, ::1] placements_out,
    i64[::1] placed,
    const i64[::1] bps_order,
    const i64[::1] sku_id_per_box,
    double[::1] placement_top_loads,
    i64 start_i, i64 n,
    i64 best_bin, i64 best_rot,
    i64 best_x, i64 best_y, i64 best_z,
    i64 dx, i64 dy, i64 dz,
    i64 k, i64 l, i64 m,
    i64 my_sku,
    double box_weight,
) noexcept nogil:
    """Assign block positions (X-fastest, then Y, then Z) and seed each
    placement's internal top-load (m-1-mm) * box_weight. Returns count placed.
    """
    cdef i64 placed_count = 0
    cdef i64 kk = 0, ll = 0, mm = 0
    cdef i64 j, px, py, pz
    cdef i64 cap = k * l * m
    for j in range(start_i, n):
        if placed_count >= cap:
            break
        if placed[j] == 1:
            continue
        if sku_id_per_box[bps_order[j]] != my_sku:
            continue
        px = best_x + kk * dx
        py = best_y + ll * dy
        pz = best_z + mm * dz
        placements_out[j, 0] = best_bin
        placements_out[j, 1] = best_rot
        placements_out[j, 2] = px
        placements_out[j, 3] = py
        placements_out[j, 4] = pz
        placements_out[j, 5] = 1
        placed[j] = 1
        placement_top_loads[j] = <double>(m - 1 - mm) * box_weight
        placed_count += 1
        kk += 1
        if kk >= k:
            kk = 0
            ll += 1
            if ll >= l:
                ll = 0
                mm += 1
    return placed_count


def decode_blocks_njit_mode_cstr(
    bps_order, n_rots_per_box, dims_all, sku_id_per_box,
    L, W, H, max_pallets, placements_out, n_skus,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
):
    """Cython implementation of decode_blocks_njit_mode_cstr (cstr mode 4)."""
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef i64[:, ::1] po_v = placements_out

    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs

    cdef i64 n = bo_v.shape[0]
    cdef i64[::1] placed = np.zeros(n, dtype=np.int64)
    cdef i64[::1] sku_remaining = np.zeros(cn_skus, dtype=np.int64)
    cdef double[::1] pal_w = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] ptl = np.zeros(n, dtype=np.float64)
    cdef double[::1] psx = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] psy = np.zeros(MAX_BINS, dtype=np.float64)

    return _decode_cstr_blocks_loop(
        bo_v, nr_v, da_v, sku_v, cL, cW, cH, MAX_BINS, po_v,
        bin_emss, bin_ems_count, scratch,
        w_v, mlot_v, rfs_v, pmw, sr, rc,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
        placed, sku_remaining, pal_w, ptl, psx, psy, n,
    )


cdef i64 _decode_cstr_blocks_loop(
    const i64[::1] bps_order,
    const i64[::1] n_rots_per_box,
    const i64[:, :, ::1] dims_all,
    const i64[::1] sku_id_per_box,
    i64 L, i64 W, i64 H,
    i64 MAX_BINS,
    i64[:, ::1] placements_out,
    i64[:, :, :, ::1] bin_emss,
    i64[::1] bin_ems_count,
    i64[:, :, ::1] scratch,
    const double[::1] weights,
    const double[::1] mlot,
    const i64[::1] rfs,
    double pallet_max_weight,
    double support_ratio,
    int require_centroid,
    double cog_x_min, double cog_x_max,
    double cog_y_min, double cog_y_max,
    double cog_min_load_frac,
    int cog_active,
    i64[::1] placed,
    i64[::1] sku_remaining,
    double[::1] pallet_weights,
    double[::1] placement_top_loads,
    double[::1] pallet_sum_xw,
    double[::1] pallet_sum_yw,
    i64 n,
) noexcept nogil:
    """All-nogil cstr mode 4 inner driver."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, jj, box_idx, my_sku, n_rots, max_count
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, sc
    cdef i64 best_rot, best_x, best_y, best_z, best_score
    cdef i64 best_rot_n, best_score_n, best_x_n, best_y_n, best_z_n
    cdef i64 k, l, m, new_count, placed_so_far
    cdef i64 kk, ll, mm, kk2, ll2, mm2, applied, applied2, blk_count
    cdef i64 px, py, pz
    cdef double box_weight, box_mlot, cap_remain, bottom_w, block_total_w
    cdef bint block_committed, bottom_ok
    cdef i64 bx_kk, bx_ll, bx_px, bx_py

    # Count remaining boxes per SKU.
    for i in range(n):
        sku_remaining[sku_id_per_box[bps_order[i]]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        n_rots = n_rots_per_box[box_idx]
        max_count = sku_remaining[my_sku]
        box_weight = weights[box_idx]
        box_mlot = mlot[box_idx]

        # Single box alone exceeds pallet cap → mark unpacked and move on.
        if box_weight > pallet_max_weight + 1e-6:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue

        block_committed = False
        for b in range(n_bins):
            # Pre-check: at least a single box must fit weight-wise.
            if pallet_weights[b] + box_weight > pallet_max_weight + 1e-6:
                continue
            # Phase 1: DFTRC place single box.
            best_rot = -1
            best_score = -1
            best_x = 0; best_y = 0; best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                _find_best_dftrc(
                    bin_emss[b], bin_ems_count[b],
                    dx, dy, dz, L, W, H,
                    &idx, &x, &y, &z)
                if idx < 0:
                    continue
                sc = ((L - x - dx) * (L - x - dx)
                      + (W - y - dy) * (W - y - dy)
                      + (H - z - dz) * (H - z - dz))
                if sc > best_score:
                    best_score = sc
                    best_rot = r
                    best_x = x; best_y = y; best_z = z
            if best_rot < 0:
                continue
            dx = dims_all[box_idx, best_rot, 0]
            dy = dims_all[box_idx, best_rot, 1]
            dz = dims_all[box_idx, best_rot, 2]
            # Phase 2: geometric max block.
            _find_best_block_at_pos(
                bin_emss[b], bin_ems_count[b],
                best_x, best_y, best_z, dx, dy, dz, max_count,
                &k, &l, &m)
            # Phase 2b: shrink under pallet-cap + stack-limit.
            cap_remain = pallet_max_weight - pallet_weights[b]
            if _max_block_under_constraints(
                    k, l, m, box_weight, box_mlot, cap_remain,
                    &k, &l, &m) == 0:
                continue
            # Phase 2c: PER-bottom-box support + load check. The validator
            # checks every box individually, so a block-aggregate check is
            # wrong: a 3x1 bottom layer can average >= support_ratio while a
            # corner box sits at 0.4 -> the validator floats it. Check each of
            # the k*l bottom boxes at its own footprint; if any fails, fall
            # back to a single box at the DFTRC position. Each box rests on the
            # already-placed layer below (the block's own bottom boxes share
            # z=best_z and don't support each other), so per-box is exact.
            bottom_ok = True
            for bx_ll in range(l):
                for bx_kk in range(k):
                    bx_px = best_x + bx_kk * dx
                    bx_py = best_y + bx_ll * dy
                    if not _ck_load_on_top(
                            placements_out, dims_all, bps_order, mlot,
                            placement_top_loads, n,
                            b, bx_px, bx_py, best_z,
                            dx, dy, dz, box_weight, support_ratio,
                            require_centroid, <int>rfs[box_idx]):
                        bottom_ok = False
                        break
                if not bottom_ok:
                    break
            if not bottom_ok:
                # Shrink block to (1, 1, m) and retry; if still fails, skip.
                k = 1; l = 1
                if not _ck_load_on_top(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, best_x, best_y, best_z,
                        dx, dy, dz, box_weight, support_ratio,
                        require_centroid, <int>rfs[box_idx]):
                    continue
            # Phase 2d: CoG envelope (block as point mass at bottom centroid).
            if cog_active != 0:
                block_total_w = <double>(k * l * m) * box_weight
                if not _ck_cog_envelope(
                        best_x, best_y, k * dx, l * dy,
                        block_total_w, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac):
                    continue
            # Phase 3: commit block.
            placed_so_far = _commit_block_placements(
                placements_out, placed, bps_order, sku_id_per_box,
                placement_top_loads, i, n,
                b, best_rot, best_x, best_y, best_z,
                dx, dy, dz, k, l, m, my_sku, box_weight)
            sku_remaining[my_sku] -= placed_so_far
            pallet_weights[b] += <double>placed_so_far * box_weight
            # Apply load contribution from bottom layer to external supporters.
            if best_z > 0:
                applied = 0
                for jj in range(i, n):
                    if applied >= k * l:
                        break
                    if placed[jj] != 1:
                        continue
                    if placements_out[jj, 0] != b:
                        continue
                    if placements_out[jj, 4] != best_z:
                        continue
                    if sku_id_per_box[bps_order[jj]] != my_sku:
                        continue
                    _ap_load_contribution(
                        placements_out, dims_all, bps_order,
                        placement_top_loads, n,
                        b, placements_out[jj, 2], placements_out[jj, 3],
                        best_z, dx, dy, dz, box_weight)
                    applied += 1
            # CoG: per-box point-mass contribution for every block box.
            if box_weight > 0:
                kk2 = 0; ll2 = 0; mm2 = 0
                blk_count = k * l * m
                applied2 = 0
                for jj in range(i, n):
                    if applied2 >= blk_count:
                        break
                    if placed[jj] != 1:
                        continue
                    if placements_out[jj, 0] != b:
                        continue
                    if sku_id_per_box[bps_order[jj]] != my_sku:
                        continue
                    px = best_x + kk2 * dx
                    py = best_y + ll2 * dy
                    pz = best_z + mm2 * dz
                    if (placements_out[jj, 2] != px
                            or placements_out[jj, 3] != py
                            or placements_out[jj, 4] != pz):
                        continue
                    _ap_cog_contribution(
                        px, py, dx, dy, box_weight, b,
                        pallet_sum_xw, pallet_sum_yw)
                    applied2 += 1
                    kk2 += 1
                    if kk2 >= k:
                        kk2 = 0
                        ll2 += 1
                        if ll2 >= l:
                            ll2 = 0
                            mm2 += 1
            # Single EMS commit for the block region.
            new_count = _commit_ems(
                bin_emss[b], bin_ems_count[b],
                best_x, best_y, best_z,
                best_x + k * dx, best_y + l * dy, best_z + m * dz,
                scratch)
            for j in range(new_count):
                bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[b] = new_count
            block_committed = True
            break

        if block_committed:
            continue

        # No existing bin took it → open a new bin.
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        best_rot_n = -1
        best_score_n = -1
        best_x_n = 0; best_y_n = 0; best_z_n = 0
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            _find_best_dftrc(
                bin_emss[n_bins], bin_ems_count[n_bins],
                dx, dy, dz, L, W, H,
                &idx, &x, &y, &z)
            if idx < 0:
                continue
            sc = ((L - x - dx) * (L - x - dx)
                  + (W - y - dy) * (W - y - dy)
                  + (H - z - dz) * (H - z - dz))
            if sc > best_score_n:
                best_score_n = sc
                best_rot_n = r
                best_x_n = x; best_y_n = y; best_z_n = z
        if best_rot_n < 0:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        dx = dims_all[box_idx, best_rot_n, 0]
        dy = dims_all[box_idx, best_rot_n, 1]
        dz = dims_all[box_idx, best_rot_n, 2]
        _find_best_block_at_pos(
            bin_emss[n_bins], bin_ems_count[n_bins],
            best_x_n, best_y_n, best_z_n, dx, dy, dz, max_count,
            &k, &l, &m)
        if _max_block_under_constraints(
                k, l, m, box_weight, box_mlot, pallet_max_weight,
                &k, &l, &m) == 0:
            placements_out[i, 5] = 0
            placed[i] = 1
            sku_remaining[my_sku] -= 1
            continue
        # Floor placement (z=0): no support/load check needed for bottom layer.
        placed_so_far = _commit_block_placements(
            placements_out, placed, bps_order, sku_id_per_box,
            placement_top_loads, i, n,
            n_bins, best_rot_n, best_x_n, best_y_n, best_z_n,
            dx, dy, dz, k, l, m, my_sku, box_weight)
        sku_remaining[my_sku] -= placed_so_far
        pallet_weights[n_bins] += <double>placed_so_far * box_weight
        # CoG per box.
        if box_weight > 0:
            kk2 = 0; ll2 = 0; mm2 = 0
            blk_count = k * l * m
            applied2 = 0
            for jj in range(i, n):
                if applied2 >= blk_count:
                    break
                if placed[jj] != 1:
                    continue
                if placements_out[jj, 0] != n_bins:
                    continue
                if sku_id_per_box[bps_order[jj]] != my_sku:
                    continue
                px = best_x_n + kk2 * dx
                py = best_y_n + ll2 * dy
                pz = best_z_n + mm2 * dz
                if (placements_out[jj, 2] != px
                        or placements_out[jj, 3] != py
                        or placements_out[jj, 4] != pz):
                    continue
                _ap_cog_contribution(
                    px, py, dx, dy, box_weight, n_bins,
                    pallet_sum_xw, pallet_sum_yw)
                applied2 += 1
                kk2 += 1
                if kk2 >= k:
                    kk2 = 0
                    ll2 += 1
                    if ll2 >= l:
                        ll2 = 0
                        mm2 += 1
        new_count = _commit_ems(
            bin_emss[n_bins], bin_ems_count[n_bins],
            best_x_n, best_y_n, best_z_n,
            best_x_n + k * dx, best_y_n + l * dy, best_z_n + m * dz,
            scratch)
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        n_bins += 1

    return n_bins


# ===========================================================================
# Phase 4d: decode_layer_njit_cstr (cstr mode 3 — Bischoff-Ratcliff layers).
#
# Each bin holds a current X-slab [slab_min, slab_max]. The slab depth is
# set by the first ("seed") box. Subsequent BPS-ordered boxes try to fit in
# the current slab via _find_best_in_slab; on miss, open a new slab at
# slab_max_x; on miss-everywhere, open a new bin. Per-placement enforces
# pallet weight cap + load_on_top + cog (when active).
# ===========================================================================


def decode_layer_njit_cstr(
    bps_order, n_rots_per_box, dims_all,
    L, W, H, max_pallets, placements_out,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
):
    """Cython implementation of decode_layer_njit_cstr (cstr mode 3)."""
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[::1] bin_slab_min_x = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[::1] bin_slab_max_x = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef i64[:, ::1] po_v = placements_out

    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs

    cdef i64 n = bo_v.shape[0]
    cdef double[::1] pal_w = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] ptl = np.zeros(n, dtype=np.float64)
    cdef double[::1] psx = np.zeros(MAX_BINS, dtype=np.float64)
    cdef double[::1] psy = np.zeros(MAX_BINS, dtype=np.float64)

    return _decode_cstr_layer_loop(
        bo_v, nr_v, da_v, cL, cW, cH, MAX_BINS, po_v,
        bin_emss, bin_ems_count, bin_slab_min_x, bin_slab_max_x, scratch,
        w_v, mlot_v, rfs_v, pmw, sr, rc,
        cxmn, cxmx, cymn, cymx, cmlf, cact,
        pal_w, ptl, psx, psy, n,
    )


cdef i64 _decode_cstr_layer_loop(
    const i64[::1] bps_order,
    const i64[::1] n_rots_per_box,
    const i64[:, :, ::1] dims_all,
    i64 L, i64 W, i64 H,
    i64 MAX_BINS,
    i64[:, ::1] placements_out,
    i64[:, :, :, ::1] bin_emss,
    i64[::1] bin_ems_count,
    i64[::1] bin_slab_min_x,
    i64[::1] bin_slab_max_x,
    i64[:, :, ::1] scratch,
    const double[::1] weights,
    const double[::1] mlot,
    const i64[::1] rfs,
    double pallet_max_weight,
    double support_ratio,
    int require_centroid,
    double cog_x_min, double cog_x_max,
    double cog_y_min, double cog_y_max,
    double cog_min_load_frac,
    int cog_active,
    double[::1] pallet_weights,
    double[::1] placement_top_loads,
    double[::1] pallet_sum_xw,
    double[::1] pallet_sum_yw,
    i64 n,
) noexcept nogil:
    """All-nogil cstr mode 3 inner driver."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, box_idx, n_rots
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, yz
    cdef i64 bin_best_rot, bin_best_yz
    cdef i64 bin_best_x, bin_best_y, bin_best_z
    cdef i64 new_slab_start, new_best_rot, new_best_yz, new_best_y, new_best_z, new_best_dx
    cdef i64 seed_rot, seed_yz, seed_y, seed_z
    cdef i64 new_count
    cdef double cand_weight
    cdef bint placed
    cdef bint cog_ok

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        cand_weight = weights[box_idx]
        placed = False

        for b in range(n_bins):
            if pallet_weights[b] + cand_weight > pallet_max_weight + 1e-6:
                continue

            # Phase 1: try current slab.
            bin_best_rot = -1
            bin_best_yz = -1
            bin_best_x = 0
            bin_best_y = 0
            bin_best_z = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                _find_best_in_slab(
                    bin_emss[b], bin_ems_count[b],
                    dx, dy, dz, L, W, H,
                    bin_slab_min_x[b], bin_slab_max_x[b],
                    &idx, &x, &y, &z)
                if idx < 0:
                    continue
                yz = ((W - y - dy) * (W - y - dy)
                      + (H - z - dz) * (H - z - dz))
                if yz > bin_best_yz:
                    bin_best_yz = yz
                    bin_best_rot = r
                    bin_best_x = x; bin_best_y = y; bin_best_z = z
            if bin_best_rot >= 0:
                dx = dims_all[box_idx, bin_best_rot, 0]
                dy = dims_all[box_idx, bin_best_rot, 1]
                dz = dims_all[box_idx, bin_best_rot, 2]
                cog_ok = True
                if cog_active != 0:
                    cog_ok = _ck_cog_envelope(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac)
                if cog_ok and _ck_load_on_top(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, bin_best_x, bin_best_y, bin_best_z,
                        dx, dy, dz, cand_weight, support_ratio,
                        require_centroid, <int>rfs[box_idx]):
                    new_count = _commit_ems(
                        bin_emss[b], bin_ems_count[b],
                        bin_best_x, bin_best_y, bin_best_z,
                        bin_best_x + dx, bin_best_y + dy, bin_best_z + dz,
                        scratch)
                    for j in range(new_count):
                        bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                        bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                        bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                        bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                        bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                        bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                    bin_ems_count[b] = new_count
                    placements_out[i, 0] = b
                    placements_out[i, 1] = bin_best_rot
                    placements_out[i, 2] = bin_best_x
                    placements_out[i, 3] = bin_best_y
                    placements_out[i, 4] = bin_best_z
                    placements_out[i, 5] = 1
                    pallet_weights[b] += cand_weight
                    _ap_load_contribution(
                        placements_out, dims_all, bps_order,
                        placement_top_loads, n,
                        b, bin_best_x, bin_best_y, bin_best_z,
                        dx, dy, dz, cand_weight)
                    _ap_cog_contribution(
                        bin_best_x, bin_best_y, dx, dy,
                        cand_weight, b, pallet_sum_xw, pallet_sum_yw)
                    placed = True
                    break

            # Phase 2: open a new slab at slab_max_x in this bin.
            new_slab_start = bin_slab_max_x[b]
            new_best_rot = -1
            new_best_yz = -1
            new_best_y = 0
            new_best_z = 0
            new_best_dx = 0
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if new_slab_start + dx > L:
                    continue
                _find_best_in_slab(
                    bin_emss[b], bin_ems_count[b],
                    dx, dy, dz, L, W, H,
                    new_slab_start, new_slab_start + dx,
                    &idx, &x, &y, &z)
                if idx < 0:
                    continue
                yz = ((W - y - dy) * (W - y - dy)
                      + (H - z - dz) * (H - z - dz))
                if yz > new_best_yz:
                    new_best_yz = yz
                    new_best_rot = r
                    new_best_y = y; new_best_z = z
                    new_best_dx = dx
            if new_best_rot >= 0:
                r = new_best_rot
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                cog_ok = True
                if cog_active != 0:
                    cog_ok = _ck_cog_envelope(
                        new_slab_start, new_best_y, dx, dy,
                        cand_weight, b,
                        pallet_weights, pallet_sum_xw, pallet_sum_yw,
                        pallet_max_weight,
                        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                        cog_min_load_frac)
                if cog_ok and _ck_load_on_top(
                        placements_out, dims_all, bps_order, mlot,
                        placement_top_loads, n,
                        b, new_slab_start, new_best_y, new_best_z,
                        dx, dy, dz, cand_weight, support_ratio,
                        require_centroid, <int>rfs[box_idx]):
                    new_count = _commit_ems(
                        bin_emss[b], bin_ems_count[b],
                        new_slab_start, new_best_y, new_best_z,
                        new_slab_start + dx, new_best_y + dy, new_best_z + dz,
                        scratch)
                    for j in range(new_count):
                        bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                        bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                        bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                        bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                        bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                        bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                    bin_ems_count[b] = new_count
                    placements_out[i, 0] = b
                    placements_out[i, 1] = r
                    placements_out[i, 2] = new_slab_start
                    placements_out[i, 3] = new_best_y
                    placements_out[i, 4] = new_best_z
                    placements_out[i, 5] = 1
                    pallet_weights[b] += cand_weight
                    bin_slab_min_x[b] = new_slab_start
                    bin_slab_max_x[b] = new_slab_start + dx
                    _ap_load_contribution(
                        placements_out, dims_all, bps_order,
                        placement_top_loads, n,
                        b, new_slab_start, new_best_y, new_best_z,
                        dx, dy, dz, cand_weight)
                    _ap_cog_contribution(
                        new_slab_start, new_best_y, dx, dy,
                        cand_weight, b, pallet_sum_xw, pallet_sum_yw)
                    placed = True
                    break

        if placed:
            continue

        # Phase 3: open a new bin (first box is the seed).
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            continue
        if cand_weight > pallet_max_weight + 1e-6:
            placements_out[i, 5] = 0
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        seed_rot = -1
        seed_yz = -1
        seed_y = 0
        seed_z = 0
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            if dx > L or dy > W or dz > H:
                continue
            yz = (W - dy) * (W - dy) + (H - dz) * (H - dz)
            if yz > seed_yz:
                seed_yz = yz
                seed_rot = r
                seed_y = 0
                seed_z = 0
        if seed_rot < 0:
            placements_out[i, 5] = 0
            continue
        dx = dims_all[box_idx, seed_rot, 0]
        dy = dims_all[box_idx, seed_rot, 1]
        dz = dims_all[box_idx, seed_rot, 2]
        new_count = _commit_ems(
            bin_emss[n_bins], bin_ems_count[n_bins],
            0, seed_y, seed_z, dx, seed_y + dy, seed_z + dz, scratch)
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        placements_out[i, 0] = n_bins
        placements_out[i, 1] = seed_rot
        placements_out[i, 2] = 0
        placements_out[i, 3] = seed_y
        placements_out[i, 4] = seed_z
        placements_out[i, 5] = 1
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = dx
        pallet_weights[n_bins] += cand_weight
        _ap_cog_contribution(
            0, seed_y, dx, dy, cand_weight, n_bins,
            pallet_sum_xw, pallet_sum_yw)
        n_bins += 1

    return n_bins


# ===========================================================================
# Phase 5b: batch entry points for the constraint-aware decoders.
# Per-chromosome scratch arrays + cython.parallel.prange. Same determinism
# contract as Phase 5a — schedule='static' makes chromosome i always land
# on the same thread for a given pop_size.
# ===========================================================================


def decode_batch_njit_mode_cstr(
    chromosomes, n_rots_per_box, dims_all,
    L, W, H, max_pallets,
    mode,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
    placements_out_all, n_bins_out,
):
    """Batch-decode pop_size chromosomes for cstr modes 0/1/2 in parallel."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef int cmode = mode
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef double[:, ::1] palw_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] ptl_all = np.zeros((pop_size, n_boxes), dtype=np.float64)
    cdef double[:, ::1] psx_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] psy_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _decode_cstr_012_loop(
                orders[i], nr_v, da_v, cL, cW, cH, MAX_BINS,
                po_all[i], cmode,
                be_all[i], bec_all[i], sc_all[i],
                w_v, mlot_v, rfs_v, pmw, sr, rc,
                cxmn, cxmx, cymn, cymx, cmlf, cact,
                palw_all[i], ptl_all[i], psx_all[i], psy_all[i], n_boxes,
            )
    return n_bins_out


def decode_batch_blocks_njit_mode_cstr(
    chromosomes, n_rots_per_box, dims_all, sku_id_per_box,
    L, W, H, max_pallets,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
    placements_out_all, n_bins_out, n_skus,
):
    """Batch-decode pop_size chromosomes for cstr mode 4 in parallel."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] placed_all = np.zeros((pop_size, n_boxes), dtype=np.int64)
    cdef i64[:, ::1] skur_all = np.zeros((pop_size, cn_skus), dtype=np.int64)
    cdef double[:, ::1] palw_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] ptl_all = np.zeros((pop_size, n_boxes), dtype=np.float64)
    cdef double[:, ::1] psx_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] psy_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _decode_cstr_blocks_loop(
                orders[i], nr_v, da_v, sku_v, cL, cW, cH, MAX_BINS,
                po_all[i],
                be_all[i], bec_all[i], sc_all[i],
                w_v, mlot_v, rfs_v, pmw, sr, rc,
                cxmn, cxmx, cymn, cymx, cmlf, cact,
                placed_all[i], skur_all[i],
                palw_all[i], ptl_all[i], psx_all[i], psy_all[i], n_boxes,
            )
    return n_bins_out


def decode_batch_layer_njit_cstr(
    chromosomes, n_rots_per_box, dims_all,
    L, W, H, max_pallets,
    weights, mlot, rfs, pallet_max_weight, support_ratio,
    require_centroid,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac, cog_active,
    placements_out_all, n_bins_out,
):
    """Batch-decode pop_size chromosomes for cstr mode 3 (layer) in parallel."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef double pmw = pallet_max_weight
    cdef double sr = support_ratio
    cdef int rc = require_centroid
    cdef double cxmn = cog_x_min, cxmx = cog_x_max
    cdef double cymn = cog_y_min, cymx = cog_y_max
    cdef double cmlf = cog_min_load_frac
    cdef int cact = cog_active

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const double[::1] w_v = weights
    cdef const double[::1] mlot_v = mlot
    cdef const i64[::1] rfs_v = rfs
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, ::1] sm_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, ::1] sx_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef double[:, ::1] palw_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] ptl_all = np.zeros((pop_size, n_boxes), dtype=np.float64)
    cdef double[:, ::1] psx_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)
    cdef double[:, ::1] psy_all = np.zeros((pop_size, MAX_BINS), dtype=np.float64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _decode_cstr_layer_loop(
                orders[i], nr_v, da_v, cL, cW, cH, MAX_BINS,
                po_all[i],
                be_all[i], bec_all[i], sm_all[i], sx_all[i], sc_all[i],
                w_v, mlot_v, rfs_v, pmw, sr, rc,
                cxmn, cxmx, cymn, cymx, cmlf, cact,
                palw_all[i], ptl_all[i], psx_all[i], psy_all[i], n_boxes,
            )
    return n_bins_out
