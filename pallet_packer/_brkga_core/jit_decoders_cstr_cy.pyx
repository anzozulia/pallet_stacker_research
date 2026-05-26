# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_decoders_cstr_cy — Cython port of the
constraint-aware decoders.

Phase 4b (this file): decode_njit_mode_cstr (cstr modes 0/1/2).
Phase 4c will add decode_blocks_njit_mode_cstr (cstr mode 4) here.
Phase 4d will add decode_layer_njit_cstr   (cstr mode 3) here.

All inner loops are nogil. Constraint helpers are cimported from
jit_constraints_cy via .pxd so the feasibility checks + bookkeeping
updates happen at C level with no Python boundary or GIL overhead.

Bit-identical to the Numba reference at fixed seed — see
scripts/ab_test_decoder_cstr_modes012.py.
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

from .v3fast_cy cimport _find_best_dftrc, _commit_ems, MAX_EMS_C
from .jit_primitives_cy cimport (
    _find_best_wall, _find_best_corner, _find_best_in_slab,
)
from .jit_constraints_cy cimport (
    _ck_load_on_top, _ck_cog_envelope,
    _ap_load_contribution, _ap_cog_contribution,
)

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
