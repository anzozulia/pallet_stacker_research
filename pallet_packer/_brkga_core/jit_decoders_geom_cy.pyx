# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_decoders_geom_cy — Cython port of the
geometric decoders.

Phase 3a (this file): decode_njit_mode (modes 0/1/2 — DFTRC, wall, corner).
Phase 3b (this file): decode_layer_njit (mode 3 — layer-build).
Phase 3c-d will land blocks / precomputed_blocks here.

All cross-module calls use the .pxd `cimport` interface so the entire
inner loop runs nogil — no Python boundary, no tuple boxing.
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

from .v3fast_cy cimport _find_best_dftrc, _commit_ems, MAX_EMS_C
from .jit_primitives_cy cimport (
    _find_best_wall, _find_best_corner, _find_best_in_slab,
)

ctypedef int64_t i64


def decode_njit_mode(
    bps_order, n_rots_per_box, dims_all,
    L, W, H, max_pallets, placements_out, mode,
):
    """Cython implementation of decode_njit_mode. Matches the Numba
    reference signature + semantics exactly.

    The Python wrapper builds typed memoryviews + scratch arrays; the
    inner loop is all nogil cdef.
    """
    cdef i64 cL = L, cW = W, cH = H
    cdef int cmode = mode
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef i64[:, ::1] po_v = placements_out

    cdef i64 n = bo_v.shape[0]
    return _decode_loop(
        bo_v, nr_v, da_v, cL, cW, cH, MAX_BINS, po_v, cmode,
        bin_emss, bin_ems_count, scratch, n,
    )


cdef i64 _decode_loop(
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
    i64 n,
) noexcept nogil:
    """All-nogil inner driver. The wrapper above only sets up arrays."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, box_idx, n_rots
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, sc, yz, cand, BIG
    cdef i64 best_bin, best_rot, best_x, best_y, best_z
    cdef i64 best_score, best_minimise
    cdef i64 best_rot_n, best_score_n, best_min_n, best_x_n, best_y_n, best_z_n
    cdef i64 new_count, j
    cdef bint placed
    BIG = (W + H) * (W + H) + 1

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        placed = False
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_score = -1
        best_minimise = <i64>(1) << 62

        for b in range(n_bins):
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
                    if sc > best_score:
                        best_score = sc
                        best_bin = b
                        best_rot = r
                        best_x = x; best_y = y; best_z = z
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
                    if cand < best_minimise:
                        best_minimise = cand
                        best_bin = b
                        best_rot = r
                        best_x = x; best_y = y; best_z = z
                else:
                    _find_best_corner(
                        bin_emss[b], bin_ems_count[b],
                        dx, dy, dz, L, W, H,
                        &idx, &x, &y, &z)
                    if idx < 0:
                        continue
                    cand = x + y + z
                    if cand < best_minimise:
                        best_minimise = cand
                        best_bin = b
                        best_rot = r
                        best_x = x; best_y = y; best_z = z
            if best_bin >= 0:
                break

        if best_bin >= 0:
            r = best_rot
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            new_count = _commit_ems(
                bin_emss[best_bin], bin_ems_count[best_bin],
                best_x, best_y, best_z,
                best_x + dx, best_y + dy, best_z + dz,
                scratch)
            for j in range(new_count):
                bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[best_bin] = new_count
            placements_out[i, 0] = best_bin
            placements_out[i, 1] = best_rot
            placements_out[i, 2] = best_x
            placements_out[i, 3] = best_y
            placements_out[i, 4] = best_z
            placements_out[i, 5] = 1
            placed = True

        if not placed:
            if n_bins >= MAX_BINS:
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
            n_bins += 1

    return n_bins


# ---------------------------------------------------------------------------
# Phase 3b: decode_layer_njit (mode 3 — Bischoff-Ratcliff layer-building).
# Each bin maintains a current X-slab [slab_min_x, slab_max_x] whose depth
# is set by the first ("seed") box. Subsequent BPS-ordered boxes try to fit
# into the current slab via _find_best_in_slab; on miss, open a new slab in
# an existing bin; on miss-everywhere, open a new bin. Bit-identical to the
# Numba reference (decode_layer_njit in jit_decoders_geom.py).
# ---------------------------------------------------------------------------

def decode_layer_njit(
    bps_order, n_rots_per_box, dims_all,
    L, W, H, max_pallets, placements_out,
):
    """Cython implementation of decode_layer_njit. Matches the Numba
    reference signature + semantics exactly.
    """
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32

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

    cdef i64 n = bo_v.shape[0]
    return _layer_loop(
        bo_v, nr_v, da_v, cL, cW, cH, MAX_BINS, po_v,
        bin_emss, bin_ems_count, bin_slab_min_x, bin_slab_max_x,
        scratch, n,
    )


cdef i64 _layer_loop(
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
    i64 n,
) noexcept nogil:
    """All-nogil inner driver. The wrapper above only sets up arrays."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, box_idx, n_rots
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, yz
    cdef i64 best_bin, best_rot, best_x, best_y, best_z, best_yz
    cdef i64 slab_start, new_count, best_rot_n
    cdef bint placed

    for i in range(n):
        box_idx = bps_order[i]
        n_rots = n_rots_per_box[box_idx]
        placed = False

        # Phase 1: try existing slab in each bin (DFTRC-YZ scoring).
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_yz = -1
        for b in range(n_bins):
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
                if yz > best_yz:
                    best_yz = yz
                    best_bin = b
                    best_rot = r
                    best_x = x; best_y = y; best_z = z
            if best_bin >= 0:
                break

        if best_bin >= 0:
            r = best_rot
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            new_count = _commit_ems(
                bin_emss[best_bin], bin_ems_count[best_bin],
                best_x, best_y, best_z,
                best_x + dx, best_y + dy, best_z + dz,
                scratch)
            for j in range(new_count):
                bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
                bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
                bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
                bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
                bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
                bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
            bin_ems_count[best_bin] = new_count
            placements_out[i, 0] = best_bin
            placements_out[i, 1] = best_rot
            placements_out[i, 2] = best_x
            placements_out[i, 3] = best_y
            placements_out[i, 4] = best_z
            placements_out[i, 5] = 1
            continue

        # Phase 2: open new slab in an existing bin (this box becomes the seed).
        for b in range(n_bins):
            slab_start = bin_slab_max_x[b]
            if slab_start >= L:
                continue
            for r in range(n_rots):
                dx = dims_all[box_idx, r, 0]
                dy = dims_all[box_idx, r, 1]
                dz = dims_all[box_idx, r, 2]
                if slab_start + dx > L:
                    continue
                _find_best_in_slab(
                    bin_emss[b], bin_ems_count[b],
                    dx, dy, dz, L, W, H,
                    slab_start, slab_start + dx,
                    &idx, &x, &y, &z)
                if idx < 0:
                    continue
                new_count = _commit_ems(
                    bin_emss[b], bin_ems_count[b],
                    x, y, z, x + dx, y + dy, z + dz,
                    scratch)
                for j in range(new_count):
                    bin_emss[b, j, 0, 0] = scratch[j, 0, 0]
                    bin_emss[b, j, 0, 1] = scratch[j, 0, 1]
                    bin_emss[b, j, 0, 2] = scratch[j, 0, 2]
                    bin_emss[b, j, 1, 0] = scratch[j, 1, 0]
                    bin_emss[b, j, 1, 1] = scratch[j, 1, 1]
                    bin_emss[b, j, 1, 2] = scratch[j, 1, 2]
                bin_ems_count[b] = new_count
                bin_slab_min_x[b] = slab_start
                bin_slab_max_x[b] = slab_start + dx
                placements_out[i, 0] = b
                placements_out[i, 1] = r
                placements_out[i, 2] = x
                placements_out[i, 3] = y
                placements_out[i, 4] = z
                placements_out[i, 5] = 1
                placed = True
                break
            if placed:
                break
        if placed:
            continue

        # Phase 3: open a new bin (first-fit rotation; seeds the slab at x=0).
        if n_bins >= MAX_BINS:
            placements_out[i, 5] = 0
            continue
        bin_emss[n_bins, 0, 0, 0] = 0
        bin_emss[n_bins, 0, 0, 1] = 0
        bin_emss[n_bins, 0, 0, 2] = 0
        bin_emss[n_bins, 0, 1, 0] = L
        bin_emss[n_bins, 0, 1, 1] = W
        bin_emss[n_bins, 0, 1, 2] = H
        bin_ems_count[n_bins] = 1
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = 0
        best_rot_n = -1
        for r in range(n_rots):
            dx = dims_all[box_idx, r, 0]
            dy = dims_all[box_idx, r, 1]
            dz = dims_all[box_idx, r, 2]
            if dx <= L and dy <= W and dz <= H:
                best_rot_n = r
                break
        if best_rot_n < 0:
            placements_out[i, 5] = 0
            continue
        r = best_rot_n
        dx = dims_all[box_idx, r, 0]
        dy = dims_all[box_idx, r, 1]
        dz = dims_all[box_idx, r, 2]
        new_count = _commit_ems(
            bin_emss[n_bins], bin_ems_count[n_bins],
            0, 0, 0, dx, dy, dz, scratch)
        for j in range(new_count):
            bin_emss[n_bins, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[n_bins, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[n_bins, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[n_bins, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[n_bins, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[n_bins, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[n_bins] = new_count
        bin_slab_min_x[n_bins] = 0
        bin_slab_max_x[n_bins] = dx
        placements_out[i, 0] = n_bins
        placements_out[i, 1] = best_rot_n
        placements_out[i, 2] = 0
        placements_out[i, 3] = 0
        placements_out[i, 4] = 0
        placements_out[i, 5] = 1
        n_bins += 1
    return n_bins
