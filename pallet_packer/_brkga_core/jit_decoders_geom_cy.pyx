# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_decoders_geom_cy — Cython port of the
geometric decoders.

Phase 3a (this file): decode_njit_mode (modes 0/1/2 — DFTRC, wall, corner).
Phase 3b (this file): decode_layer_njit (mode 3 — layer-build).
Phase 3c (this file): decode_blocks_njit_mode (mode 4 — dynamic composite
                      blocks) + find_best_block_at_pos_njit helper.
Phase 3d (this file): decode_precomputed_blocks_njit_mode (mode 5 —
                      Bischoff 2002 pre-computed top-K blocks).

ALL geometric decoders now Cython. Phase 4 (constraint-aware decoders)
lives in jit_decoders_cstr_cy.pyx.

All cross-module calls use the .pxd `cimport` interface so the entire
inner loop runs nogil — no Python boundary, no tuple boxing.
"""
import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
# i64 + int64_t come from the companion jit_decoders_geom_cy.pxd

from .v3fast_cy cimport _find_best_dftrc, _commit_ems, MAX_EMS_C
from .jit_primitives_cy cimport (
    _find_best_wall, _find_best_corner, _find_best_in_slab,
)


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


# ---------------------------------------------------------------------------
# Phase 3c: find_best_block_at_pos + decode_blocks_njit_mode (mode 4 —
# Bischoff-Ratcliff 1995 dynamic composite blocks).
#
# Place single box via DFTRC, then extend to the largest (k, l, m) block of
# same-SKU unplaced boxes that fits inside an EMS containing the seed. Big
# wins on homogeneous loads (BR1/BR3); degenerates to k=l=m=1 with no
# measurable overhead on heterogeneous ones (BR7).
#
# The helper is declared in jit_decoders_geom_cy.pxd so the cstr blocks
# decoder (Phase 4) can reuse it via cimport.
# ---------------------------------------------------------------------------

cdef void _find_best_block_at_pos(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 x, i64 y, i64 z,
    i64 dx, i64 dy, i64 dz,
    i64 max_count,
    i64* out_k, i64* out_l, i64* out_m,
) noexcept nogil:
    """Find largest (k, l, m) block at position (x, y, z) that fits in some
    EMS containing the single-box footprint, with k*l*m <= max_count.
    Always returns at least (1, 1, 1).
    """
    # Floor-first block shape — see find_best_block_at_pos_njit in
    # jit_decoders_geom.py for the rationale (bit-identical twin): prefer the
    # largest floor FOOTPRINT (k*l), then the tallest stack (m) within it.
    cdef i64 best_k = 1, best_l = 1, best_m = 1, best_fp = 1, best_count = 1
    cdef i64 ei, ex_min, ey_min, ez_min, ex_max, ey_max, ez_max
    cdef i64 max_k, max_l, max_m, k, l, m, fp, count
    for ei in range(n_ems):
        ex_min = emss[ei, 0, 0]
        ey_min = emss[ei, 0, 1]
        ez_min = emss[ei, 0, 2]
        ex_max = emss[ei, 1, 0]
        ey_max = emss[ei, 1, 1]
        ez_max = emss[ei, 1, 2]
        if ex_min > x or ey_min > y or ez_min > z:
            continue
        if ex_max < x + dx or ey_max < y + dy or ez_max < z + dz:
            continue
        max_k = (ex_max - x) // dx
        if max_k > max_count:
            max_k = max_count
        max_l = (ey_max - y) // dy
        max_m = (ez_max - z) // dz
        if max_k < 1 or max_l < 1 or max_m < 1:
            continue
        for k in range(1, max_k + 1):
            if k > max_count:
                break
            for l in range(1, max_l + 1):
                fp = k * l
                if fp > max_count:
                    break
                m = max_m
                if fp * m > max_count:
                    m = max_count // fp
                count = fp * m
                if fp > best_fp or (fp == best_fp and count > best_count):
                    best_fp = fp
                    best_count = count
                    best_k = k
                    best_l = l
                    best_m = m
    out_k[0] = best_k
    out_l[0] = best_l
    out_m[0] = best_m


def find_best_block_at_pos_njit(emss, n_ems, x, y, z, dx, dy, dz, max_count):
    """Python wrapper. Returns (best_k, best_l, best_m)."""
    cdef i64 k, l, m
    cdef const i64[:, :, ::1] emss_v = emss
    _find_best_block_at_pos(
        emss_v, n_ems, x, y, z, dx, dy, dz, max_count,
        &k, &l, &m)
    return k, l, m


def decode_blocks_njit_mode(
    bps_order, n_rots_per_box, dims_all, sku_id_per_box,
    L, W, H, max_pallets, placements_out, n_skus,
):
    """Cython implementation of decode_blocks_njit_mode. Matches Numba
    reference signature + semantics exactly. Bit-identical at fixed seed.
    """
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef i64[:, ::1] po_v = placements_out

    cdef i64 n = bo_v.shape[0]
    cdef i64[::1] placed = np.zeros(n, dtype=np.int64)
    cdef i64[::1] sku_remaining = np.zeros(cn_skus, dtype=np.int64)
    return _blocks_loop(
        bo_v, nr_v, da_v, sku_v, cL, cW, cH, MAX_BINS, po_v,
        bin_emss, bin_ems_count, scratch, placed, sku_remaining, n,
    )


cdef i64 _blocks_loop(
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
    i64[::1] placed,
    i64[::1] sku_remaining,
    i64 n,
) noexcept nogil:
    """All-nogil inner driver for Phase 3c."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, box_idx, my_sku, n_rots, max_count
    cdef i64 dx, dy, dz
    cdef i64 idx, x, y, z, sc
    cdef i64 best_bin, best_rot, best_x, best_y, best_z, best_score
    cdef i64 best_rot_n, best_score_n, best_x_n, best_y_n, best_z_n
    cdef i64 k, l, m, block_count, placed_so_far, kk, ll, mm
    cdef i64 px, py, pz, new_count

    # Count remaining boxes per SKU.
    for i in range(n):
        box_idx = bps_order[i]
        sku_remaining[sku_id_per_box[box_idx]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        n_rots = n_rots_per_box[box_idx]
        max_count = sku_remaining[my_sku]

        # --- Phase 1: DFTRC placement (existing bins) ---
        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_score = -1
        for b in range(n_bins):
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
                    best_bin = b
                    best_rot = r
                    best_x = x; best_y = y; best_z = z
            if best_bin >= 0:
                break

        # --- Phase 2: open new bin if no fit in existing ones ---
        if best_bin < 0:
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
            best_x_n = 0
            best_y_n = 0
            best_z_n = 0
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
            best_bin = n_bins
            best_rot = best_rot_n
            best_x = best_x_n; best_y = best_y_n; best_z = best_z_n
            n_bins += 1

        # --- Phase 3: block extension at chosen DFTRC position ---
        dx = dims_all[box_idx, best_rot, 0]
        dy = dims_all[box_idx, best_rot, 1]
        dz = dims_all[box_idx, best_rot, 2]
        _find_best_block_at_pos(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z, dx, dy, dz, max_count,
            &k, &l, &m)
        block_count = k * l * m

        # --- Phase 4: assign positions to next 'block_count' unplaced
        # same-SKU boxes in BPS order (first is current i). ---
        placed_so_far = 0
        kk = 0
        ll = 0
        mm = 0
        for j in range(i, n):
            if placed_so_far >= block_count:
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
            placed_so_far += 1
            kk += 1
            if kk >= k:
                kk = 0
                ll += 1
                if ll >= l:
                    ll = 0
                    mm += 1
        sku_remaining[my_sku] -= placed_so_far

        # --- Phase 5: single EMS commit for the whole block region ---
        new_count = _commit_ems(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z,
            best_x + k * dx, best_y + l * dy, best_z + m * dz,
            scratch)
        for j in range(new_count):
            bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[best_bin] = new_count

    return n_bins


# ---------------------------------------------------------------------------
# Phase 3d: decode_precomputed_blocks_njit_mode (mode 5 — Bischoff 2002).
#
# Per-SKU pre-computed (k, l, m, rot) block sized for the whole pallet (not
# the current EMS). Phase 1 tries to place that exact pre-computed block at
# DFTRC for its block dims. Phase 2 falls back to mode-4 (single-box DFTRC +
# dynamic block extension). Phase 3 (new bin) tries pre-computed block first.
#
# Bit-identical to the Numba reference at fixed seed (see
# scripts/ab_test_decoder_precomputed.py).
# ---------------------------------------------------------------------------

def decode_precomputed_blocks_njit_mode(
    bps_order, n_rots_per_box, dims_all, sku_id_per_box, sku_best_block,
    L, W, H, max_pallets, placements_out, n_skus,
):
    """Cython implementation of decode_precomputed_blocks_njit_mode.

    sku_best_block: (n_skus, 4) — columns (k, l, m, rot) per SKU.
    """
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus

    cdef i64[:, :, :, ::1] bin_emss = np.zeros(
        (MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[::1] bin_ems_count = np.zeros(MAX_BINS, dtype=np.int64)
    cdef i64[:, :, ::1] scratch = np.zeros((MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef const i64[::1] bo_v = bps_order
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef const i64[:, ::1] sbb_v = sku_best_block
    cdef i64[:, ::1] po_v = placements_out

    cdef i64 n = bo_v.shape[0]
    cdef i64[::1] placed = np.zeros(n, dtype=np.int64)
    cdef i64[::1] sku_remaining = np.zeros(cn_skus, dtype=np.int64)
    return _precomputed_blocks_loop(
        bo_v, nr_v, da_v, sku_v, sbb_v, cL, cW, cH, MAX_BINS, po_v,
        bin_emss, bin_ems_count, scratch, placed, sku_remaining, n,
    )


cdef i64 _precomputed_blocks_loop(
    const i64[::1] bps_order,
    const i64[::1] n_rots_per_box,
    const i64[:, :, ::1] dims_all,
    const i64[::1] sku_id_per_box,
    const i64[:, ::1] sku_best_block,
    i64 L, i64 W, i64 H,
    i64 MAX_BINS,
    i64[:, ::1] placements_out,
    i64[:, :, :, ::1] bin_emss,
    i64[::1] bin_ems_count,
    i64[:, :, ::1] scratch,
    i64[::1] placed,
    i64[::1] sku_remaining,
    i64 n,
) noexcept nogil:
    """All-nogil inner driver for Phase 3d."""
    cdef i64 n_bins = 0
    cdef i64 i, b, r, j, box_idx, my_sku, max_count, n_rots
    cdef i64 k_pre, l_pre, m_pre, rot_pre, n_pre
    cdef i64 dx_pre, dy_pre, dz_pre, block_dx, block_dy, block_dz
    cdef i64 best_bin, best_rot, best_x, best_y, best_z
    cdef i64 best_k, best_l, best_m
    cdef i64 best_score
    cdef i64 idx, x, y, z, sc
    cdef i64 single_best_score, single_best_rot, single_best_bin
    cdef i64 single_best_x, single_best_y, single_best_z
    cdef i64 dx, dy, dz, ek, el, em
    cdef bint placed_in_new
    cdef i64 k, l, m, block_count, placed_so_far, kk, ll, mm
    cdef i64 px, py, pz, new_count

    # Count remaining boxes per SKU.
    for i in range(n):
        sku_remaining[sku_id_per_box[bps_order[i]]] += 1

    for i in range(n):
        if placed[i] == 1:
            continue
        box_idx = bps_order[i]
        my_sku = sku_id_per_box[box_idx]
        max_count = sku_remaining[my_sku]
        n_rots = n_rots_per_box[box_idx]

        # Pre-computed block dims for this SKU.
        k_pre = sku_best_block[my_sku, 0]
        l_pre = sku_best_block[my_sku, 1]
        m_pre = sku_best_block[my_sku, 2]
        rot_pre = sku_best_block[my_sku, 3]
        n_pre = k_pre * l_pre * m_pre

        best_bin = -1
        best_rot = -1
        best_x = 0
        best_y = 0
        best_z = 0
        best_k = 1
        best_l = 1
        best_m = 1

        # --- Phase 1: try pre-computed block at DFTRC for block dims ---
        if n_pre > 1 and max_count >= n_pre:
            dx_pre = dims_all[box_idx, rot_pre, 0]
            dy_pre = dims_all[box_idx, rot_pre, 1]
            dz_pre = dims_all[box_idx, rot_pre, 2]
            block_dx = k_pre * dx_pre
            block_dy = l_pre * dy_pre
            block_dz = m_pre * dz_pre
            best_score = -1
            for b in range(n_bins):
                _find_best_dftrc(
                    bin_emss[b], bin_ems_count[b],
                    block_dx, block_dy, block_dz, L, W, H,
                    &idx, &x, &y, &z)
                if idx < 0:
                    continue
                sc = ((L - x - block_dx) * (L - x - block_dx)
                      + (W - y - block_dy) * (W - y - block_dy)
                      + (H - z - block_dz) * (H - z - block_dz))
                if sc > best_score:
                    best_score = sc
                    best_bin = b
                    best_rot = rot_pre
                    best_x = x; best_y = y; best_z = z
                    best_k = k_pre; best_l = l_pre; best_m = m_pre

        # --- Phase 2: fallback — mode 4 style single-box DFTRC + dynamic ---
        if best_bin < 0:
            single_best_score = -1
            single_best_rot = -1
            single_best_x = 0
            single_best_y = 0
            single_best_z = 0
            single_best_bin = -1
            for b in range(n_bins):
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
                    if sc > single_best_score:
                        single_best_score = sc
                        single_best_bin = b
                        single_best_rot = r
                        single_best_x = x
                        single_best_y = y
                        single_best_z = z
                if single_best_bin >= 0:
                    break
            if single_best_bin >= 0:
                dx = dims_all[box_idx, single_best_rot, 0]
                dy = dims_all[box_idx, single_best_rot, 1]
                dz = dims_all[box_idx, single_best_rot, 2]
                _find_best_block_at_pos(
                    bin_emss[single_best_bin],
                    bin_ems_count[single_best_bin],
                    single_best_x, single_best_y, single_best_z,
                    dx, dy, dz, max_count,
                    &ek, &el, &em)
                best_bin = single_best_bin
                best_rot = single_best_rot
                best_x = single_best_x
                best_y = single_best_y
                best_z = single_best_z
                best_k = ek; best_l = el; best_m = em

        # --- Phase 3: still no fit → open new bin (pre-computed first) ---
        if best_bin < 0:
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
            placed_in_new = False
            if n_pre > 1 and max_count >= n_pre:
                dx_pre = dims_all[box_idx, rot_pre, 0]
                dy_pre = dims_all[box_idx, rot_pre, 1]
                dz_pre = dims_all[box_idx, rot_pre, 2]
                block_dx = k_pre * dx_pre
                block_dy = l_pre * dy_pre
                block_dz = m_pre * dz_pre
                if block_dx <= L and block_dy <= W and block_dz <= H:
                    best_bin = n_bins
                    best_rot = rot_pre
                    best_x = 0; best_y = 0; best_z = 0
                    best_k = k_pre; best_l = l_pre; best_m = m_pre
                    placed_in_new = True
                    n_bins += 1
            if not placed_in_new:
                single_best_rot = -1
                for r in range(n_rots):
                    dx = dims_all[box_idx, r, 0]
                    dy = dims_all[box_idx, r, 1]
                    dz = dims_all[box_idx, r, 2]
                    if dx <= L and dy <= W and dz <= H:
                        single_best_rot = r
                        break
                if single_best_rot < 0:
                    placements_out[i, 5] = 0
                    placed[i] = 1
                    sku_remaining[my_sku] -= 1
                    continue
                dx = dims_all[box_idx, single_best_rot, 0]
                dy = dims_all[box_idx, single_best_rot, 1]
                dz = dims_all[box_idx, single_best_rot, 2]
                _find_best_block_at_pos(
                    bin_emss[n_bins], 1, 0, 0, 0, dx, dy, dz, max_count,
                    &ek, &el, &em)
                best_bin = n_bins
                best_rot = single_best_rot
                best_x = 0; best_y = 0; best_z = 0
                best_k = ek; best_l = el; best_m = em
                n_bins += 1

        # --- Phase 4: assign positions to next block_count same-SKU boxes ---
        dx = dims_all[box_idx, best_rot, 0]
        dy = dims_all[box_idx, best_rot, 1]
        dz = dims_all[box_idx, best_rot, 2]
        k = best_k; l = best_l; m = best_m
        block_count = k * l * m

        placed_so_far = 0
        kk = 0; ll = 0; mm = 0
        for j in range(i, n):
            if placed_so_far >= block_count:
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
            placed_so_far += 1
            kk += 1
            if kk >= k:
                kk = 0
                ll += 1
                if ll >= l:
                    ll = 0
                    mm += 1
        sku_remaining[my_sku] -= placed_so_far

        # --- Phase 5: commit EMS for the block region ---
        new_count = _commit_ems(
            bin_emss[best_bin], bin_ems_count[best_bin],
            best_x, best_y, best_z,
            best_x + k * dx, best_y + l * dy, best_z + m * dz,
            scratch)
        for j in range(new_count):
            bin_emss[best_bin, j, 0, 0] = scratch[j, 0, 0]
            bin_emss[best_bin, j, 0, 1] = scratch[j, 0, 1]
            bin_emss[best_bin, j, 0, 2] = scratch[j, 0, 2]
            bin_emss[best_bin, j, 1, 0] = scratch[j, 1, 0]
            bin_emss[best_bin, j, 1, 1] = scratch[j, 1, 1]
            bin_emss[best_bin, j, 1, 2] = scratch[j, 1, 2]
        bin_ems_count[best_bin] = new_count

    return n_bins


# ===========================================================================
# Phase 5a: batch decode for modes 0/1/2 via cython.parallel.prange.
#
# Each chromosome in the population is decoded independently → embarrassingly
# parallel. Per-chromosome scratch arrays are pre-allocated (one set per
# chromosome) so the inner loop runs nogil without threadid-based fan-out.
# `schedule='static'` makes chromosome i always land on the same thread for a
# given pop_size → bit-identical regardless of thread count.
# ===========================================================================


def decode_batch_njit_mode(
    chromosomes,         # (pop_size, chrom_len) double — only [:, :n] used
    n_rots_per_box,      # (n_boxes,) int64
    dims_all,            # (n_boxes, n_rots_max, 3) int64
    L, W, H,
    max_pallets,
    mode,                # 0=DFTRC, 1=wall, 2=corner — uniform across batch
    placements_out_all,  # (pop_size, n_boxes, 6) int64
    n_bins_out,          # (pop_size,) int64
):
    """Batch-decode pop_size chromosomes with modes 0/1/2 in parallel.

    Bit-identical to running decode_njit_mode serially for each chromosome.
    Wrapper does the per-chromosome BPS argsort (numpy/Python) up front;
    the parallel prange dispatches the nogil decode loops.
    """
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef int cmode = mode
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32

    # Per-chromosome BPS argsort. Numpy is GIL-bound; do it once up front.
    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    # Per-chromosome scratch — one slot per chromosome means no thread-id
    # fan-out is needed; each prange iteration writes only into its own slot.
    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            # Cython 3.x: indexing contiguous memoryviews returns a sub-view
            # without acquiring GIL. _decode_loop already takes per-call
            # scratch + placements, so each chromosome's slot is independent.
            nb_out[i] = _decode_loop(
                orders[i], nr_v, da_v, cL, cW, cH, MAX_BINS,
                po_all[i], cmode,
                be_all[i], bec_all[i], sc_all[i], n_boxes,
            )
    return n_bins_out


# ===========================================================================
# Phase 5b: batch entry points for the remaining geometric decoders
# (layer / blocks / precomputed_blocks). Same per-chromosome scratch +
# prange pattern as Phase 5a.
# ===========================================================================


def decode_batch_layer_njit(
    chromosomes, n_rots_per_box, dims_all,
    L, W, H, max_pallets,
    placements_out_all, n_bins_out,
):
    """Batch-decode pop_size chromosomes for mode 3 (layer-build) in parallel."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, ::1] sm_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, ::1] sx_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _layer_loop(
                orders[i], nr_v, da_v, cL, cW, cH, MAX_BINS,
                po_all[i],
                be_all[i], bec_all[i], sm_all[i], sx_all[i], sc_all[i],
                n_boxes,
            )
    return n_bins_out


def decode_batch_blocks_njit_mode(
    chromosomes, n_rots_per_box, dims_all, sku_id_per_box,
    L, W, H, max_pallets,
    placements_out_all, n_bins_out, n_skus,
):
    """Batch-decode pop_size chromosomes for mode 4 (blocks) in parallel."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] placed_all = np.zeros((pop_size, n_boxes), dtype=np.int64)
    cdef i64[:, ::1] skur_all = np.zeros((pop_size, cn_skus), dtype=np.int64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _blocks_loop(
                orders[i], nr_v, da_v, sku_v, cL, cW, cH, MAX_BINS,
                po_all[i],
                be_all[i], bec_all[i], sc_all[i],
                placed_all[i], skur_all[i], n_boxes,
            )
    return n_bins_out


def decode_batch_precomputed_blocks_njit_mode(
    chromosomes, n_rots_per_box, dims_all, sku_id_per_box, sku_best_block,
    L, W, H, max_pallets,
    placements_out_all, n_bins_out, n_skus,
):
    """Batch-decode pop_size chromosomes for mode 5 (precomputed blocks)."""
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef const i64[:, ::1] sbb_v = sku_best_block
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] placed_all = np.zeros((pop_size, n_boxes), dtype=np.int64)
    cdef i64[:, ::1] skur_all = np.zeros((pop_size, cn_skus), dtype=np.int64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _precomputed_blocks_loop(
                orders[i], nr_v, da_v, sku_v, sbb_v,
                cL, cW, cH, MAX_BINS, po_all[i],
                be_all[i], bec_all[i], sc_all[i],
                placed_all[i], skur_all[i], n_boxes,
            )
    return n_bins_out


def decode_batch_precomputed_blocks_per_chrom(
    chromosomes, n_rots_per_box, dims_all, sku_id_per_box,
    chosen_per_chrom,        # (pop_size, n_skus, 4) — pre-resolved by caller
    L, W, H, max_pallets,
    placements_out_all, n_bins_out, n_skus,
):
    """Phase 5c fix: mode 5 with top-K needs per-chromosome block resolution.

    The Phase 5b decode_batch_precomputed_blocks_njit_mode took a single
    sku_best_block shared across the population — fine when block selection
    is global (v3.8 best-per-SKU). With the v3.8 top-K enhancement, each
    chromosome resolves its own (k, l, m, rot) per SKU from chrom keys, so
    the batch entry needs (pop_size, n_skus, 4) chosen arrays.

    Caller resolves chosen_per_chrom outside this function (numpy/Python);
    this wrapper slices per-chromosome and runs the existing nogil
    _precomputed_blocks_loop in prange.
    """
    cdef Py_ssize_t pop_size = chromosomes.shape[0]
    cdef i64 n_boxes = n_rots_per_box.shape[0]
    cdef i64 cL = L, cW = W, cH = H
    cdef i64 MAX_BINS = max_pallets if max_pallets > 0 else 32
    cdef i64 cn_skus = n_skus

    bps = np.ascontiguousarray(chromosomes)[:, :n_boxes]
    orders_np = np.argsort(bps, axis=1).astype(np.int64)
    cdef const i64[:, ::1] orders = orders_np
    cdef const i64[::1] nr_v = n_rots_per_box
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] sku_v = sku_id_per_box
    cdef const i64[:, :, ::1] cpc = chosen_per_chrom    # (pop_size, n_skus, 4)
    cdef i64[:, :, ::1] po_all = placements_out_all
    cdef i64[::1] nb_out = n_bins_out

    cdef i64[:, :, :, :, ::1] be_all = np.zeros(
        (pop_size, MAX_BINS, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] bec_all = np.zeros((pop_size, MAX_BINS), dtype=np.int64)
    cdef i64[:, :, :, ::1] sc_all = np.zeros(
        (pop_size, MAX_EMS_C, 2, 3), dtype=np.int64)
    cdef i64[:, ::1] placed_all = np.zeros((pop_size, n_boxes), dtype=np.int64)
    cdef i64[:, ::1] skur_all = np.zeros((pop_size, cn_skus), dtype=np.int64)

    cdef Py_ssize_t i
    with nogil:
        for i in prange(pop_size, schedule='static'):
            nb_out[i] = _precomputed_blocks_loop(
                orders[i], nr_v, da_v, sku_v, cpc[i],
                cL, cW, cH, MAX_BINS, po_all[i],
                be_all[i], bec_all[i], sc_all[i],
                placed_all[i], skur_all[i], n_boxes,
            )
    return n_bins_out
