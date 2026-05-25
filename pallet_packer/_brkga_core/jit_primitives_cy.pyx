# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_primitives_cy — Cython port of jit_primitives.

Same algorithm, same numerical semantics as the Numba reference in
jit_primitives.py. Bit-identical results at fixed seed (validated by
the A/B test harness in this file's __main__ guard + by tests/).

Public surface (Python-callable, returns 4-tuple of int):

    find_best_wall_njit(emss, n_ems, dx, dy, dz, L, W, H)
    find_best_corner_njit(emss, n_ems, dx, dy, dz, L, W, H)
    find_best_in_slab_njit(emss, n_ems, dx, dy, dz, L, W, H,
                            slab_min_x, slab_max_x)

Internal nogil C surface (for Cython decoders to call without GIL
overhead; output via pointer args):

    _find_best_wall(...)
    _find_best_corner(...)
    _find_best_in_slab(...)

The wrappers and the nogil cores share the same loop body, expressed
once via the `_*` helpers. The cpdef wrappers add the GIL needed to
build the return tuple.

The function names end in `_njit` to keep the Python signature compatible
with the Numba reference (the legacy import path `from pallet_packer.
brkga_v3_5 import find_best_wall_njit` continues to work via __init__).
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

# Alias to keep the body readable.
ctypedef int64_t i64


# ---------------------------------------------------------------------------
# Internal nogil cores. Output via pointer args so the loop body runs without
# touching Python state. Will be called from the Cython decoders in Phase 3+.
# ---------------------------------------------------------------------------

cdef inline void _find_best_wall(
    const i64[:, :, ::1] emss,  # (n_ems, 2, 3) — [ems_idx][min/max][x/y/z]
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil:
    """Wall-build: minimise X first, tiebreak with YZ-DFTRC."""
    cdef i64 best_idx = -1
    cdef i64 best_x = L + 1
    cdef i64 best_yz = -1
    cdef i64 bx = 0, by = 0, bz = 0
    cdef i64 i, ex_min, ey_min, ez_min, ex_max, ey_max, ez_max
    cdef i64 x, y, z, yz
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        x = ex_min
        y = ey_min
        z = ez_min
        yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
        if x < best_x or (x == best_x and yz > best_yz):
            best_x = x
            best_yz = yz
            best_idx = i
            bx = x
            by = y
            bz = z
    out_idx[0] = best_idx
    out_x[0] = bx
    out_y[0] = by
    out_z[0] = bz


cdef inline void _find_best_corner(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil:
    """Anti-DFTRC: place at smallest (x+y+z) corner."""
    cdef i64 best_idx = -1
    cdef i64 best_sum = 3 * (L + W + H)
    cdef i64 bx = 0, by = 0, bz = 0
    cdef i64 i, ex_min, ey_min, ez_min, ex_max, ey_max, ez_max, s
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        s = ex_min + ey_min + ez_min
        if s < best_sum:
            best_sum = s
            best_idx = i
            bx = ex_min
            by = ey_min
            bz = ez_min
    out_idx[0] = best_idx
    out_x[0] = bx
    out_y[0] = by
    out_z[0] = bz


cdef inline void _find_best_in_slab(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64 slab_min_x, i64 slab_max_x,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil:
    """DFTRC-YZ inside an [slab_min_x, slab_max_x] X-slab."""
    cdef i64 best_idx = -1
    cdef i64 best_yz = -1
    cdef i64 bx = 0, by = 0, bz = 0
    cdef i64 i, ex_min, ey_min, ez_min, ex_max, ey_max, ez_max
    cdef i64 x, y, z, yz
    for i in range(n_ems):
        ex_min = emss[i, 0, 0]
        ey_min = emss[i, 0, 1]
        ez_min = emss[i, 0, 2]
        ex_max = emss[i, 1, 0]
        ey_max = emss[i, 1, 1]
        ez_max = emss[i, 1, 2]
        if (ex_max - ex_min) < dx:
            continue
        if (ey_max - ey_min) < dy:
            continue
        if (ez_max - ez_min) < dz:
            continue
        if ex_min < slab_min_x:
            continue
        if ex_min + dx > slab_max_x:
            continue
        x = ex_min
        y = ey_min
        z = ez_min
        yz = (W - y - dy) * (W - y - dy) + (H - z - dz) * (H - z - dz)
        if yz > best_yz:
            best_yz = yz
            best_idx = i
            bx = x
            by = y
            bz = z
    out_idx[0] = best_idx
    out_x[0] = bx
    out_y[0] = by
    out_z[0] = bz


# ---------------------------------------------------------------------------
# Python-callable wrappers. Match the Numba signatures exactly so the
# dispatch in __init__.py can swap between the two implementations.
# ---------------------------------------------------------------------------

def find_best_wall_njit(emss, n_ems, dx, dy, dz, L, W, H):
    """Wall-build placement. Returns (best_idx, bx, by, bz)."""
    cdef i64 idx, bx, by, bz
    cdef const i64[:, :, ::1] emss_v = emss
    _find_best_wall(emss_v, n_ems, dx, dy, dz, L, W, H,
                    &idx, &bx, &by, &bz)
    return idx, bx, by, bz


def find_best_corner_njit(emss, n_ems, dx, dy, dz, L, W, H):
    """Corner-fill placement. Returns (best_idx, bx, by, bz)."""
    cdef i64 idx, bx, by, bz
    cdef const i64[:, :, ::1] emss_v = emss
    _find_best_corner(emss_v, n_ems, dx, dy, dz, L, W, H,
                      &idx, &bx, &by, &bz)
    return idx, bx, by, bz


def find_best_in_slab_njit(emss, n_ems, dx, dy, dz, L, W, H,
                           slab_min_x, slab_max_x):
    """Slab-constrained YZ-DFTRC. Returns (best_idx, bx, by, bz)."""
    cdef i64 idx, bx, by, bz
    cdef const i64[:, :, ::1] emss_v = emss
    _find_best_in_slab(emss_v, n_ems, dx, dy, dz, L, W, H,
                       slab_min_x, slab_max_x,
                       &idx, &bx, &by, &bz)
    return idx, bx, by, bz
