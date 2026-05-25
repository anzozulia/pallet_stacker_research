# cython: language_level=3
"""
Declarations exposed to other Cython modules via `cimport`. The actual
implementations live in jit_primitives_cy.pyx.
"""
from libc.stdint cimport int64_t

ctypedef int64_t i64


cdef void _find_best_wall(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil


cdef void _find_best_corner(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil


cdef void _find_best_in_slab(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64 slab_min_x, i64 slab_max_x,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil
