# cython: language_level=3
"""
Declarations exposed to other Cython modules via `cimport`. The actual
implementations live in v3fast_cy.pyx. C functions declared here are
inlined into callers (no Python boundary, no GIL).
"""
from libc.stdint cimport int64_t

ctypedef int64_t i64

# Match the constant in brkga_v3_fast.py — keep both in sync.
cdef int MAX_EMS_C


cdef void _find_best_dftrc(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil


cdef i64 _commit_ems(
    const i64[:, :, ::1] emss, i64 n_ems,
    i64 bx1, i64 by1, i64 bz1,
    i64 bx2, i64 by2, i64 bz2,
    i64[:, :, ::1] out_buf,
) noexcept nogil
