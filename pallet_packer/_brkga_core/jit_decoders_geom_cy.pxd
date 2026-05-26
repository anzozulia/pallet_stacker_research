# cython: language_level=3
"""
Declarations exposed to other Cython modules via `cimport`. Implementations
live in jit_decoders_geom_cy.pyx.

Only helpers that the constraint-aware decoders (Phase 4,
jit_decoders_cstr_cy.pyx) will reuse appear here. The decoder entry points
themselves are Python-callable defs and don't need .pxd declarations.
"""
from libc.stdint cimport int64_t

ctypedef int64_t i64


cdef void _find_best_block_at_pos(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 x, i64 y, i64 z,
    i64 dx, i64 dy, i64 dz,
    i64 max_count,
    i64* out_k, i64* out_l, i64* out_m,
) noexcept nogil
