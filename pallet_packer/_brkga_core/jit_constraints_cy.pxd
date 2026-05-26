# cython: language_level=3
"""
Declarations exposed to other Cython modules via `cimport`. Implementations
live in jit_constraints_cy.pyx.

Used by the constraint-aware decoders in jit_decoders_cstr_cy.pyx (Phase 4)
to call the feasibility checks + bookkeeping helpers at C level with no
Python boundary or GIL acquisition.
"""
from libc.stdint cimport int64_t

ctypedef int64_t i64


cdef bint _ck_load_on_top(
    const i64[:, ::1] placements_out,
    const i64[:, :, ::1] dims_all,
    const i64[::1] bps_order,
    const double[::1] mlot,
    const double[::1] placement_top_loads,
    i64 n_placed,
    i64 cand_pallet,
    i64 cand_x, i64 cand_y, i64 cand_z,
    i64 cand_dx, i64 cand_dy, i64 cand_dz,
    double cand_weight,
    double support_ratio,
    int require_centroid,
    int require_full_support,
) noexcept nogil


cdef bint _ck_cog_envelope(
    i64 cand_x, i64 cand_y,
    i64 cand_dx, i64 cand_dy,
    double cand_weight,
    i64 cand_pallet,
    const double[::1] pallet_weights,
    const double[::1] pallet_sum_xw,
    const double[::1] pallet_sum_yw,
    double pallet_max_weight,
    double cog_x_min, double cog_x_max,
    double cog_y_min, double cog_y_max,
    double cog_min_load_frac,
) noexcept nogil


cdef void _ap_cog_contribution(
    i64 cand_x, i64 cand_y,
    i64 cand_dx, i64 cand_dy,
    double cand_weight,
    i64 cand_pallet,
    double[::1] pallet_sum_xw,
    double[::1] pallet_sum_yw,
) noexcept nogil


cdef void _ap_load_contribution(
    const i64[:, ::1] placements_out,
    const i64[:, :, ::1] dims_all,
    const i64[::1] bps_order,
    double[::1] placement_top_loads,
    i64 n_placed,
    i64 cand_pallet,
    i64 cand_x, i64 cand_y, i64 cand_z,
    i64 cand_dx, i64 cand_dy, i64 cand_dz,
    double cand_weight,
) noexcept nogil
