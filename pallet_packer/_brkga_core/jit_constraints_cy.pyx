# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.jit_constraints_cy — Cython port of jit_constraints.

Four feasibility / bookkeeping helpers, same numerical semantics as the
Numba reference in jit_constraints.py. Bit-identical results on the
A/B harness (see scripts/ab_test_constraints.py).

Public surface (Python-callable, drop-in for the Numba originals):

    _check_load_on_top_njit(placements_out, dims_all, bps_order, mlot,
                             placement_top_loads, n_placed,
                             cand_pallet, cand_x, cand_y, cand_z,
                             cand_dx, cand_dy, cand_dz,
                             cand_weight, support_ratio,
                             require_centroid=0, require_full_support=0)
    _check_cog_envelope_njit(cand_x, cand_y, cand_dx, cand_dy,
                             cand_weight, cand_pallet,
                             pallet_weights, pallet_sum_xw, pallet_sum_yw,
                             pallet_max_weight,
                             cog_x_min, cog_x_max, cog_y_min, cog_y_max,
                             cog_min_load_frac)
    _apply_cog_contribution_njit(cand_x, cand_y, cand_dx, cand_dy,
                                  cand_weight, cand_pallet,
                                  pallet_sum_xw, pallet_sum_yw)
    _apply_load_contribution_njit(placements_out, dims_all, bps_order,
                                   placement_top_loads, n_placed,
                                   cand_pallet,
                                   cand_x, cand_y, cand_z,
                                   cand_dx, cand_dy, cand_dz,
                                   cand_weight)

Internal nogil C surface (called by Phase 3+ Cython decoders):

    _ck_load_on_top(...) -> bint
    _ck_cog_envelope(...) -> bint
    _ap_cog_contribution(...) -> void
    _ap_load_contribution(...) -> void
"""
import numpy as np
cimport numpy as cnp
from libc.stdint cimport int64_t

ctypedef int64_t i64

# Sentinel mirroring jit_constraints._NO_LIMIT. Keep in sync with the
# Numba module — used to short-circuit the CoG check when there's no
# pallet weight cap to compute a load-fraction against.
cdef double _NO_LIMIT = 1e18


# ---------------------------------------------------------------------------
# Internal nogil cores.
# ---------------------------------------------------------------------------

cdef bint _ck_load_on_top(
    const i64[:, ::1] placements_out,    # (n, 6)
    const i64[:, :, ::1] dims_all,       # (n_boxes, n_rots_max, 3)
    const i64[::1] bps_order,            # (n_placed,)
    const double[::1] mlot,              # (n_boxes,)
    const double[::1] placement_top_loads,  # (n_placed,)
    i64 n_placed,
    i64 cand_pallet,
    i64 cand_x, i64 cand_y, i64 cand_z,
    i64 cand_dx, i64 cand_dy, i64 cand_dz,
    double cand_weight,
    double support_ratio,
    int require_centroid,
    int require_full_support,
) noexcept nogil:
    """Two-pass: total contact area → support_ratio + centroid; then per-
    supporter load distribution → max_load_on_top. Floor placements (z<=0)
    return True immediately.
    """
    if cand_z <= 0:
        return True
    cdef i64 cx2 = 2 * cand_x + cand_dx
    cdef i64 cy2 = 2 * cand_y + cand_dy
    cdef int centroid_supported = 0
    cdef double total_area = 0.0
    cdef i64 i, sup_box, sup_rot
    cdef i64 sup_dx, sup_dy, sup_dz, sup_x, sup_y, sup_z
    cdef i64 x_lo, x_hi, y_lo, y_hi
    cdef double area, share, footprint, eff_sr
    # Pass 1: accumulate contact area + centroid check.
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        total_area += <double>((x_hi - x_lo) * (y_hi - y_lo))
        if centroid_supported == 0 and require_centroid != 0:
            if (cx2 >= 2 * sup_x and cx2 <= 2 * (sup_x + sup_dx)
                    and cy2 >= 2 * sup_y and cy2 <= 2 * (sup_y + sup_dy)):
                centroid_supported = 1
    if total_area <= 0.0:
        return False
    eff_sr = 1.0 if require_full_support != 0 else support_ratio
    if eff_sr > 0.0:
        footprint = <double>(cand_dx * cand_dy)
        if footprint > 0 and total_area / footprint < eff_sr - 1e-6:
            return False
    if require_centroid != 0 and centroid_supported == 0:
        return False
    # Pass 2: per-supporter load capacity.
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        area = <double>((x_hi - x_lo) * (y_hi - y_lo))
        share = cand_weight * (area / total_area)
        if placement_top_loads[i] + share > mlot[sup_box] + 1e-6:
            return False
    return True


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
) noexcept nogil:
    """Pallet CoG within envelope after adding cand. Skips while pallet
    is lightly loaded (matches v2 packer.py:_cog_ok)."""
    cdef double new_total = pallet_weights[cand_pallet] + cand_weight
    if new_total <= 0.0:
        return True
    if pallet_max_weight < _NO_LIMIT:
        if new_total < cog_min_load_frac * pallet_max_weight:
            return True
    cdef double cand_cx = <double>cand_x + 0.5 * <double>cand_dx
    cdef double cand_cy = <double>cand_y + 0.5 * <double>cand_dy
    cdef double new_sum_xw = pallet_sum_xw[cand_pallet] + cand_weight * cand_cx
    cdef double new_sum_yw = pallet_sum_yw[cand_pallet] + cand_weight * cand_cy
    cdef double cx = new_sum_xw / new_total
    cdef double cy = new_sum_yw / new_total
    if cx < cog_x_min - 1e-6 or cx > cog_x_max + 1e-6:
        return False
    if cy < cog_y_min - 1e-6 or cy > cog_y_max + 1e-6:
        return False
    return True


cdef void _ap_cog_contribution(
    i64 cand_x, i64 cand_y,
    i64 cand_dx, i64 cand_dy,
    double cand_weight,
    i64 cand_pallet,
    double[::1] pallet_sum_xw,
    double[::1] pallet_sum_yw,
) noexcept nogil:
    """Additive maintenance of per-pallet weighted-position sums."""
    if cand_weight <= 0.0:
        return
    pallet_sum_xw[cand_pallet] += cand_weight * (<double>cand_x + 0.5 * <double>cand_dx)
    pallet_sum_yw[cand_pallet] += cand_weight * (<double>cand_y + 0.5 * <double>cand_dy)


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
) noexcept nogil:
    """Update placement_top_loads after a successful placement. Must be
    called only after _ck_load_on_top returned True."""
    if cand_z <= 0 or cand_weight <= 0:
        return
    cdef double total_area = 0.0
    cdef i64 i, sup_box, sup_rot
    cdef i64 sup_dx, sup_dy, sup_dz, sup_x, sup_y, sup_z
    cdef i64 x_lo, x_hi, y_lo, y_hi
    cdef double area
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        total_area += <double>((x_hi - x_lo) * (y_hi - y_lo))
    if total_area <= 0.0:
        return
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        sup_box = bps_order[i]
        sup_rot = placements_out[i, 1]
        sup_dz = dims_all[sup_box, sup_rot, 2]
        sup_z = placements_out[i, 4]
        if sup_z + sup_dz != cand_z:
            continue
        sup_dx = dims_all[sup_box, sup_rot, 0]
        sup_dy = dims_all[sup_box, sup_rot, 1]
        sup_x = placements_out[i, 2]
        sup_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > sup_x else sup_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < sup_x + sup_dx else sup_x + sup_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > sup_y else sup_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < sup_y + sup_dy else sup_y + sup_dy
        if y_hi <= y_lo:
            continue
        area = <double>((x_hi - x_lo) * (y_hi - y_lo))
        placement_top_loads[i] += cand_weight * (area / total_area)


# ---------------------------------------------------------------------------
# Python-callable wrappers — match Numba signatures exactly.
# ---------------------------------------------------------------------------

def _check_load_on_top_njit(
    placements_out, dims_all, bps_order, mlot,
    placement_top_loads, n_placed,
    cand_pallet,
    cand_x, cand_y, cand_z,
    cand_dx, cand_dy, cand_dz,
    cand_weight,
    support_ratio,
    require_centroid=0,
    require_full_support=0,
):
    cdef const i64[:, ::1] po_v = placements_out
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] bo_v = bps_order
    cdef const double[::1] mlot_v = mlot
    cdef const double[::1] ptl_v = placement_top_loads
    return _ck_load_on_top(
        po_v, da_v, bo_v, mlot_v, ptl_v, n_placed,
        cand_pallet, cand_x, cand_y, cand_z,
        cand_dx, cand_dy, cand_dz,
        cand_weight, support_ratio,
        require_centroid, require_full_support,
    )


def _check_cog_envelope_njit(
    cand_x, cand_y, cand_dx, cand_dy, cand_weight, cand_pallet,
    pallet_weights, pallet_sum_xw, pallet_sum_yw,
    pallet_max_weight,
    cog_x_min, cog_x_max, cog_y_min, cog_y_max,
    cog_min_load_frac,
):
    cdef const double[::1] pw_v = pallet_weights
    cdef const double[::1] pxw_v = pallet_sum_xw
    cdef const double[::1] pyw_v = pallet_sum_yw
    return _ck_cog_envelope(
        cand_x, cand_y, cand_dx, cand_dy, cand_weight, cand_pallet,
        pw_v, pxw_v, pyw_v,
        pallet_max_weight,
        cog_x_min, cog_x_max, cog_y_min, cog_y_max,
        cog_min_load_frac,
    )


def _apply_cog_contribution_njit(
    cand_x, cand_y, cand_dx, cand_dy, cand_weight, cand_pallet,
    pallet_sum_xw, pallet_sum_yw,
):
    cdef double[::1] pxw_v = pallet_sum_xw
    cdef double[::1] pyw_v = pallet_sum_yw
    _ap_cog_contribution(
        cand_x, cand_y, cand_dx, cand_dy, cand_weight, cand_pallet,
        pxw_v, pyw_v,
    )


def _apply_load_contribution_njit(
    placements_out, dims_all, bps_order,
    placement_top_loads, n_placed, cand_pallet,
    cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_weight,
):
    cdef const i64[:, ::1] po_v = placements_out
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] bo_v = bps_order
    cdef double[::1] ptl_v = placement_top_loads
    _ap_load_contribution(
        po_v, da_v, bo_v, ptl_v, n_placed, cand_pallet,
        cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_weight,
    )
