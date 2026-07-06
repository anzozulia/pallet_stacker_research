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
# i64 + int64_t come from the companion jit_constraints_cy.pxd

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
    i64 pallet_l,
    i64 pallet_w,
    const double[::1] pending_loads,
) noexcept nogil:
    """Two-pass: total contact area → support_ratio + centroid; then per-
    supporter load distribution → max_load_on_top. Floor placements (z<=0)
    return True immediately UNLESS pallet_l > 0 (the raw deck dims, passed
    only when overhang inflates the container bounds): then the box must
    rest ON the deck — deck-contact area >= the effective support ratio
    (F17) — and, when require_centroid, its footprint centroid must lie
    over the deck-contact rectangle (round 6, F30: integer-doubled compare,
    exact; centroid ON the edge accepts; mirrors the Numba reference).
    pending_loads (round 3, F20): the block sibling-column overlay — all
    zero from every non-block caller; the load compare reads
    placement_top_loads[i] + pending_loads[i].
    """
    cdef i64 x_hi0, y_hi0
    cdef double contact, eff_sr0, footprint0
    cdef double lim, eps
    if cand_z <= 0:
        if pallet_l <= 0:
            return True
        x_hi0 = cand_x + cand_dx if cand_x + cand_dx < pallet_l else pallet_l
        y_hi0 = cand_y + cand_dy if cand_y + cand_dy < pallet_w else pallet_w
        if x_hi0 <= cand_x or y_hi0 <= cand_y:
            return False
        contact = <double>((x_hi0 - cand_x) * (y_hi0 - cand_y))
        eff_sr0 = 1.0 if require_full_support != 0 else support_ratio
        if eff_sr0 > 0.0:
            footprint0 = <double>(cand_dx * cand_dy)
            if footprint0 > 0 and contact / footprint0 < eff_sr0 - 1e-6:
                return False
        # F30 (round 6): centroid must sit over the deck-contact rectangle
        # or the box tips past the deck edge. Integer-exact; boundary
        # (centroid ON the edge) accepts.
        if require_centroid != 0 and (
                2 * cand_x + cand_dx > 2 * x_hi0
                or 2 * cand_y + cand_dy > 2 * y_hi0):
            return False
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
        # Scale-aware tolerance (round 3, F25) — mirrors the Numba twin.
        lim = mlot[sup_box]
        eps = 1e-9 * lim
        if eps < 1e-6:
            eps = 1e-6
        if placement_top_loads[i] + pending_loads[i] + share > lim + eps:
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


cdef bint _ck_load_transitive(
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
    double[::1] tl_inc,
    i64[::1] tl_touched,
    const double[::1] pending_loads,
) noexcept nogil:
    """Transitive load CHECK — exact mirror of _check_load_transitive_njit:
    a dry-run of the transitive commit that rejects if any box in the
    downward chain would exceed its max_load_on_top (the direct check never
    rejects a fresh pure column — F19). Scratch all-zero in/out.
    pending_loads: the block sibling-column overlay (F20); all-zero from
    every non-block caller.
    """
    if cand_z <= 0 or cand_weight <= 0:
        return True
    cdef double total_area = 0.0
    cdef i64 i, k, sup_box, sup_rot
    cdef i64 sup_dx, sup_dy, sup_dz, sup_x, sup_y, sup_z
    cdef i64 x_lo, x_hi, y_lo, y_hi
    cdef double area, share, w, inc, tarea, lim, eps
    cdef i64 n_touched = 0, n_done = 0
    cdef i64 best_k, best_row, best_z, row_k, z_k, row_z
    cdef i64 r_box, r_rot, r_x, r_y, r_dx, r_dy, r_pallet
    cdef bint ok = True
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
        return True
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
        if share <= 0.0:
            continue
        if tl_inc[i] == 0.0:
            tl_touched[n_touched] = i
            n_touched += 1
        tl_inc[i] += share
    while n_done < n_touched:
        best_k = n_done
        best_row = tl_touched[n_done]
        best_z = placements_out[best_row, 4]
        for k in range(n_done + 1, n_touched):
            row_k = tl_touched[k]
            z_k = placements_out[row_k, 4]
            if z_k > best_z or (z_k == best_z and row_k < best_row):
                best_k = k
                best_row = row_k
                best_z = z_k
        tl_touched[best_k] = tl_touched[n_done]
        tl_touched[n_done] = best_row
        n_done += 1
        inc = tl_inc[best_row]
        lim = mlot[bps_order[best_row]]
        eps = 1e-9 * lim         # scale-aware tolerance (round 3, F25)
        if eps < 1e-6:
            eps = 1e-6
        if ok and (placement_top_loads[best_row] + pending_loads[best_row]
                   + inc > lim + eps):
            ok = False        # keep walking only to zero the scratch
        tl_inc[best_row] = 0.0
        if not ok:
            continue
        row_z = placements_out[best_row, 4]
        if row_z <= 0 or inc <= 0.0:
            continue
        r_box = bps_order[best_row]
        r_rot = placements_out[best_row, 1]
        r_x = placements_out[best_row, 2]
        r_y = placements_out[best_row, 3]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_pallet = placements_out[best_row, 0]
        tarea = 0.0
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            tarea += <double>((x_hi - x_lo) * (y_hi - y_lo))
        if tarea <= 0.0:
            continue
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            area = <double>((x_hi - x_lo) * (y_hi - y_lo))
            w = inc * (area / tarea)
            if w <= 0.0:
                continue
            if tl_inc[i] == 0.0:
                tl_touched[n_touched] = i
                n_touched += 1
            tl_inc[i] += w
    return ok


cdef void _ap_load_contribution_transitive(
    const i64[:, ::1] placements_out,
    const i64[:, :, ::1] dims_all,
    const i64[::1] bps_order,
    double[::1] placement_top_loads,
    i64 n_placed,
    i64 cand_pallet,
    i64 cand_x, i64 cand_y, i64 cand_z,
    i64 cand_dx, i64 cand_dy, i64 cand_dz,
    double cand_weight,
    double[::1] tl_inc,
    i64[::1] tl_touched,
) noexcept nogil:
    """Transitive load commit (PackerConfig.transitive_load_bearing) —
    exact mirror of _apply_load_contribution_transitive_njit: weight flows
    through direct supporters and down every chain to the floor, split by
    contact-area fraction per hop. Rows processed highest bottom-z first,
    ties by lowest row index (the twin bit-identity contract). tl_inc /
    tl_touched are caller-owned scratch; tl_inc all-zero on entry and exit.
    """
    if cand_z <= 0 or cand_weight <= 0:
        return
    cdef double total_area = 0.0
    cdef i64 i, k, sup_box, sup_rot
    cdef i64 sup_dx, sup_dy, sup_dz, sup_x, sup_y, sup_z
    cdef i64 x_lo, x_hi, y_lo, y_hi
    cdef double area, share, w, inc, tarea
    cdef i64 n_touched = 0, n_done = 0
    cdef i64 best_k, best_row, best_z, row_k, z_k, row_z
    cdef i64 r_box, r_rot, r_x, r_y, r_dx, r_dy, r_pallet
    # Seed pass 1: total direct-supporter contact area.
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
    # Seed pass 2: credit each direct supporter and queue it.
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
        if share <= 0.0:
            continue
        placement_top_loads[i] += share
        if tl_inc[i] == 0.0:
            tl_touched[n_touched] = i
            n_touched += 1
        tl_inc[i] += share
    # Worklist: max bottom-z first, ties lowest row index — single-visit.
    while n_done < n_touched:
        best_k = n_done
        best_row = tl_touched[n_done]
        best_z = placements_out[best_row, 4]
        for k in range(n_done + 1, n_touched):
            row_k = tl_touched[k]
            z_k = placements_out[row_k, 4]
            if z_k > best_z or (z_k == best_z and row_k < best_row):
                best_k = k
                best_row = row_k
                best_z = z_k
        tl_touched[best_k] = tl_touched[n_done]
        tl_touched[n_done] = best_row
        n_done += 1
        inc = tl_inc[best_row]
        tl_inc[best_row] = 0.0
        row_z = placements_out[best_row, 4]
        if row_z <= 0 or inc <= 0.0:
            continue
        r_box = bps_order[best_row]
        r_rot = placements_out[best_row, 1]
        r_x = placements_out[best_row, 2]
        r_y = placements_out[best_row, 3]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_pallet = placements_out[best_row, 0]
        tarea = 0.0
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            tarea += <double>((x_hi - x_lo) * (y_hi - y_lo))
        if tarea <= 0.0:
            continue
        for i in range(n_placed):
            if placements_out[i, 5] == 0:
                continue
            if placements_out[i, 0] != r_pallet:
                continue
            sup_box = bps_order[i]
            sup_rot = placements_out[i, 1]
            sup_dz = dims_all[sup_box, sup_rot, 2]
            sup_z = placements_out[i, 4]
            if sup_z + sup_dz != row_z:
                continue
            sup_dx = dims_all[sup_box, sup_rot, 0]
            sup_dy = dims_all[sup_box, sup_rot, 1]
            sup_x = placements_out[i, 2]
            sup_y = placements_out[i, 3]
            x_lo = r_x if r_x > sup_x else sup_x
            x_hi = r_x + r_dx if r_x + r_dx < sup_x + sup_dx else sup_x + sup_dx
            if x_hi <= x_lo:
                continue
            y_lo = r_y if r_y > sup_y else sup_y
            y_hi = r_y + r_dy if r_y + r_dy < sup_y + sup_dy else sup_y + sup_dy
            if y_hi <= y_lo:
                continue
            area = <double>((x_hi - x_lo) * (y_hi - y_lo))
            w = inc * (area / tarea)
            if w <= 0.0:
                continue
            placement_top_loads[i] += w
            if tl_inc[i] == 0.0:
                tl_touched[n_touched] = i
                n_touched += 1
            tl_inc[i] += w


cdef double _rider_inflow(
    const i64[:, ::1] placements_out,
    const i64[:, :, ::1] dims_all,
    const i64[::1] bps_order,
    const double[::1] weights,
    const double[::1] placement_top_loads,
    i64 n_placed,
    i64 cand_pallet,
    i64 cand_x, i64 cand_y, i64 cand_z,
    i64 cand_dx, i64 cand_dy, i64 cand_dz,
    int transitive,
) noexcept nogil:
    """Load the candidate would INHERIT by becoming a NEW supporter of
    already-placed boxes (round 3, F21 "under-fill") — exact mirror of
    _rider_inflow_njit. share = out_R * a / (T_old + a) per rider; old
    supporters not debited (conservative). Zero-weight riders contribute
    nothing.
    """
    cdef i64 cand_top = cand_z + cand_dz
    cdef double inherited = 0.0
    cdef i64 i, j, r_box, r_rot, r_dx, r_dy, r_x, r_y
    cdef i64 s_box, s_rot, s_dx, s_dy, s_dz, s_x, s_y, s_z
    cdef i64 x_lo, x_hi, y_lo, y_hi, sx_lo, sx_hi, sy_lo, sy_hi
    cdef double a, out_r, t_old
    for i in range(n_placed):
        if placements_out[i, 5] == 0:
            continue
        if placements_out[i, 0] != cand_pallet:
            continue
        if placements_out[i, 4] != cand_top:
            continue
        r_box = bps_order[i]
        r_rot = placements_out[i, 1]
        r_dx = dims_all[r_box, r_rot, 0]
        r_dy = dims_all[r_box, r_rot, 1]
        r_x = placements_out[i, 2]
        r_y = placements_out[i, 3]
        x_lo = cand_x if cand_x > r_x else r_x
        x_hi = cand_x + cand_dx if cand_x + cand_dx < r_x + r_dx else r_x + r_dx
        if x_hi <= x_lo:
            continue
        y_lo = cand_y if cand_y > r_y else r_y
        y_hi = cand_y + cand_dy if cand_y + cand_dy < r_y + r_dy else r_y + r_dy
        if y_hi <= y_lo:
            continue
        a = <double>((x_hi - x_lo) * (y_hi - y_lo))
        out_r = weights[r_box]
        if transitive != 0:
            out_r += placement_top_loads[i]
        if out_r <= 0.0:
            continue
        t_old = 0.0
        for j in range(n_placed):
            if j == i:
                continue
            if placements_out[j, 5] == 0:
                continue
            if placements_out[j, 0] != cand_pallet:
                continue
            s_box = bps_order[j]
            s_rot = placements_out[j, 1]
            s_dz = dims_all[s_box, s_rot, 2]
            s_z = placements_out[j, 4]
            if s_z + s_dz != cand_top:
                continue
            s_dx = dims_all[s_box, s_rot, 0]
            s_dy = dims_all[s_box, s_rot, 1]
            s_x = placements_out[j, 2]
            s_y = placements_out[j, 3]
            sx_lo = r_x if r_x > s_x else s_x
            sx_hi = r_x + r_dx if r_x + r_dx < s_x + s_dx else s_x + s_dx
            if sx_hi <= sx_lo:
                continue
            sy_lo = r_y if r_y > s_y else s_y
            sy_hi = r_y + r_dy if r_y + r_dy < s_y + s_dy else s_y + s_dy
            if sy_hi <= sy_lo:
                continue
            t_old += <double>((sx_hi - sx_lo) * (sy_hi - sy_lo))
        inherited += out_r * (a / (t_old + a))
    return inherited


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
    require_centroid,
    require_full_support,
    pallet_l,
    pallet_w,
    pending_loads,
):
    cdef const i64[:, ::1] po_v = placements_out
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] bo_v = bps_order
    cdef const double[::1] mlot_v = mlot
    cdef const double[::1] ptl_v = placement_top_loads
    cdef const double[::1] pl_v = pending_loads
    return _ck_load_on_top(
        po_v, da_v, bo_v, mlot_v, ptl_v, n_placed,
        cand_pallet, cand_x, cand_y, cand_z,
        cand_dx, cand_dy, cand_dz,
        cand_weight, support_ratio,
        require_centroid, require_full_support,
        pallet_l, pallet_w, pl_v,
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


def _apply_load_contribution_transitive_njit(
    placements_out, dims_all, bps_order,
    placement_top_loads, n_placed, cand_pallet,
    cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_weight,
    tl_inc, tl_touched,
):
    cdef const i64[:, ::1] po_v = placements_out
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] bo_v = bps_order
    cdef double[::1] ptl_v = placement_top_loads
    cdef double[::1] ti_v = tl_inc
    cdef i64[::1] tt_v = tl_touched
    _ap_load_contribution_transitive(
        po_v, da_v, bo_v, ptl_v, n_placed, cand_pallet,
        cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, cand_weight,
        ti_v, tt_v,
    )


def _rider_inflow_njit(
    placements_out, dims_all, bps_order, weights,
    placement_top_loads, n_placed, cand_pallet,
    cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, transitive,
):
    cdef const i64[:, ::1] po_v = placements_out
    cdef const i64[:, :, ::1] da_v = dims_all
    cdef const i64[::1] bo_v = bps_order
    cdef const double[::1] w_v = weights
    cdef const double[::1] ptl_v = placement_top_loads
    return _rider_inflow(
        po_v, da_v, bo_v, w_v, ptl_v, n_placed, cand_pallet,
        cand_x, cand_y, cand_z, cand_dx, cand_dy, cand_dz, transitive,
    )
