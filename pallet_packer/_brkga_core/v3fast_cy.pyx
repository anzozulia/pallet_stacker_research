# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
"""
pallet_packer._brkga_core.v3fast_cy — Cython port of the brkga_v3_fast
helpers used by every decoder:

  MAX_EMS_C        cap on EMS list size (matches brkga_v3_fast.MAX_EMS)
  _find_best_dftrc DFTRC-2 scoring (mode 0)
  _commit_ems      EMS difference process (Lai-Chan 1997)

Python-callable wrappers `find_best_dftrc_njit` and `commit_ems_njit`
match the Numba signatures from brkga_v3_fast so they're drop-in.
"""
import numpy as np
cimport numpy as cnp
# i64 + MAX_EMS_C declared in v3fast_cy.pxd

# Keep in sync with brkga_v3_fast.MAX_EMS. Change both together.
MAX_EMS_C = 512

# Expose to Python land as MAX_EMS so callers see the same name.
MAX_EMS = MAX_EMS_C


# ---------------------------------------------------------------------------
# Internal nogil cores.
# ---------------------------------------------------------------------------

cdef void _find_best_dftrc(
    const i64[:, :, ::1] emss,
    i64 n_ems,
    i64 dx, i64 dy, i64 dz,
    i64 L, i64 W, i64 H,
    i64* out_idx, i64* out_x, i64* out_y, i64* out_z,
) noexcept nogil:
    """DFTRC-2: maximise squared distance from front-top-right corner."""
    cdef i64 best_idx = -1
    cdef i64 best_score = -1
    cdef i64 bx = 0, by = 0, bz = 0
    cdef i64 i, ex_min, ey_min, ez_min, ex_max, ey_max, ez_max
    cdef i64 x, y, z, d_sq
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
        d_sq = ((L - x - dx) * (L - x - dx)
                + (W - y - dy) * (W - y - dy)
                + (H - z - dz) * (H - z - dz))
        if d_sq > best_score:
            best_score = d_sq
            best_idx = i
            bx = x
            by = y
            bz = z
    out_idx[0] = best_idx
    out_x[0] = bx
    out_y[0] = by
    out_z[0] = bz


cdef i64 _commit_ems(
    const i64[:, :, ::1] emss, i64 n_ems,
    i64 bx1, i64 by1, i64 bz1,
    i64 bx2, i64 by2, i64 bz2,
    i64[:, :, ::1] out_buf,
) noexcept nogil:
    """EMS difference process (Lai-Chan 1997): replace overlapping
    EMSs with up to 6 children each, then prune dominated. Writes
    into out_buf and returns new count.
    """
    cdef i64 tmp_count = 0
    cdef i64 i, j, which
    cdef i64 ex1, ey1, ez1, ex2, ey2, ez2
    cdef i64 lo0, lo1, lo2, hi0, hi1, hi2
    cdef i64 i_lo0, i_lo1, i_lo2, i_hi0, i_hi1, i_hi2
    cdef i64 j_lo0, j_lo1, j_lo2, j_hi0, j_hi1, j_hi2
    cdef bint dominated, identical

    # Phase 1: copy non-overlapping EMSs + generate children for overlapping.
    for i in range(n_ems):
        ex1 = emss[i, 0, 0]; ey1 = emss[i, 0, 1]; ez1 = emss[i, 0, 2]
        ex2 = emss[i, 1, 0]; ey2 = emss[i, 1, 1]; ez2 = emss[i, 1, 2]
        if not (bx1 < ex2 and bx2 > ex1
                and by1 < ey2 and by2 > ey1
                and bz1 < ez2 and bz2 > ez1):
            if tmp_count < MAX_EMS_C:
                out_buf[tmp_count, 0, 0] = ex1
                out_buf[tmp_count, 0, 1] = ey1
                out_buf[tmp_count, 0, 2] = ez1
                out_buf[tmp_count, 1, 0] = ex2
                out_buf[tmp_count, 1, 1] = ey2
                out_buf[tmp_count, 1, 2] = ez2
                tmp_count += 1
            continue
        for which in range(6):
            if which == 0:
                lo0 = bx2; lo1 = ey1; lo2 = ez1
                hi0 = ex2; hi1 = ey2; hi2 = ez2
            elif which == 1:
                lo0 = ex1; lo1 = ey1; lo2 = ez1
                hi0 = bx1; hi1 = ey2; hi2 = ez2
            elif which == 2:
                lo0 = ex1; lo1 = by2; lo2 = ez1
                hi0 = ex2; hi1 = ey2; hi2 = ez2
            elif which == 3:
                lo0 = ex1; lo1 = ey1; lo2 = ez1
                hi0 = ex2; hi1 = by1; hi2 = ez2
            elif which == 4:
                lo0 = ex1; lo1 = ey1; lo2 = bz2
                hi0 = ex2; hi1 = ey2; hi2 = ez2
            else:
                lo0 = ex1; lo1 = ey1; lo2 = ez1
                hi0 = ex2; hi1 = ey2; hi2 = bz1
            if hi0 > lo0 and hi1 > lo1 and hi2 > lo2:
                if tmp_count < MAX_EMS_C:
                    out_buf[tmp_count, 0, 0] = lo0
                    out_buf[tmp_count, 0, 1] = lo1
                    out_buf[tmp_count, 0, 2] = lo2
                    out_buf[tmp_count, 1, 0] = hi0
                    out_buf[tmp_count, 1, 1] = hi1
                    out_buf[tmp_count, 1, 2] = hi2
                    tmp_count += 1

    # Phase 2: dominance pruning. Process in-place.
    cdef i64 keep_count = 0
    for i in range(tmp_count):
        i_lo0 = out_buf[i, 0, 0]; i_lo1 = out_buf[i, 0, 1]; i_lo2 = out_buf[i, 0, 2]
        i_hi0 = out_buf[i, 1, 0]; i_hi1 = out_buf[i, 1, 1]; i_hi2 = out_buf[i, 1, 2]
        dominated = False
        for j in range(tmp_count):
            if i == j:
                continue
            j_lo0 = out_buf[j, 0, 0]; j_lo1 = out_buf[j, 0, 1]; j_lo2 = out_buf[j, 0, 2]
            j_hi0 = out_buf[j, 1, 0]; j_hi1 = out_buf[j, 1, 1]; j_hi2 = out_buf[j, 1, 2]
            if (j_lo0 <= i_lo0 and j_lo1 <= i_lo1 and j_lo2 <= i_lo2
                    and i_hi0 <= j_hi0 and i_hi1 <= j_hi1 and i_hi2 <= j_hi2):
                identical = (j_lo0 == i_lo0 and j_lo1 == i_lo1 and j_lo2 == i_lo2
                             and j_hi0 == i_hi0 and j_hi1 == i_hi1 and j_hi2 == i_hi2)
                if identical:
                    if j < i:
                        dominated = True
                        break
                else:
                    dominated = True
                    break
        if not dominated:
            if keep_count != i:
                out_buf[keep_count, 0, 0] = i_lo0
                out_buf[keep_count, 0, 1] = i_lo1
                out_buf[keep_count, 0, 2] = i_lo2
                out_buf[keep_count, 1, 0] = i_hi0
                out_buf[keep_count, 1, 1] = i_hi1
                out_buf[keep_count, 1, 2] = i_hi2
            keep_count += 1
    return keep_count


# ---------------------------------------------------------------------------
# Python-callable wrappers.
# ---------------------------------------------------------------------------

def find_best_dftrc_njit(emss, n_ems, dx, dy, dz, L, W, H):
    """DFTRC-2 placement. Returns (best_idx, bx, by, bz)."""
    cdef i64 idx, bx, by, bz
    cdef const i64[:, :, ::1] emss_v = emss
    _find_best_dftrc(emss_v, n_ems, dx, dy, dz, L, W, H,
                     &idx, &bx, &by, &bz)
    return idx, bx, by, bz


def commit_ems_njit(emss, n_ems, bx1, by1, bz1, bx2, by2, bz2, out_buf):
    """EMS difference process. Mutates out_buf; returns new EMS count."""
    cdef const i64[:, :, ::1] emss_v = emss
    cdef i64[:, :, ::1] obuf_v = out_buf
    return _commit_ems(emss_v, n_ems,
                       bx1, by1, bz1, bx2, by2, bz2, obuf_v)
