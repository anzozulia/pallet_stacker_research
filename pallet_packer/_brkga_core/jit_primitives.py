"""
pallet_packer._brkga_core.jit_primitives — geometric scoring helpers
used by every decoder.

Each function picks the "best" EMS for a given box rotation under a
specific scoring rule:

  find_best_wall_njit      minimise X, tiebreak with YZ-DFTRC (mode 1)
  find_best_corner_njit    minimise x+y+z (mode 2 — corner-fill)
  find_best_in_slab_njit   YZ-DFTRC within an [x_min, x_max] slab
                           (mode 3 — Bischoff-Ratcliff layer-build)

The DFTRC primitive itself (`find_best_dftrc_njit`, mode 0) lives in
`brkga_v3_fast.py` and is imported by the decoders that need it.
"""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def find_best_wall_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
) -> tuple:
    """Wall-build placement: minimise X first, then maximise
    (W-y-dy)^2 + (H-z-dz)^2 on the cross-section. Produces brick-wall
    packings ideal for homogeneous loads.
    """
    best_idx = -1
    best_x = L + 1
    best_yz = -1
    bx = 0
    by = 0
    bz = 0
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
        # Floor-first: prefer lower z, then the existing wall criterion.
        if best_idx == -1 or z < bz or (z == bz and (x < best_x or (x == best_x and yz > best_yz))):
            best_x = x
            best_yz = yz
            best_idx = i
            bx, by, bz = x, y, z
    return best_idx, bx, by, bz


@njit(cache=True, fastmath=True)
def find_best_corner_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
) -> tuple:
    """Anti-DFTRC: place at smallest (x+y+z) corner. Useful as 3rd decoder.

    Promotes tightly-packed corner-fill behaviour different from both
    DFTRC and wall-build.
    """
    best_idx = -1
    best_sum = 3 * (L + W + H)
    bx = 0
    by = 0
    bz = 0
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
        # Floor-first: prefer lower z, then the existing corner (min-sum) criterion.
        if best_idx == -1 or ez_min < bz or (ez_min == bz and s < best_sum):
            best_sum = s
            best_idx = i
            bx, by, bz = ex_min, ey_min, ez_min
    return best_idx, bx, by, bz


@njit(cache=True, fastmath=True)
def find_best_in_slab_njit(
    emss: np.ndarray, n_ems: int,
    dx: int, dy: int, dz: int,
    L: int, W: int, H: int,
    slab_min_x: int, slab_max_x: int,
) -> tuple:
    """Find best DFTRC placement constrained to current slab.

    Slab spans X in [slab_min_x, slab_max_x]. Placement requires
    ex_min >= slab_min_x AND ex_min + dx <= slab_max_x. Scoring is
    DFTRC on YZ only (X is determined by slab structure).
    """
    best_idx = -1
    best_yz = -1
    bx = 0
    by = 0
    bz = 0
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
        # Floor-first: prefer lower z, then the existing slab (max-yz) criterion.
        if best_idx == -1 or z < bz or (z == bz and yz > best_yz):
            best_yz = yz
            best_idx = i
            bx, by, bz = x, y, z
    return best_idx, bx, by, bz
