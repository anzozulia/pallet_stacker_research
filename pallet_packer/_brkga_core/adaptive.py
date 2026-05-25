"""
pallet_packer._brkga_core.adaptive — experimental adaptive mode selector.

Default OFF. Probe each decoder mode on a small seed set then bias the
chromosome selector keyspace toward better-performing modes via
inverse-CDF lookup.

Empirically NEUTRAL on BR (n=5 mean Δ ≈ -0.08 pp; W/T/L = 3/12/5) —
the probe only sees performance on smart-init chromosomes and fails to
predict mode performance on the random / crossover BPS orderings that
dominate BRKGA exploration. See docs/reports/20_v39_adaptive_selector.md.

  probe_decoder_modes        decode each seed chrom under each mode →
                             per-mode mean util
  mode_weights_from_fitness  fitness → keyspace weights (floored softmax)
  mode_weights_to_cdf        cumulative distribution for inverse-CDF lookup
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..models import Box, Pallet, PackerConfig
from ..packer import PackResult
from .dispatch import decode_chromosome


def probe_decoder_modes(
    seed_chroms: List[np.ndarray],
    boxes: List[Box],
    pallet: Pallet,
    config: PackerConfig,
    n_rots_arr: np.ndarray,
    dims_all: np.ndarray,
    n_modes: int,
    *,
    max_pallets: int = 1,
    sku_id_per_box: Optional[np.ndarray] = None,
    sku_best_block: Optional[np.ndarray] = None,
    sku_top_k_blocks: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Decode each seed chromosome under each mode; return mean util per mode.

    Result shape: (n_modes,) with values in [0, 1] — higher is better.
    Empty seeds → returns uniform array (no adaptation).
    """
    if not seed_chroms or n_modes <= 0:
        return np.full(n_modes, 1.0 / max(n_modes, 1))
    cap = pallet.length * pallet.width * pallet.height
    fits = np.zeros(n_modes, dtype=np.float64)
    counts = np.zeros(n_modes, dtype=np.int64)
    for cs in seed_chroms:
        for m in range(n_modes):
            res = decode_chromosome(
                cs, boxes, pallet, config, n_rots_arr, dims_all, m,
                max_pallets=max_pallets,
                sku_id_per_box=sku_id_per_box,
                sku_best_block=sku_best_block,
                sku_top_k_blocks=sku_top_k_blocks,
            )
            if res.pallets:
                used = sum(p.box.volume for p in res.pallets[0].placements)
                util = used / cap if cap > 0 else 0.0
            else:
                util = 0.0
            fits[m] += util
            counts[m] += 1
    counts = np.maximum(counts, 1)
    return fits / counts


def mode_weights_from_fitness(
    fits: np.ndarray,
    *,
    floor: float = 0.05,
    sharpness: float = 30.0,
) -> np.ndarray:
    """Convert per-mode mean util → selector keyspace weights summing to 1.

    Softmax over (fit - max_fit) * sharpness, then floors each mode at
    `floor / n_modes` of the total to keep exploration alive. With the
    defaults (sharpness=30, floor=0.05), a typical BR per-mode spread of
    3 pp gives the best mode ~2.5× the keyspace of the worst, and every
    mode keeps at least ~1 % share.
    """
    n = len(fits)
    if n == 0:
        return np.array([], dtype=np.float64)
    # Softmax over deltas (best mode has delta=0, worse modes negative)
    deltas = fits - fits.max()
    w = np.exp(deltas * sharpness)
    w = w / w.sum()
    # Apply floor and renormalize.
    floor_amt = floor / n
    w = np.maximum(w, floor_amt)
    w = w / w.sum()
    return w


def mode_weights_to_cdf(weights: np.ndarray) -> np.ndarray:
    """Cumulative distribution from weights for inverse-CDF mode lookup."""
    return np.cumsum(weights)


# ============================================================================
# Smart initialization
# ============================================================================
