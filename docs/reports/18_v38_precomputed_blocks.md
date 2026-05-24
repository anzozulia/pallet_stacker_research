# v3.8 — Pre-computed Blocks (Experimental, Mode 5)

Added mode 5 = DFTRC + pre-computed best block per SKU (Bischoff 2002 style).
Pre-pass enumerates the SINGLE BEST (k, l, m, rotation) block per SKU.
Decoder tries to place this full block at DFTRC for block-dimensions;
falls back to mode 4 dynamic if it doesn't fit.

## Single-decode probe (vol-desc seed, BR1#1)

| Mode | Util | Notes |
|------|------|-------|
| 0 DFTRC | 84.91% | |
| 2 corner | 88.22% | strongest deterministic |
| 4 dynamic blocks | 86.98% | |
| **5 pre-computed blocks** | **74.28%** | too greedy |

The best pre-computed block for BR1 SKU 0 is **4×2×5 = 40 boxes**. Placing
that monolithic block first consumes most of the pallet, leaving little
room for the other 2 SKUs. Net util drops.

## Status

- Mode 5 implemented and integrated (n_modes=6 default)
- `enumerate_best_block_per_sku` Python helper
- `decode_precomputed_blocks_njit_mode` JIT decoder
- BRKGA selector includes mode 5 as one of 6 modes
- Single-decode probe shows mode 5 alone is weak

## Honest assessment

Pre-computed blocks as designed (TOP-1 max-volume block) is **too greedy**
for BR1 — same failure mode as the earlier "block-aware DFTRC" experiment.
The lesson: local optimization (place the biggest possible block) hurts
global packing structure.

What would actually work (and was discussed but not implemented):
1. **Top-K blocks per SKU** with chromosome encoding which one to use
   (gives BRKGA the power to pick smaller blocks when beneficial)
2. **Block size GUIDED by remaining pallet space** rather than fixed
   per SKU
3. **Cascading block sizes**: if 4×2×5 doesn't fit, try 4×2×4, then 4×2×3,
   etc., before falling back to single boxes

These are 1-2 days of additional work each. Given v3.6 already beats SOTA
on BR3/5/7, the marginal value of pushing BR1 the last 3-4pp is research
completeness, not deployment-critical.

## Recommendation

v3.6 (with dynamic blocks mode 4) remains the production recommendation.
Mode 5 is wired in (n_modes=6) but likely won't help much — BRKGA's
selector will route most chromosomes to modes 0-4.

Closing the BR1 gap further would require either:
- Top-K pre-computed blocks with proper chromosome encoding (the actual
  Bischoff 2002 recipe), or
- C/Cython decoder for 10-100× more compute headroom
