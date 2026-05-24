# v3.7 — SKU-Aware Encoding (Negative Result)

Implemented SKU-aware chromosome encoding (2S+1 dims instead of 2N+1)
based on the hypothesis that BR1's gap to SOTA (3.75pp) was due to BRKGA
wasting compute on irrelevant box-level dimensions. Empirical result:
**doesn't help; default OFF.**

## Empirical results (n=3 per set, 30s/instance)

| Set | v3.6 | v3.7 (with SKU-aware, 5s budget) | Δ |
|-----|------|----------------------------------|-----|
| BR1 | 88.80% | 88.70% | -0.10pp |
| BR3 | 92.27% | 92.34% | +0.07pp |
| BR5 | 91.77% | 92.47% | +0.70pp |
| BR7 | 92.33% | 91.74% | -0.59pp |

Mean Δ ≈ 0pp. Mixed results within noise.

## What was built

1. `compute_sku_groups`: groups boxes by SKU
2. `sku_chrom_to_full_chrom`: translates 2S+1 SKU chromosome → 2N+1 full
3. `brkga_sku_aware_search`: mini-BRKGA over 2S+1 chromosome
4. Phase 0b integration: runs before main BRKGA, seeds best chromosome

## Why it didn't help

**Hypothesis was wrong.** SKU-aware encoding doesn't reduce search space;
it RESTRICTS it. The encoding forces SKU-grouped BPS (all SKU0 boxes
consecutive, then all SKU1, etc.). But BR1's optimal packing isn't strictly
SKU-grouped — interleaving SKUs sometimes packs tighter.

Two factors:
1. **Restricted expressiveness**: SKU-aware mini-BRKGA (standalone) tops
   out at 82-87% on BR1 vs 89% for box-level. The smaller chromosome
   can't reach all of mode 4's solution space.
2. **Budget trade-off**: SKU-aware steals 5s from main BRKGA (24s → 19s).
   For BR7, this 5s loss isn't compensated.

## The "compute trade-off" insight

For SKU-aware to help, the marginal benefit of its chromosome reduction
must exceed the marginal cost of stolen main-BRKGA time. Our test shows
this isn't true at 30s total budget. Possible regime where it helps:
- **Very tight budgets** (<10s): SKU-aware's faster convergence might win
- **Very homogeneous problems** (S=1-2 SKUs): trivially fast SKU-aware
- **Multiprocessing** (run in parallel, no budget steal): could shift
  trade-off to favor SKU-aware

None of these were tested.

## What's next

The remaining BR1 gap (3.75pp to SOTA 92.6%) likely needs:

1. **Pre-computed block enumeration** (Bischoff 2002): generate top-K
   candidate blocks per SKU upfront, treat as virtual super-items in BRKGA.
   Structurally different from dynamic blocks (mode 4); doesn't restrict
   search like SKU-aware does. Highest leverage remaining.

2. **C/Cython decoder**: 10-100× speedup → enables 200K+ decodes in 30s
   (vs current 18K). Would let BRKGA actually explore at literature scale.

3. **Accept v3.6 as the production state**: it already beats SOTA on
   BR3/5/7 and is within 4pp on BR1. For most real-world use cases this
   is excellent.

## Status

SKU-aware code retained in `brkga_v3_5.py` as `brkga_sku_aware_search`
and `use_sku_aware` flag (default `False`). Can be enabled for
experimentation. Not recommended for production.
