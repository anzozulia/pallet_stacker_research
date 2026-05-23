# v3.5 — Hybrid BRKGA: The Breakthrough

A multi-decoder hybrid BRKGA that combines DFTRC, wall-build, corner-fill,
and Bischoff-Ratcliff layer-building with a v2 warm-start, position-based
local search, and path relinking. Built on top of v3-fast's Numba JIT core.

**Final outcome**: v3.5 **beats Bortfeldt-2000 on every BR set**,
**exceeds BRKGA-2013 SOTA on BR7**, and closes the gap to 1-3pp on the
remaining sets. All in 30s/instance compute.

---

## Headline numbers (n=10 per set, 30s/instance)

| Set | v1   | v2   | v3-fast | **v3.5** | BR-95 | Bortfeldt | BRKGA-2013 SOTA |
|-----|------|------|---------|----------|-------|-----------|-----------------|
| BR1 | 81.6 | 84.2 | 85.2    | **89.08** | 83.1  | 87.8      | 92.6            |
| BR3 | 79.6 | 82.2 | 83.0    | **87.89** | 79.5  | 85.6      | 90.5            |
| BR5 | 79.0 | 81.0 | 82.9    | **87.90** | 76.3  | 83.0      | 88.7            |
| BR7 | 78.2 | 79.8 | 82.8    | **86.70** | 73.2  | 80.1      | 85.4            |

v3.5 gains over v3-fast: **+3.9-5.1pp on every set**. The gap to
BRKGA-2013 SOTA shrinks from 4-7pp to 0.8-3.5pp. On BR7 v3.5 EXCEEDS
the SOTA by +1.3pp.

| Set | v3-fast | v3.5 | Δ vs v3-fast | Gap to SOTA |
|-----|---------|------|--------------|-------------|
| BR1 | 85.2    | 89.08 | +3.88pp     | -3.52pp     |
| BR3 | 83.0    | 87.89 | +4.89pp     | -2.61pp     |
| BR5 | 82.9    | 87.90 | +5.00pp     | -0.80pp     |
| BR7 | 82.8    | 86.70 | +3.90pp     | **+1.30pp** (beats SOTA) |

---

## What changed from v3-fast → v3.5

v3-fast was: pure BRKGA with EMS-DFTRC decoder + multi-population + JIT
JIT-compiled core. Beat v2 by 0.8-3.0pp on BR sets.

v3.5 keeps the JIT core but adds 6 quality improvements:

### 1. Multi-decoder chromosome (4 modes)
The chromosome's last key picks one of 4 placement modes:
- **Mode 0 (DFTRC)**: classic Distance-to-Front-Top-Right-Corner
- **Mode 1 (wall-build)**: prefer smallest X, DFTRC tiebreaker on YZ
- **Mode 2 (corner-fill)**: minimize x+y+z (tight corner packing)
- **Mode 3 (layer-build)**: Bischoff-Ratcliff slabs along X with seed
  selection and DFTRC within slab

Different modes excel on different instance types. Random chromosomes
score (avg of 10 random on BR1#1): DFTRC=78.8%, wall=79.1%, corner=81.4%,
layer=64.8%. With informed seeds, layer can hit 86%.

### 2. Smart initial population
5 informed orderings × 4 modes = 20 informed chromosomes per run:
- volume descending
- volume ascending
- max-dimension descending
- min-dimension descending
- SKU-grouped (same SKU adjacent, larger groups first)

These give BRKGA a strong starting point instead of pure random init.

### 3. v2 warm-start
Run v2 PalletPacker.pack() once (~1-2s) to get a layer-build ordering.
Convert its BPS order + rotation choices into a chromosome (one per mode).
This seed alone, decoded by v3.5's JIT engine, often scores 5-8pp higher
than v2's own engine produces.

### 4. Position-based local search
Instead of swapping raw chromosome KEY VALUES (which often produces
identical sorts), operate on BPS POSITIONS directly. Operators:
- Position swap (30%)
- Segment reverse, k=2..7 (25%)
- Insert (20%)
- Rotation flip (~20%)
- Decoder mode flip (~5%)

Every move actually changes the decoded packing.

### 5. Path relinking between top-2 elites
After BRKGA converges, take the two distinct best chromosomes and walk
from one to the other in batched steps, evaluating each intermediate.
Often finds new optima not in either elite.

### 6. v2 hybrid polish
Return max(v3.5-BRKGA, v2-actual). Safety net for cases where v2
happens to outperform.

Plus a `n_restarts > 1` option for variance reduction (split budget
K-ways with different seeds, take best).

---

## Why this works

The biggest single improvement was **smart init + v2 warm-start**.
The first decoded chromosome in pop 0 often hits ~87% util (vs ~79%
random), giving BRKGA a strong starting point and freeing it to focus
on exploration around already-promising chromosomes rather than
discovering basic structure.

The multi-decoder helps in two ways:
1. Direct: corner-fill (mode 2) is often the strongest single decoder
   for BR-style problems.
2. Indirect: each mode explores a different basin of the search space.
   When one converges, others continue exploring.

The wall-build fix (sentinel bug in v3.5.0) is critical — the original
multi-decoder essentially ran with only mode 0 + mode 2 effective, since
mode 1 placed only 11/112 boxes due to a wrong scoring threshold.

---

## Files

| File | Purpose |
|------|---------|
| `pallet_packer/brkga_v3_5.py` | v3.5 implementation (~1000 LOC) |
| `pallet_packer/brkga_v3_fast.py` | v3-fast core JIT functions (reused) |
| `scripts/run_brkga_v35_eval.py` | BR benchmark runner |
| `scripts/run_brkga_v35_new_eval.py` | Same with NEW code path |
| `results/checkpoints/brkga_v35_results.json` | Original v3.5 results |
| `results/checkpoints/brkga_v35_new_results.json` | v3.5 + wall fix + layer |
| `results/diagnostics/brkga_v35_*.log` | Run logs |

---

## Recommendation

**For deployment**: v3.5 is the new best algorithm for pallet packing.
It's pure-Python (numba-accelerated) and runs in 30s/instance for
realistic problem sizes (N=100-200). The v2 code path remains available
as the trusted production reference with the full constraint stack
(support, weight, fragility), but v3.5 offers significantly better
geometric utilization.

**For closing the BR1 gap (~3pp to BRKGA-2013 SOTA 92.6%)**: would
require structural changes, not just more compute:

1. **Multi-restart**: tested empirically (n_restarts=2, 60s/inst on BR1
   n=10). Result: +0.11pp mean (89.08 → 89.19) at 2× cost. Only 3 of 10
   instances benefited. **Not worth the compute cost** — the BRKGA
   converges to the same local optima from the v2-seeded init regardless
   of additional restart seeds. See
   `results/diagnostics/brkga_v35_multirestart_test.log`.

2. **True SKU-aware encoding**: chromosome size O(S) where S = #SKUs (3-7
   for BR1). Would dramatically reduce the search space for homogeneous
   loads. Untested.

3. **Adaptive operator selection**: tabu search, GLS, ALNS variants. None
   tested but well-documented in the literature.

4. **GPU/C extension**: would enable literature-scale population
   (pop = 20·N) within the same wall-clock budget.

All four are research-grade efforts. None required for deployment quality.

## Empirical findings from this iteration

- **Wall-build sentinel bug**: the original v3.5 had `best_minimise = 10
  * (L + W + H)` which was much smaller than wall mode's actual score
  range. For any x≥2, no placement was accepted, so wall mode placed only
  11/112 boxes on BR1. Fix (1<<62 sentinel) restored wall to ~88% util.

- **Per-decoder strength on BR1#1 (vol-desc chromosome)**:
  DFTRC 84.91%, wall 87.82%, **corner 88.22%** (best single), layer 85.86%.
  Corner-fill is surprisingly the strongest deterministic decoder.

- **Layer-build characteristic**: highly seed-dependent. With informed
  seeds (vol-desc): 85.86%. With random chromosomes: 64.75% mean.
  Useful as one mode of multi-decoder but weak alone.

- **Smart init dominates BRKGA gains**: The first decoded chromosome
  (using v2's ordering + smart sort) typically scores 86-88%. BRKGA only
  adds 2-4pp over this baseline. Most of v3.5's improvement over
  v3-fast comes from the seed quality, not the genetic search.

- **Path relinking marginal**: contributes <0.5pp in our setup. Worth
  keeping for cases where it helps but not a major lever.

- **Most instances converge to same optimum**: in OLD vs NEW comparison,
  7 of 10 BR1 instances had identical results. Variance is concentrated
  on hard instances (e.g., BR1#3 outlier).
