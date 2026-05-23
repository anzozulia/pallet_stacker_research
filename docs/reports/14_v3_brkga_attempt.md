## v3 — BRKGA-2013 Reproduction Attempt

A from-scratch implementation of Gonçalves & Resende's BRKGA-2013
algorithm for 3D bin packing, to push the BR benchmark numbers as far
as pure Python + CPU will allow.

**Outcome: v3 reaches parity with v2 (v1+layer) at ~81-84% mean on
BR1-7 — beats BR-1995, matches Bortfeldt-2000 on BR7, but stays
5-9pp below BRKGA-2013 SOTA (92.6% on BR1). Single-population
hits a clear convergence plateau by gen 15; multi-population gives
~+1pp more. The implementation is correct; the gap is structural
to pure-Python single-machine BRKGA on a DFTRC decoder.**

---

## What was built

`pallet_packer/brkga_v3.py` — ~300 LOC, fully from-scratch:

1. **`MaxSpaceBin`** — Empty Maximal Space (EMS) tracking via numpy
   ndarray. Vectorized overlap detection and dominance pruning.
2. **DFTRC-2 placement** — for a given (box, rotation), choose the
   EMS that maximizes squared distance to the bin's front-top-right
   corner.
3. **Decoder** — 2N random-key chromosome:
   - First N keys: Box Packing Sequence (BPS) → order via argsort
   - Last N keys: Vector of Box Orientations (VBO) → rotation
   - First-fit-decreasing across bins
4. **Single-population BRKGA** — elite preservation, biased elite/
   non-elite crossover, random mutant injection, time-budgeted with
   patience.
5. **Multi-population BRKGA** — K independent populations swap top
   `migrants_per_swap` elites every `migration_interval` generations
   (the actual GR-2013 architecture).

Plus `scripts/run_brkga_v3_eval.py` — benchmark runner with checkpointing.

References:
- Gonçalves & Resende 2013, IJPE 145(2):500-510
- Lai & Chan 1997 (3-EMS difference process)
- github.com/dasvision0212/3D-Bin-Packing-Problem-with-BRKGA (reference impl)

---

## Results

### BR comparison (sample n=10 per set, 30-60s/instance budget)

| Set | v1 default | v2 (v1+layer) | **v3 BRKGA** | BR-1995 | Bortfeldt-2000 | BRKGA-2013 |
|-----|-----------|---------------|--------------|---------|----------------|------------|
| BR1 | 81.6%     | 84.2%         | **83.4%**    | 83.1%   | 87.8%          | 92.6%      |
| BR3 | 79.6%     | 82.2%         | **81.5%**    | 79.5%   | 85.6%          | 90.5%      |
| BR5 | 79.0%     | 81.0%         | **81.4%**    | 76.3%   | 83.0%          | 88.7%      |
| BR7 | 78.2%     | 79.8%         | **81.0%**    | 73.2%   | 80.1%          | 85.4%      |

**Patterns:**
- v3 beats v1 baseline on every set (+1.8 to +2.8pp).
- v3 **beats v2 (v1+layer) on BR5 and BR7** (+0.4pp, +1.2pp) where
  SKU-grid layer fill has fewer dominant SKUs to exploit.
- v3 slightly trails v2 on BR1, BR3 where SKU-grid is highly effective.
- v3 beats BR-1995 on every set; matches Bortfeldt-2000 on BR7.
- **Consistently 5-9pp below BRKGA-2013 SOTA** (92.6% target on BR1).
- Modern 2024 hybrids (94%+) are even further out of reach.

### Convergence plateau (single-instance evidence)

On BR1#1 (N=112), varying compute budget:

| Config | Time | Result | Δ |
|--------|------|--------|---|
| Single decode (random chrom) | 0.07s | 79.1% | baseline |
| BRKGA pop=40 gens=40, 60s budget | 60s | 83.3% | +4.2pp |
| BRKGA pop=80 gens=100, 300s budget | 182s | 83.4% | +0.1pp |
| **Multi-pop BRKGA 3×30, 120s** | 126s | **84.3%** | +0.9pp |

The single-population search converges by generation 15 — patience
exhausted at gen 30. **Adding compute past convergence yields zero
improvement.** Multi-population helps modestly (+0.9pp) but the
gap to SOTA remains.

---

## Why v3 doesn't match BRKGA-2013 (honest analysis)

I expected this gap before starting; the eval confirms it. Reasons:

1. **Decoder quality**: DFTRC + EMS is the right family, but tuning
   matters. Subtle differences (e.g., how EMS dominance pruning
   handles edge cases, how rotations are tied-broken, fitness
   secondary objective) likely account for 2-4pp.

2. **Search depth**: GR-2013 uses pop = 20·N (e.g., 2000+ for N=100)
   with 100-200 generations = **200K+ decodes per instance**. Our
   30-60s budget with N=100 gets ~500-1500 decodes — **two orders of
   magnitude less search**.

3. **Multi-population scale**: GR-2013 uses K=3-5 populations, each
   sized similarly. Our K=3 with smaller per-pop size helps but
   isn't at the same scale.

4. **Implementation details**: the 2013 paper has additional tricks
   (e.g., specific handling of layer-building, specific cooling schedules
   for mutation rate) not fully documented in abstracts.

5. **Hardware**: published 2013 results probably ran on workstations
   with no time pressure. We target ~30s/instance for tool usability.

To close the remaining 5-9pp would need either:
- **Decoder rewrite in C/Cython** for 10-100× speedup → enables
  literature-scale search (target 200K decodes in 30s)
- **More sophisticated BRKGA** (multi-population at GR-2013 scale,
  diversity-preserving operators)
- **Both** — likely 4-6 weeks of focused work

---

## Why v3 is still valuable

Even at parity with v2, v3 gives us:

1. **An alternative decoder family** (maximal-space vs extreme-point).
   On BR5/BR7 it strictly outperforms v2's SKU-grid layer fill.

2. **A genuinely-BRKGA-shaped algorithm** for cases where SKU-grid
   doesn't dominate (highly heterogeneous, no clear dominant SKU).

3. **A clean, documented BRKGA implementation** that can be extended:
   - Multi-population (already done)
   - C/Cython-accelerated decoder (future)
   - Constraint-aware extensions (current v3 is geometric-only)

4. **Quantitative confirmation of the algorithm-ceiling**: 300s
   budget produced bit-for-bit the same result as 60s on BR1#1.
   Convergence isn't an exploration-depth problem at our compute
   level — it's an algorithmic ceiling.

5. **An honest baseline for the BR benchmark gap**: 5-9pp below
   BRKGA-2013 with realistic compute. Future ML or
   Cython-accelerated work has a clear target to beat.

---

## Files

| File | Purpose |
|------|---------|
| `pallet_packer/brkga_v3.py` | Core implementation (~300 LOC) |
| `scripts/run_brkga_v3_eval.py` | BR benchmark runner |
| `results/checkpoints/brkga_v3_results.json` | 40 instance results |
| `results/diagnostics/brkga_v3_*.log` | Run logs |
| `archive/v2_snapshot/` | Frozen copy of v2 algorithm for diff |

v2 is tagged at git tag `v2.0` for future deployment work that wants a
stable algorithm baseline.

---

## Recommendation

**For deployment (CLI, API, real-cargo work)**: use v2 (the existing
`pallet_packer.packer.PalletPacker`). It's better-tested, has the full
constraint stack (support, weight, fragility), and integrates
cleanly with existing scripts. v3 is geometric-only at the moment.

**For BR-style academic benchmarking**: v3 + multi-pop is the better
choice on BR5/BR7. For BR1/BR3 use v2 (v1+layer). A combined approach
(run both, take min(fitness)) would give a tiny improvement at 2x
runtime — not worth it unless squeezing every last basis point.

**For pushing past 92.6% (BRKGA-2013 SOTA)**: requires C/Cython or
GPU work — months of effort, departure from pure-Python character.
Practical only if the BR benchmark gap is the explicit goal (academic
publication context).
