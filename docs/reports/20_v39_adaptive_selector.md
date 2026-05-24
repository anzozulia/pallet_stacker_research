# v3.9 — Adaptive Mode Selector (Negative Result)

Implemented an adaptive decoder-mode selector that probes each of the
six decoder modes on a small seed set, then biases the chromosome
selector keyspace via inverse-CDF lookup so better-performing modes
get more keyspace. **Empirical result: neutral-to-slightly-negative;
default OFF.**

## TL;DR

| Set | v3.8 (n=5) | v3.9 (n=5) | Δ | W/T/L |
|---|---|---|---|---|
| BR1 | 90.41 ± 3.02 | 90.52 ± 2.85 | +0.11 pp | 1/4/0 |
| BR3 | 93.97 ± 1.03 | 93.60 ± 0.93 | -0.37 pp | 0/3/2 |
| BR5 | 92.68 ± 0.85 | 92.87 ± 0.98 | +0.19 pp | 1/3/1 |
| BR7 | 92.57 ± 0.75 | 92.32 ± 1.15 | -0.25 pp | 1/2/2 |

Mean Δ ≈ -0.08 pp. W/T/L = 3/12/5 across 20 instances. Variance went
down on BR1/3 (good), up on BR7 (bad). Net: within run-to-run noise.

## What was built

### 1. Probe phase (`probe_decoder_modes`)
Decode each of N seed chromosomes (from the smart-init / v2-seed
pool at the front of pop[0]) under each of the n_modes decoders.
Return per-mode mean util. Cost: ~6×N quick decodes, well under 1 s
for typical N=100-200 boxes and 4 seeds.

### 2. Weight derivation (`mode_weights_from_fitness`)
Softmax over `(util - max_util) * sharpness` with sharpness=30, then
floor each mode at `0.05 / n_modes` of total share to preserve
exploration. On a typical 3 pp spread, the best mode gets ~2.5× the
keyspace of the worst.

### 3. CDF dispatch (`_selector_to_mode` + `mode_cdf` in `decode_auto_mode`)
Inverse-CDF lookup remaps the uniform selector key into a non-uniform
mode distribution. CDF is computed once after the probe and held
constant for the rest of the run, so all chromosomes (initial, mutant,
crossover, post-BRKGA polish) decode under one consistent mapping.

### 4. Plumbing
`mode_cdf` threaded through `decode_auto_mode`, `local_search_2opt`,
`path_relinking`, and `lns_polish` so the polish phase uses the same
mapping the BRKGA loop used.

### 5. Driver wiring (`brkga_pack_v35`)
New flag `use_adaptive_mode_selector` (default `False` after the
negative result). When enabled, Phase 1c runs the probe right after
smart-init / v2-seed populations are seeded and stores the CDF in a
closure-captured state dict that the decoder lambda reads.

## Why it didn't help

The hypothesis was: BRKGA's selector picks weak modes too often, so an
upfront probe + bias should improve mean fitness. Two failure modes
showed up empirically:

1. **The probe sees only smart-init chromosomes.** Those chromosomes
   have very specific BPS orderings (vol-desc, SKU-grouped, v2-derived).
   A mode that performs poorly on those orderings may perform well on
   the random / crossover BPS orderings that dominate BRKGA's
   exploration after generation 1. By down-weighting such a mode upfront
   we starve BRKGA of a useful exploration basin.

   The verbose log on BR1 # 1 showed this exact pattern: mode 5 (top-K
   blocks) scored 80.6 % on smart-init probes — the worst of the six —
   and got down-weighted to 4 % keyspace. But on BR3, BR5 with random
   BPS orderings mode 5 is competitive. Down-weighting it cost ~0.4 pp.

2. **BRKGA already biases toward good modes implicitly.** The 20 %
   elite are selected by fitness, and crossover inherits 70 % of keys
   (including the selector) from the elite parent. After a few
   generations the selector keyspace is already dominated by whatever
   modes produce wins. An adaptive remap on top of this is redundant
   at best, fighting natural selection at worst.

This is structurally similar to the v3.7 SKU-aware finding: a
plausible-sounding restriction of the search space turns out to be
neutral-to-negative because BRKGA's elite-inheritance mechanism
already handles the underlying optimization.

## What did work (in tune-up cost)

- The probe overhead is real but tiny: ~0.01 s on a 30 s budget.
  Adaptive-OFF and adaptive-ON have effectively identical wall time.
- Mode 5 down-weighting on BR1 was correct — the top-K mode does
  underperform on BR1's specific characteristics. Just not in a way
  the probe-then-fix mechanism can exploit.

## What could conceivably make it work (not pursued)

- **Mid-run re-probe**: recompute mode weights every M generations on
  current-elite chromosomes. Might track BRKGA's basin shifts. But
  introduces non-stationary fitness landscape, complicating
  convergence detection.
- **Per-mode probe with random chromosomes**: probe each mode against
  pure-random BPS chromosomes rather than smart-init. More
  representative of BRKGA's search space, less biased toward the
  v2-seed basin.
- **ALNS-style per-improvement crediting**: track which mode produced
  each new elite, increment that mode's weight. Standard adaptive-LNS
  pattern. ~3-4 days of work for what would likely be a similar
  ±0.5 pp result.

None of these were tested. They each have non-trivial risk of also
landing in the noise band given the BRKGA elite-inheritance already
handles the basic problem.

## Status

Adaptive selector code retained in `brkga_v3_5.py`:
- `probe_decoder_modes`
- `mode_weights_from_fitness`
- `mode_weights_to_cdf`
- `_selector_to_mode`
- `mode_cdf` parameter on `decode_auto_mode` / LS / PR / LNS
- `use_adaptive_mode_selector` flag (default `False`)

Enable via `brkga_pack_v35(..., use_adaptive_mode_selector=True)` for
experimentation. **Not recommended for production**.

## What's next

Per the v3.8 full-eval report (`19_v38_topk_full_eval.md`), the two
next-step items with material expected return remain:

1. **Constraint-aware decoder** — wire `max_load_on_top`, `max_weight`,
   `THIS_SIDE_UP` enforcement into the JIT decoders so v3.8 is safely
   deployable on constraint-heavy industry workloads.
2. **C/Cython decoder port** — 5-10× more decodes per second, the only
   route to closing the BR1 gap to BRKGA-2013 (~1.6 pp remaining).

Adaptive selection turned out to be (correctly identified as)
incremental tuning rather than a structural lever. Both bigger items
remain on the table.
