# Phase 1 — Bischoff-Ratcliff Benchmark Report

Anchors our packer against the published literature. Sets BR1, BR3, BR5 were
loaded from OR-Library's thpack1/3/5 files; the first 3 instances of each were
run in pure-geometric mode (no weight, fragility, CoG, or load-bearing, to
match the published benchmark setup). Metric is mean volume utilization on
pallet #1 across the sample.

A note on sample size: we ran 3 instances per set (the published baselines
report means over 100). With stdev of 7-10 percentage points across our
sample, the 95% confidence interval on our mean is roughly ±10pp. So our
numbers are noisy point estimates — read the *patterns*, not the precise
values. Larger samples can be run with `python3 br_benchmark.py --sample N`
once Phase 5 reduces per-instance runtime.

## Headline numbers

| Set | N inst. | Boxes per instance | v1 util | v2 util | BR-1995 | Bortfeldt-2000 | CPT-2008 | BRKGA-2013 |
|---|---|---|---|---|---|---|---|---|
| BR1 (3 SKUs) | 3 | 112-138 | **78.7% ± 6.7** | 78.7% ± 6.7 | 83.1% | 87.8% | 87.9% | 92.6% |
| BR3 (8 SKUs) | 3 | 94-143 | **75.8% ± 7.4** | 75.8% ± 7.4 | 79.5% | 85.6% | 86.4% | 90.5% |
| BR5 (12 SKUs) | 3 | 98-138 | **72.1% ± 9.6** | (n/r) | 76.3% | 83.0% | 84.0% | 88.7% |

Per-instance breakdown:

| Set | Instance | N | v1 util | v1 time | v2 util | v2 time |
|---|---|---|---|---|---|---|
| BR1 | #1 | 112 | 79.3% | 3.4s | 79.3% | 3.9s |
| BR1 | #2 | 138 | 85.1% | 8.6s | 85.1% | 9.7s |
| BR1 | #3 | 127 | 71.8% | 3.7s | 71.8% | 4.1s |
| BR3 | #1 | 94  | 68.7% | 2.1s | 68.7% | 3.4s |
| BR3 | #2 | 115 | 83.5% | 4.6s | 83.5% | 6.1s |
| BR3 | #3 | 143 | 75.3% | 8.3s | 75.3% | 9.1s |
| BR5 | #1 | 98  | 67.7% | 2.9s | — | — |
| BR5 | #2 | 138 | 83.1% | 8.1s | — | — |
| BR5 | #3 | 133 | 65.5% | 7.9s | — | — |

All instances passed the validator (zero errors) on both v1 and v2 runs.

## Where we stand

**Roughly 4 percentage points behind the original 1995 baseline; 13-17 points
behind modern state-of-the-art.**

- **BR1** — v1 at 78.7%, BR-1995 at 83.1%. Gap −4.4 pp. The 1995 paper used
  a layer-building heuristic; we're behind because the extreme-point engine
  builds vertically rather than in layers, which is suboptimal when the
  container is much longer than tall.
- **BR3** — v1 at 75.8%, BR-1995 at 79.5%. Gap −3.7 pp. Consistent with BR1.
- **BR5** — v1 at 72.1%, BR-1995 at 76.3%. Gap −4.2 pp. Same pattern.
- Gap to **BRKGA-2013** is roughly −14 to −17 pp on all three sets — that's
  the modern bar (population-based search with real crossover, layer
  decoding, and 100 generations). Closing it is the Phase 2 + Phase 3 work.

## Why v2 didn't help

Identical utilization on every instance where both ran. Two reasons:

**BRKGA auto-disabled at BR scale.** The `brkga_n_threshold` is 40; every BR
instance has N=94 or more, so BRKGA falls back to multi-start and (with
`multi_start_trials=1`) reduces to a single decoder run. This was the
deliberate choice from the v2 ablation work — BRKGA's overhead wasn't paying
off at large N — but it means v2 contributes nothing on BR.

**Block-building runs but converges to the same packing.** BR instances do
have SKUs above the `block_threshold=4` (BR1 instance 1 has SKU quantities
40/33/39, all qualifying). The block-builder forms super-blocks; the extreme-
point decoder packs them; but the resulting layout happens to match the
no-blocks baseline on these inputs. The "block placement reduces fragmentation"
benefit doesn't materialize when the container is much larger than any single
block — the heterogeneous remainder fills the leftover space the same way
either way.

This is exactly the dynamic Phase 2c (adversarial block shapes in the BRKGA
chromosome) and Phase 2e (layer-building decoder) are aimed at: give the
search qualitatively different decoders to compare, so block-building's value
shows up against a different decoding strategy.

## Phase 2 implications

This anchoring run reaffirms the priorities in ROADMAP.md, with one
adjustment to the order:

**Promote Phase 2b (GRASP randomization) and Phase 2e (layer-building
decoder) earlier in Phase 2.** Three of the four published baselines that
beat us heavily — Bortfeldt 2000, CPT 2008, BRKGA-2013 — all rely on either
layer-building or randomized-greedy as a key ingredient. Our pure
extreme-point decoder is the floor of what the literature reports. Adding a
layer-building decoder as an alternative inside the BRKGA chromosome should
move us +5-8 pp toward Bortfeldt's numbers.

**Raise `brkga_n_threshold` for BR comparison runs.** Setting it to 200+
(or removing the threshold) would let BRKGA actually engage on BR instances.
The original threshold was added because BRKGA wasn't helping at N>40 on
our 41-case suite; on BR's larger, more heterogeneous inputs the calculus
may differ. Worth re-evaluating after Phase 2b/e.

**SKU-consistent rotation (Phase 2a) may be high-ROI on BR1.** BR1 has only
3 SKUs per instance — perfect candidate for the Bortfeldt-Gehring 2001
pre-decision. Likely +2-3 pp on BR1 alone.

## Reproducing

```bash
# v1 + v2, 3 instances per set (≈90 seconds total)
python3 br_benchmark.py --sample 3 --sets thpack1 thpack3 thpack5

# Just v1, larger sample (faster — skip v2)
python3 br_benchmark.py --sample 10 --sets thpack1 thpack3 thpack5 --skip-v2

# Add BR7 (slow — 20 SKUs per instance, large N)
python3 br_benchmark.py --sample 5 --sets thpack7 --skip-v2
```

OR-Library source: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/thpackinfo.html

## File layout

| File | Purpose |
|---|---|
| `br_benchmark.py` | Parser + harness + report writer. |
| `br_data/thpack{1,3,5,7}.txt` | First 20 (or 10 for thpack7) instances of each BR set. |
| `BR_REPORT.md` | This file. |
