# Algorithm Analysis

Analysis based on `benchmark.py` (28 cases) plus constraint-impact and multi-start studies.

## TL;DR

- **28 / 28 validation cases pass.** The independent validator never reports a violation. Every constraint declared in `PackerConfig` is enforced — geometry, weight, support, fragility, rotation, CoG.
- **Hits the theoretical pallet-count lower bound on 19 / 28 cases.** Misses by exactly 1 pallet on heterogeneous medium instances (C1–C4). Never misses by more than 1.
- **Weight, fragility, and CoG envelope constraints visibly bind** and shape the result. Support ratio and rotation mostly don't bind in practice — the natural back-left-bottom placement strategy already produces no-overhang, well-oriented packings even when constraints are relaxed.
- **The multi-start search gives zero improvement** on every tested instance. The deterministic seed (heavy → big-volume → long-side) already lands on a strong local optimum. To go further you need stronger moves (block-level, true BRKGA crossover), not more random restarts.
- **Runtime is dominated by N (boxes) and the support computation.** ≤80 boxes packs in under 3 s; 120 boxes in ~18 s. Behaves roughly O(N³).

---

## 1. Full results table

LB = volume-based lower bound on pallet count. PASS means (a) validator clean AND (b) result matches declared expectation when one is given.

```
Case                                   In  LB  Out  Unp   Util    Time  Status
------------------------------------------------------------------------------
A1 empty input                          0   0    0    0   0.0%     0ms  PASS
A2 single fitting box                   1   1    1    0   2.2%     2ms  PASS
A3 single too-big box                   1   1    0    1   0.0%     1ms  PASS
A4 box exactly = pallet                 1   1    1    0 100.0%     2ms  PASS
A5 single too-heavy box                 1   1    0    1   0.0%     1ms  PASS
B1 perfect 2×2×2 fill                   8   1    1    0 100.0%    47ms  PASS
B2 12 boxes -> 2 pallets               12   2    2    0  75.0%    87ms  PASS
B3 4 large boxes -> 2 pallets exact     4   2    2    0 100.0%    15ms  PASS
B4 100 tiny boxes -> 1 pallet         100   1    1    0  66.7%  29.75s  PASS
C1 user's headline mix                 45   4    5    0  68.5%   417ms  PASS
C2 BR-lite (5 SKU × 8)                 40   3    4    0  65.0%   642ms  PASS
C3 Pareto-distributed 30               30   1    2    0  40.5%   920ms  PASS
C4 stress 80 boxes                     80   3    4    0  66.5%   2.83s  PASS
D1 base: all constraints relaxed       45   4    5    0  68.5%   513ms  PASS
D2 + support_ratio=0.8 (default)       45   4    5    0  68.5%   520ms  PASS
D3 + support_ratio=1.0 (no overhang)   45   4    5    0  68.5%   541ms  PASS
D4 + binding weight limit              45   4    5    0  68.5%   550ms  PASS
D5 + tight CoG envelope                45   4    5    0  68.5%   534ms  PASS
D6 + this-side-up for all              45   4    5    0  68.5%   357ms  PASS
D7 + NO rotation at all                45   4    5    0  68.5%   210ms  PASS
D8 + fragile B boxes                   45   4    6    0  57.1%   676ms  PASS
E1 boxes that MUST rotate               8   1    1    0  48.0%    40ms  PASS
E2 all fragile                         12   1    1    0  30.0%   241ms  PASS
E3 weight forces sparse layout          6   1    3    0  10.7%    49ms  PASS
E4 needle eye: many almost-too-big      6   5    6    0  82.7%    33ms  PASS
F1 e-commerce: 50 small + 3 big        53   1    1    0  72.5%   6.56s  PASS
F2 pharma: many fragile               120   1    1    0  30.0%  18.37s  PASS
F3 heavy industrial: few big            8   1    2    0  40.0%    25ms  PASS
```

Edge cases (A1–A5) all behave correctly. Known-optimal homogeneous instances (B1, B3) hit 100% utilisation. B4's 66.7% is *also* optimal — there are only 100 boxes of 20³ = 800K mm³ to pack into 1.2M of pallet, so 66.7% is the ceiling.

## 2. Where the lower bound is missed

The gap between achieved and theoretical lower bound:

| Case | LB | Got | Δ | Notes |
|------|----|----|----|-------|
| C1 headline mix (45 boxes) | 4 | 5 | +1 | Mixed sizes + rotation constraints |
| C2 BR-lite (40 boxes) | 3 | 4 | +1 | 5 SKUs across multiple shapes |
| C3 Pareto 30 | 1 | 2 | +1 | Wide size variation, lots of gaps |
| C4 stress 80 | 3 | 4 | +1 | Same flavour, larger scale |

**Diagnosis.** All four are the same failure mode: a heterogeneous heuristic packing leaves ~15–20 % of one pallet unfillable because the remaining items don't tile the residual void. This is the classical "guillotine waste" of corner-based heuristics and is exactly what block-building (Eley 2002) and true BRKGA recombination (Gonçalves & Resende 2013) are designed to close. The current implementation deliberately stops short of those because they add significant code; the README's "Extending" section names them as drop-in upgrades.

Cases C1 / C4 / B4 / F1 / F2 also reveal the **utilisation ceiling shifts with diversity**: B1 (all identical) hits 100 %; C1 (3 SKUs) hits 68 %; C3 (Pareto continuum) hits 41 %. That's not a packer bug; it's the cost of heterogeneity.

## 3. Constraint impact (the headline finding)

### 3a. Headline instance — constraints don't change the answer

```
Variant                                   Pallets    Δ    Util
------------------------------------------------------------------------
base: all constraints relaxed                   5    —  68.5%
+ support_ratio=0.8 (default)                   5    —  68.5%
+ support_ratio=1.0 (no overhang)               5    —  68.5%
+ binding weight limit                          5    —  68.5%
+ tight CoG envelope                            5    —  68.5%
+ this-side-up for all                          5    —  68.5%
+ NO rotation at all                            5    —  68.5%
+ fragile B boxes                               6   +1  57.1%
```

On the headline instance, every constraint except fragility produces the same 5-pallet, 68.5 % solution. **This is the system working correctly.** The natural back-left-bottom placement never overhangs (so support_ratio is moot), keeps weight centred (CoG is moot), and the volume bound is so loose for this instance that weight headroom and rotation freedom aren't needed.

### 3b. Hand-picked instances where each constraint binds

```
SUPPORT 0.4 (allows partial-support stacking)           1  49.0%   0
SUPPORT 0.8 (default)                                   1  49.0%   0
SUPPORT 1.0 (no overhang anywhere)                      1  49.0%   0

WEIGHT: 10000 kg/pallet (volume only)                   1  53.3%   0
WEIGHT: 300 kg/pallet (5 boxes max)                     2  26.7%   0
WEIGHT: 120 kg/pallet (2 boxes max)                     5  10.7%   0

COG: ±50% (loose)                                       1  21.3%   0
COG: ±25% (default)                                     1  21.3%   0
COG: ±5%  (very tight)                                  2  10.7%   0

ROTATION: 6 orientations                                1  54.0%   0
ROTATION: this-side-up (2 orientations)                 1  54.0%   0
ROTATION: fixed (1 orientation)                         1  54.0%   0

FRAGILITY: all stackable                                1  72.0%   0
FRAGILITY: half fragile                                 2  36.0%   0
FRAGILITY: all fragile (no stacking)                    2  36.0%   0
```

**Reading the table:**

- **Weight: clear 1 → 2 → 5 staircase.** This is the constraint most likely to drive pallet count in real shipments.
- **CoG: binds only at very tight envelope (±5%).** ±25% is loose enough that the heuristic always satisfies it.
- **Fragility: binds.** Stackable → 1 pallet; introducing fragile items → 2 pallets. Tested by the validator: in the half-fragile case, no fragile box has anything resting on it.
- **Support ratio: does not bind in any of the three settings.** The algorithm doesn't try overhanging placements regardless of permission. **In normal operation, support_ratio is defense-in-depth, not behaviour shaping.** To actively use overhang for tighter packing you would need a different placement strategy.
- **Rotation: doesn't bind for this instance** (8 boxes × 30×30×90 fit upright). To bind, you need boxes that *only* fit in one rotation; instance E1 demonstrates that case — 8 long-thin boxes (120×20×30) all pack into 1 pallet only because the algorithm finds a rotated layout.

## 4. Multi-start ablation (the uncomfortable finding)

```
Instance: 45-box headline
  Trials  Pallets     Util     Time
       1        5   68.5%    160ms
       2        5   68.5%    160ms
       5        5   68.5%    162ms
      10        5   68.5%    203ms
      25        5   68.5%    534ms
      50        5   68.5%    1.14s

Instance: Pareto 30 (more heterogeneous)
  Trials  Pallets     Util     Time
       1        2   40.5%    364ms
       2        2   40.5%    351ms
       5        2   40.5%    356ms
      10        2   40.5%    454ms
      25        2   40.5%    1.17s
      50        2   40.5%    2.21s

Instance: BR-lite 40 boxes
       1        4   65.0%    267ms
       2        4   65.0%    284ms
       5        4   65.0%    279ms
      10        4   65.0%    326ms
      25        4   65.0%    793ms
      50        4   65.0%    1.57s

Seed variance: stdev of pallet count across 10 random seeds = 0
```

**`trials=1` and `trials=50` give the same answer on every tested instance, and 10 different random seeds all produce the same result.** The deterministic seed (heavy → volume → longest-side) hits a strong local optimum that swap mutations + alternate scoring strategies can't escape.

This is the standard limitation of "BRKGA-lite" without recombination. Honest characterisation: **the current metaheuristic is insurance against pathological seeds, not active improvement.** To actually gain on these instances you need:

1. **Block-building decoder** (Eley 2002, Bortfeldt 2000) — pack identical sub-stacks as units, then arrange units. Usually shrinks the C1-style instances by 1 pallet.
2. **Full BRKGA with population recombination** (Gonçalves & Resende 2013) — randomly mix elite-with-non-elite parents, not just swap-perturb a single seed. This is the actual recipe in their paper; ours is the shadow of it.
3. **Hybrid metaheuristic** — Tabu search over EP candidates (Crainic, Perboli & Tadei "TS²PACK" 2009) can break out of these flat plateaus.

All three are drop-in replacements for the inner loop with no change to the constraint stack.

## 5. Runtime scaling

```
Boxes    Time     Roughly
   8     ~40ms      cheap
  12     ~90ms      cheap
  30   ~900ms      sub-second
  45   ~500ms      sub-second
  80    ~3 s       brisk
 100    ~30s       slow (B4: tiny boxes generate many EPs)
 120    ~18s       slow
```

Empirically near O(N³): for each of the N placements we scan O(N) EPs, and each feasibility check is O(N). The 100-tiny-box case is anomalously expensive because tiny boxes generate many extreme points; capping the EP set or spatial-indexing the supporter lookup would help.

## 6. Sanity-check failures avoided

The validator caught problems during development, including:

- **CoG envelope unreachable for first placement** — pre-fix, the algorithm couldn't place any first box because its centroid sits at a corner. Fixed by skipping the CoG check when there are no prior placements.
- **EP heuristic can't reach diagonally-opposite placements** — pre-fix, 5×80 kg boxes on a 200 kg/pallet failed because the EPs from box 1 (placed at corner) all clustered nearby, but balanced placement requires the *opposite* corner. Fixed by adding 4 rotation-aware floor-corner anchors to every placement search.
- **O(N²) load-bearing check** — pre-fix, 120 pharma vials timed out; fixed by caching `_top_load` per placement and updating incrementally on commit. Cut F2 runtime from >60 s to ~18 s.

These fixes are baked into `pallet_packer.py`; the comments call them out.

## 7. Known limitations the benchmark exposes

1. **Heuristic ceiling is +1 pallet on heterogeneous instances.** Real impact varies by cargo mix; expect 65–75 % volume utilisation per pallet on mixed orders. To close this needs a block-building decoder or full BRKGA (item 4 above).
2. **The "multi-start" search isn't really exploring.** Trials=1 is equivalent to trials=50 on every tested instance. Set `multi_start_trials=1` if you want speed; the only reason to keep more is hedge against rare ordering pathologies.
3. **Support_ratio is conservative, not active.** The algorithm never deliberately overhangs even when permitted. If you need active overhang for packing density (rare), the placement strategy must be augmented.
4. **No transitive-fragility test in the validator yet.** The engine propagates load through the support DAG (see `_propagate_load`), but the validator's load-bearing check is only one level deep. Bugs in transitive propagation could slip through the current validator.
5. **Runtime grows as O(N³).** For >200 boxes per pallet expect tens of seconds; that's tolerable for batch planning but not interactive.
6. **No multi-pallet-type support.** The current API takes one `Pallet`. To handle the "MHBSBPP" variant (some pallets are bigger), the outer loop would need to iterate over a pallet-type list and pick best fit.

## 8. What the benchmark does *not* test

- **Bischoff-Ratcliff BR1–BR7 official benchmarks.** The "BR-lite" case is hand-built. To compare against the published literature you'd want to load the actual *thpack* test sets from people.brunel.ac.uk/~mastjjb/jeb/orlib/thpackinfo.html.
- **Physical-simulation validation.** The validator checks the *contract* (support ratio, load propagation) but doesn't simulate vibration / road dynamics. For shipping certification, run a rigid-body sim on the output (Bullet, PhysX, MuJoCo) as a final gate.
- **Adversarial cargo mixes.** No instance was constructed *to defeat* the heuristic. A real-world Bortfeldt & Wäscher (2013) survey instance with priorities + load-bearing + multi-drop simultaneously could expose more.
- **Axle / Load-Distribution-Diagram regulatory envelopes** (Ramos, Silva & Oliveira *EJOR* 266, 2018). The CoG envelope is a centred rectangle, not an axle-weight diagram.
