# 3D Pallet Packer

A 3D bin-packing / pallet-loading service in pure Python that handles realistic
physical constraints existing libraries (`py3dbp`, OR-Tools, `rectpack`) miss:
per-box weight, configurable rotation, overhangs, anti-floating support, fragility
/ load-bearing limits, and per-pallet centre-of-gravity envelope.

Built in two iterations: a v1 baseline that hits the lower bound on 19/28
benchmark cases, and a v2 layer adding block-building (Eley 2002) and true
BRKGA crossover (Gonçalves & Resende 2013) that closes one more failure case
(F3) to the lower bound at the cost of 30–40 % extra runtime.

---

## Quick start

```python
from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    THIS_SIDE_UP, save_json, validate,
)

# Define cargo
boxes = [
    Box(id=f"A-{i}", length=30, width=40, height=70, weight=5.0,
        allowed_rotations=THIS_SIDE_UP, max_load_on_top=50.0)
    for i in range(20)
]

# Define pallet
pallet = Pallet(length=120, width=100, height=100, max_weight=500)

# Pack
result = PalletPacker(pallet, PackerConfig()).pack(boxes)

# Verify nothing's wrong (geometry, support, weight, etc.)
errors = validate(result, pallet, PackerConfig())
assert not errors, errors

# Save coordinates + orientations for a downstream visualizer
save_json(result, pallet, "packing.json")
```

Then `python visualize.py` renders the result as a 3D matplotlib figure.

To enable v2 improvements (block-building + BRKGA crossover), set the flags:

```python
config = PackerConfig(use_block_building=True, use_brkga=True)
```

These are opt-in because they add ~30 % runtime and only help on a small
subset of cases (see [v2 evaluation](#v2-evaluation-headline-results)).

---

## What this solves

The problem class is **3D Multi-pallet Bin Packing Problem with Side Constraints
(3D-MIBSBPP)** in Wäscher's typology — NP-hard, no closed-form solution.
Production pallet-loading goes beyond raw bin-packing because:

- Boxes have **weight** that must fit pallet capacity, and load distribution
  matters (CoG, axle-load in trucking).
- Boxes have **load-bearing limits** — fragile items can't have heavy items
  stacked on them.
- Boxes have **rotation restrictions** — bottles, electronics, anything labelled
  "this side up" can only rotate around Z.
- Boxes can't **float** — every placement except the floor must be supported by
  items below.
- Pallets often allow **edge overhang** but rarely the full pallet area.

Pure 3D bin-packing libraries optimise one of these constraints (usually
geometric fit) and ignore the rest. This packer's constraint stack handles
all of them in one engine.

The motivating scenario (also shipped as `example.py`):

> 20 boxes 30×40×70 mm + 15 boxes 40×45×50 mm + 10 boxes 60×60×30 mm,
> all on 120×100×100 mm pallets capped at 500 kg.

Result: 5 pallets, 68.5 % overall utilisation, lead pallet at 91 %.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ PalletPacker.pack()                                     │
│                                                         │
│   ┌─────────────────┐    ┌──────────────────────────┐   │
│   │ block-building  │    │ multi-start search       │   │
│   │ (v2, optional)  │ ─→ │ or BRKGA (v2, optional)  │   │
│   └─────────────────┘    └──────────────────────────┘   │
│                                       │                 │
│                                       ▼                 │
│                          ┌──────────────────────────┐   │
│                          │ _pack_once: outer loop   │   │
│                          │ (first-fit across open   │   │
│                          │  pallets)                │   │
│                          └──────────────────────────┘   │
│                                       │                 │
│                                       ▼                 │
│                          ┌──────────────────────────┐   │
│                          │ PalletState.try_place    │   │
│                          │ (extreme-point engine +  │   │
│                          │  constraint stack)       │   │
│                          └──────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### The placement engine (PalletState)

Each pallet maintains a set of **extreme points** — candidate corner positions
where the next box might fit. Algorithm from Crainic, Perboli & Tadei
(*INFORMS J. Computing* 20(3), 2008). To place a box, the engine:

1. Enumerates `(extreme_point, allowed_rotation)` pairs.
2. Adds rotation-aware floor-corner anchors at `(L − dx, 0, 0)`,
   `(0, W − dy, 0)`, `(L − dx, W − dy, 0)` per rotation — these are needed for
   the bottom-right / top-left of the pallet which a pure extreme-point set
   doesn't reach (this was a bug found in early sanity testing).
3. Filters candidates through the constraint stack (cheapest checks first).
4. Scores survivors using one of 4 strategies (`blb` bottom-left-back,
   `bbl` back-bottom-left, `max_touch`, `corner_fit`) and selects the best.
5. After placement, generates new extreme points at the placed box's
   three "exposed" corners.

### The constraint stack (PalletState.feasible)

Runs in fail-fast order — cheap rejections first:

1. **Geometry** — within pallet (or pallet + `max_overhang`), no overlap.
2. **Pallet weight budget** — `total_weight + box.weight ≤ pallet.max_weight`.
3. **Support / no-floating** — fraction of base area resting on items below
   ≥ `support_ratio` (default 0.8). Plus optional centroid-supported check
   (Junqueira, Morabito & Yamashita, *COR* 39(1), 2012). Super-blocks formed
   by v2 block-building enforce 1.0 (full support) to keep internal
   decomposition valid.
4. **Load bearing** — propagate weight through contact areas; reject if any
   direct supporter would exceed its `max_load_on_top` (Bischoff,
   *EJOR* 168(3), 2006). Cached `_top_load` per placement, updated
   incrementally on each placement (avoids the naïve O(N²) recompute).
5. **CoG envelope** — pallet centre-of-gravity must stay inside a centred
   box (default ±25 % of length/width). Only enforced once the pallet is
   ≥35 % loaded by weight (early placements naturally drift the CoG).
6. **Rotation whitelist** — only orientations in `box.allowed_rotations`.

### v1 search: multi-start

Sweeps the cartesian product of:

- **Box ordering** — seeded by heaviest-first / largest-volume-first (heavy
  on bottom for stability), plus K random perturbations (small-swap mutations).
- **Placement strategy** — 4 scoring rules above.
- **Pallet selection** — `best_fit` or `first_fit` across already-open pallets.

Picks the best result by lexicographic `(num_unpacked, num_pallets, −util)`.
Configurable via `PackerConfig.multi_start_trials` (default 20).

### v2 search: BRKGA (optional)

True biased random-key genetic algorithm with crossover (Gonçalves & Resende,
*Int. J. Production Economics* 145, 2013). Chromosome = N + 2 random keys
encoding box order + strategy + pallet-selection. Each generation:

1. Decode each chromosome → packing → fitness `(unpacked, pallets, −util)`.
2. Top 20 % = elite, carried over unchanged.
3. Elite × non-elite crossover offspring (70 % gene probability from elite).
4. 15 % fresh mutants for diversity.

Default population × generations = 30 × 8 = 240 evaluations. Auto-falls back
to multi-start when N > `brkga_n_threshold` (default 40) because BRKGA's
exploration value drops on large inputs where the placement decoder
dominates cost.

### v2 block-building (optional)

Eley 2002 / Bortfeldt 2000. Groups identical SKUs (same dims, weight,
rotations, properties); for any group with ≥ `block_threshold` units
(default 4), forms a single rectangular super-block (n_x × n_y × n_z grid)
of those boxes. The packer treats the super-block as one heavy cuboid; after
packing, super-blocks are decomposed back into individual placements.

Three block-shape strategies are tried (best result wins):

- **max** — largest single block; prefer non-spanning footprints + taller stacks.
- **column** — pure 1×1×n_z vertical columns; smallest footprint per block.
- **layer** — flat n_x×n_y×1 layers; minimal vertical footprint.

Plus a no-blocks baseline is always tried, so v2 can only match or beat v1 —
never regress.

The choice of strategy is **contextual** (depends on what other items are
co-packed); running all three is cheap because super-blocks reduce the
item count per search.

Defense-in-depth: blocks are capped at `pallet.max_weight / box.weight` units
so an over-weight super-block can't strand its member boxes as unpacked.

---

## Configuration reference

### `PackerConfig`

| Field | Default | Purpose |
|---|---|---|
| `support_ratio` | `0.8` | Min fraction of base area on supporters (1.0 = no overhang). |
| `require_centroid_supported` | `True` | Reject placements where footprint centroid isn't over a supporter. |
| `allow_pallet_overhang` | `False` | Whether boxes may extend past pallet edges (up to `pallet.max_overhang`). |
| `heavy_on_bottom` | `True` | Sort heaviest-first as seed ordering. |
| `cog_envelope_fraction` | `0.25` | Pallet CoG must stay within ±25 % of pallet centre. |
| `cog_check_min_load_fraction` | `0.35` | CoG check only kicks in when pallet ≥35 % loaded by weight. |
| `enforce_load_bearing` | `True` | Apply `max_load_on_top` propagation. |
| `multi_start_trials` | `20` | Number of (order × strategy × selection) trials in v1 search. |
| `seed` | `42` | RNG seed for reproducibility. |
| `use_block_building` | `False` | **v2 opt-in.** Enable block-building preprocessor. |
| `block_threshold` | `4` | Min K identical units to form a block. |
| `use_brkga` | `False` | **v2 opt-in.** Use BRKGA in place of multi-start. |
| `brkga_population_size` | `30` | BRKGA chromosomes per generation. |
| `brkga_generations` | `8` | BRKGA generations. |
| `brkga_elite_fraction` | `0.20` | Top fraction kept as elite. |
| `brkga_mutant_fraction` | `0.15` | Fraction added as fresh mutants. |
| `brkga_p_elite` | `0.70` | Gene-from-elite probability in crossover. |
| `brkga_n_threshold` | `40` | Above N=this, BRKGA auto-falls back to multi-start. |

### `Box`

| Field | Default | Purpose |
|---|---|---|
| `id` | required | Identifier used in placements and JSON output. |
| `length`, `width`, `height` | required | Native dimensions (mm). |
| `weight` | `0.0` | Mass for CoG and pallet weight budget. |
| `max_load_on_top` | `inf` | Max kg this box can bear stacked above. Use `0.0` for fragile (nothing on top). |
| `allowed_rotations` | `ALL_ROTATIONS` | Subset of {LWH, WLH, LHW, HWL, WHL, HLW}. Use `THIS_SIDE_UP` (around Z only) or `NO_ROTATION` (fixed). |
| `group` | `None` | Optional group label — boxes in same group prefer same pallet. |
| `requires_full_support` | `False` | If True, override `support_ratio` to 1.0. Used internally by v2 super-blocks. |

### `Pallet`

| Field | Default | Purpose |
|---|---|---|
| `length`, `width`, `height` | required | Pallet footprint and max stack height (mm). |
| `max_weight` | `inf` | Total cargo weight limit (kg). |
| `max_overhang` | `0.0` | How far boxes may extend past pallet edges (mm) — only used when `config.allow_pallet_overhang=True`. |
| `cog_x_range`, `cog_y_range` | computed | Override pallet CoG envelope; defaults derived from `cog_envelope_fraction`. |

---

## File layout

### Core library
| File | Purpose |
|---|---|
| `pallet_packer.py` | Data models, extreme-point engine, constraint stack, multi-start + BRKGA search, block-building, JSON I/O, validator. ~1100 lines. |
| `example.py` | Headline scenario (20+15+10 mixed boxes). |
| `visualize.py` | Matplotlib 3D renderer reading `packing_result.json`. |

### Benchmark suite
| File | Purpose |
|---|---|
| `benchmark.py` | 28 cases: A1–A5 (trivial), B1–B4 (known-optimal), C1–C4 (heterogeneous), D1–D8 (constraint sweep), E1–E4 (pathological), F1–F3 (real-world scale). |
| `benchmark_output.txt` | v1 results snapshot. |
| `visualize_cases.py` | Per-case 3D rendering → `case_gallery.png`. |

### Failure-mode suite
| File | Purpose |
|---|---|
| `failure_cases.py` | 13 cases F1–F13: targeted at suspected v1 weaknesses. Maps each failure to a literature-known fix. |
| `failure_output.txt` | v1 results snapshot. |
| `visualize_failures.py` | Per-case rendering → `failure_gallery.png`. |
| `FAILURE_ANALYSIS.md` | Failure-mode → literature-reference mapping. |

### v2 comparison
| File | Purpose |
|---|---|
| `compare_v1_v2.py` | Side-by-side harness across all 41 cases. |
| `benchmark_v1_v2_comparison.txt` | Benchmark suite v1 vs v2 results. |
| `failure_v1_v2_comparison.txt` | Failure suite v1 vs v2 results. |
| `visualize_v1_v2.py` | F3 before/after rendering. |
| `f3_v1_vs_v2.png` | The F3 packing visualization (the headline improvement). |
| `V2_COMPARISON.md` | Full v2 evaluation writeup. |

### Reports
| File | Purpose |
|---|---|
| `ANALYSIS.md` | v1 benchmark deep-dive: constraint sensitivity, multi-start ablation, seed variance. |
| `FAILURE_ANALYSIS.md` | Failure-mode → literature-fix mapping. |
| `V2_COMPARISON.md` | What v2 delivered, what it didn't, runtime tradeoffs, when to enable it. |

### v1 snapshot
| File | Purpose |
|---|---|
| `v1_snapshot/` | Exact v1 source + outputs preserved before v2 changes. |

---

## v2 evaluation: headline results

Across all 41 cases (28 benchmark + 13 failure), with `seed=42`:

| Metric | v1 baseline | v2 (block + BRKGA) |
|---|---|---|
| Benchmark cases (28) — pallets | 79 | 79 |
| Failure cases (13) — pallets | 26 | **25 (−1)** |
| Failure suite — avg util | 70 % | **74 % (+4 pp)** |
| Validator errors | 0 | 0 |
| Improvements (Δp < 0) | — | **1 (F3)** |
| Regressions (Δp > 0) | — | **0** |
| Total runtime | 109 s | 150 s (+38 %) |

**Net: v2 strictly dominates v1.** Either matches or beats; never worse.

### The improvement: F3 "interlock big + small"

| | Pallets | Util |
|---|---|---|
| v1 | 2 | 45 % |
| v2 | **1** | **90 %** |

The case: 6 boxes 80×50×30 + 6 boxes 40×50×30 mm on a 120×100×100 mm pallet.
Block-building forms a 1×2×3 block of small boxes (40×100×90) and a 1×2×3
block of big boxes (80×100×90); BRKGA finds a chromosome with small-first
ordering that lets both blocks fit side-by-side along X (80 + 40 = 120).

v1's heaviest-first sort plus extreme-point placement misses this — two
small boxes end up on a second pallet at 21 % util.

See `f3_v1_vs_v2.png` for the visualization.

### Reproducing

```bash
python compare_v1_v2.py --suite both --pop 16 --gen 4
```

Outputs are deterministic with `seed=42` (default).

---

## What v2 doesn't fix

Three failure modes remain with v1's +1-pallet gap:

| Case | N | v1 result | volume LB | Likely cause |
|---|---|---|---|---|
| F1 Pareto continuum | 53 | 5 pallets, 60 % util | 4 | Volume LB is likely loose; true geometric LB could be 5. Heterogeneous SKUs (8 SKUs, 5–8 each) don't have enough K per group for blocks to compose well. |
| F9 long this-side-up | 18 | 3 pallets, 66 % util | 2 | Volume LB ignores rotation restrictions; unrotatable items need specific orientations and the geometric LB is probably 3. |
| F12 strongly heterogeneous at scale | 72 | 3 pallets, 65 % util | 2 | N=72 exceeds BRKGA threshold so falls back to multi-start; 40 SKUs means almost no group meets K≥4 for blocks. Closer to random bin-packing than structured. |

These need fundamentally different approaches:

- **Tighter LB computation** would tell us whether v1 is actually optimal
  (the volume LB may be the only thing flagging these as "failures").
- **MIP** (do Nascimento, Queiroz & Junqueira, *COR* 128, 2021) would
  give true optimality but at 100–1000× cost.
- **Layer-building decoder** (George & Robinson, 1980; Bischoff & Ratcliff,
  *OMEGA* 1995) would explore qualitatively different packings than our
  extreme-point engine.
- **Ejection chains** (Glover-style, applied to bin-packing by Lodi,
  Martello & Vigo) could escape local optima multi-start and BRKGA both
  converge to.

---

## When to use what

| Use case | Recommendation |
|---|---|
| Default / production | v1 (`PackerConfig()`). Fast, simple, hits LB on 19/28 benchmark cases. |
| Cargo has 4+ identical-SKU groups | Enable `use_block_building=True`. Highest-ROI v2 feature. |
| Small shipments (N ≤ 40) with low-util last pallet | Enable `use_brkga=True`. Explores orderings v1's perturbation misses. |
| Large shipments (N > 50) | Stick with v1 — BRKGA auto-disables above N=40, block-building rarely finds groups in heterogeneous mixes. |
| Need provable optimality | Out of scope. Use a MIP solver (Gurobi, CPLEX) with formulation from do Nascimento et al. 2021. |

---

## Validator

The `validate()` function checks every result against the full constraint
stack independently of the packer. Used by every test in `benchmark.py`,
`failure_cases.py`, and `compare_v1_v2.py`. As of the latest evaluation:

- 28 / 28 benchmark cases — 0 validator errors (v1 and v2).
- 13 / 13 failure cases — 0 validator errors (v1 and v2).

Bugs caught and fixed during testing:

1. **CoG check on first placement** — was rejecting first box because CoG
   would be off-centre. Fixed by gating CoG check on
   `cog_check_min_load_fraction` (default 35 %).
2. **EPs missing pallet far corners** — corner-anchor enumeration was
   incomplete. Fixed by adding rotation-aware anchors at
   `(max(0, L − dx), max(0, W − dy), 0)`.
3. **O(N²) load-bearing recompute** — replaced with cached `_top_load`
   updated incrementally on each placement.
4. **v2 quality-comparison inversion** — `min(candidates, key=(pallets, unpacked, util))` preferred
   "0 pallets / N unpacked" over "3 pallets / 0 unpacked" because tuple
   compare prioritised pallets first. Fixed to
   `(unpacked, pallets, −util)`.
5. **v2 overweight blocks** — super-block could exceed pallet weight, stranding all member boxes. Now capped by
   `max_units_by_weight` in `_find_best_block`.

---

## Possible next steps

In rough order of effort × payoff:

1. **Compute a tighter LB** — per-SKU max-fit-per-pallet aggregated via LP
   relaxation. Would tell us whether F1, F9, F12 are actually unsolved or
   if v1 is already optimal. Cheap, high information value.

2. **Layer-building decoder** as alternative to extreme-point — group items
   by height, do 2D packing per layer. Would help F13 (pure layered cargo)
   and provide qualitatively different solutions for BRKGA to explore.
   Medium effort (~300 LOC), modest expected gain.

3. **MIP solver wrapper** — call Gurobi/CPLEX for small inputs (N ≤ 20) with
   the formulation from do Nascimento et al. 2021. Provides provable
   optimality for the small-shipment slice. Adds external dependency.

4. **Ejection chains** — after greedy packing, try removing 1–3 placed items
   and inserting unplaced ones; iterate. Standard local-search improvement.
   Medium effort, unknown gain on top of current pipeline.

5. **SKU-consistent rotation** — when multiple identical SKUs are present,
   force them all to use the same rotation. Reduces search space, may
   improve F9-style cases. Cheap.

6. **Adversarial block shapes** — let BRKGA's chromosome encode which block
   shape to use per group (currently we try 3 fixed strategies and pick
   the best). Would let the search learn shape preferences per problem.

7. **Truck-loading / axle-load** — extend pallet model to vehicle bed +
   axle weight distribution constraints (Ramos, Silva & Oliveira,
   *EJOR* 266(3), 2018). Real-world deployment likely needs this.

8. **Multi-pallet co-optimisation** — currently the outer loop is greedy
   first-fit-decreasing across pallets. Co-optimising the assignment with
   the placement could close some +1-pallet gaps (especially F1).

---

## References

Key papers the algorithm draws from:

- **Crainic, Perboli & Tadei (2008)** — Extreme-point heuristic for 3D-BPP.
  *INFORMS J. Computing* 20(3).
- **Junqueira, Morabito & Yamashita (2012)** — Support-factor α model
  (centroid-supported constraint). *Computers & OR* 39(1).
- **Bischoff (2006)** — Load-bearing constraints / pressure propagation.
  *EJOR* 168(3).
- **Gonçalves & Resende (2013)** — BRKGA for 3D bin-packing.
  *Int. J. Production Economics* 145.
- **Eley (2002)** — Block-building heuristic. *EJOR* 141.
- **Bortfeldt (2000)** — Earlier block-building. *EJOR* 124.
- **Ramos, Silva & Oliveira (2018)** — Road-transport / axle-load
  constraints. *EJOR* 266(3).
- **do Nascimento, Queiroz & Junqueira (2021)** — Exact MIP with 12 side
  constraints. *Computers & OR* 128.
- **Wäscher, Haußner & Schumann (2007)** — Cutting & packing typology
  (where "3D-MIBSBPP" comes from). *EJOR* 183(3).

---

## Git history

```
cb7e9a5  v2 complete: comparison harness + reports + visualization
a3472d1  v2 fixes: quality function and overweight blocks
9664bd1  v1 baseline + v2 in-progress (block-building bug pending)
```

The v1_snapshot/ directory preserves the v1 source exactly as it was before
the v2 changes, for direct comparison.
