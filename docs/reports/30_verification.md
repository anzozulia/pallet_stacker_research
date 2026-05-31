# Deep Verification — Cython BRKGA Pallet Packer

A pre-productionization audit answering one question: **is the algorithm
correct and ready for real-world cases?** Run as a 15-dimension adversarial
verification (30 agents, every defect independently re-confirmed via a second
counter-probe), plus a controlled timing/throughput pass. All execution inside
the project Docker image (`pallet-packer:dev`, Cython backend active,
`_BATCH_AVAILABLE=True`), verified bit-identical and thread-invariant.

> **Bottom line.** The **Cython port itself is faithful, correct, deterministic,
> and memory-safe** — bit-identical to the Numba reference across ~7,000
> comparisons, conservation-safe, no out-of-bounds, no data races. But the
> **underlying algorithm is *not* ready for arbitrary real-world input.** Seven
> distinct defects surfaced — two of them blockers for the planned public
> service: (1) the engine packs on an **integer grid but validates against float
> geometry**, so non-integer real-world dimensions produce genuinely overlapping
> packings (85.7% of fractional-dim inputs fail the independent validator); and
> (2) it **does not scale to the intended 5,000-box limit** and does not honor
> its own time budget at scale. The academic BR benchmark and the 10 small
> integer-dimension industry cases never exercised these paths, which is why the
> prior audit graded the algorithm "production-ready."

---

## 1. Verdict at a glance

| Area | Verdict |
|---|---|
| Cython port fidelity (≡ Numba reference) | ✅ **Solid** — bit-identical, no port-introduced defect |
| Determinism (seed + thread-invariance) | ✅ **Solid** — bit-identical, OMP-invariant in converged regime |
| Memory safety (`.pyx` kernels) | ✅ **Solid** — no OOB / aliasing / div-by-zero / overflow at real scale |
| Box conservation (multi-pallet) | ✅ **Solid** — 0 lost / duplicated across 66 runs |
| Parallel decode throughput | ✅ **Solid** — ~6× on 8 threads, near-linear |
| Correctness on **integer-dim** academic/industry workloads | ✅ Good (with the stability caveat below) |
| Correctness on **non-integer real-world input** | ❌ **Blocker** — invalid packings |
| Scale to 5,000 boxes / time-budget SLA | ❌ **Blocker** — infeasible / soft budget |
| Physical stability on weightless input | ❌ High — support enforcement silently dropped |
| Constraint decoders at scale (modes 4/5) | ❌ High — emit support/load violations |
| `Box.group` co-location | ❌ High — documented feature never implemented |
| `max_pallets` arg honored | ⚠️ Medium — leaked on v2-seed path |
| Malformed-input handling | ⚠️ Medium — silent-wrong / no boundary validation |

---

## 1b. Scope decisions & resolution path (post-verification)

Decisions taken after reviewing the findings, and what each one closes:

| Decision | Effect on the defect list |
|---|---|
| **Integer dimensions only** (unit-agnostic; reject non-integer input at the API) | **Closes Blocker 1 entirely.** Verification proved integer inputs are 100% validator-clean; the int/float gap only exists for fractional input. No engine change — an input-validation contract. |
| **Drop the 5,000 target; use a feasible cap** | **Closes Blocker 2.** Measured feasible cap ≈ **500 boxes at a ~60s budget** (constrained: N=500 → 60s on-budget, all placed; N=750 → +52% overrun). Geometric workloads scale higher. Bigger budget → higher cap once the hard-deadline fix lands. |
| **Enforce stability always** (Option A) | **Fixes the weightless-stability defect** by decoupling `support_ratio` from `has_constraints` — every box rests on ≥`config.support_ratio` support regardless of whether weights were supplied. |

**Remaining work to make the algorithm service-ready (within integer + capped scope):**

1. **Block-decoder constraint-awareness** (the HIGH defect above) — *the #1 algorithm fix*. Put `mlot`/`rfs` in the SKU key + enforce per-box load/support inside block decomposition. Preserves the 2–3× density the block modes provide while making them valid.
2. **Always-on stability** — decouple `support_ratio` (and `require_centroid`) from `has_constraints` (driver.py:190-191).
3. **Input-validation gate** — integer dims, positivity, finiteness, the N cap; reject with a 4xx instead of silent-wrong. Also closes the NaN-weight / negative-dim cases.
4. **Hard time deadline** — check budget inside the decode/build loop so the async worker timeout is a real SLA.
5. **`max_pallets` propagation** to the v2-seed packer (trivial).
6. **`Box.group`** — implement co-location, or drop it from the API for the MVP.

None are research-grade; this is an engineering pass, not a redesign.

**Status (post-verification fixes):**

- ✅ **Item 1 — block-decoder constraint-awareness** — DONE (commit `ca8d813`).
  Part A: `mlot`/`rfs` in the SKU key. Part B: per-box support/load check in the
  block decoder (mirrored to the Numba reference). Constrained ladder + 10
  industry cases: 0 errors (was 18–64); geometric BR bit-identical; backends
  re-verified bit-equivalent (2670 comparisons, 0 mismatches).
- ✅ **Item 2 — always-on stability** — DONE (commit `048a83c`).
  `support_ratio`/centroid/CoG/overhang decoupled from `has_constraints` and
  driven from config (`use_cstr_path`). Weightless input now enforces support:
  0 validator errors at every support level; enforcement binds (BR1#1 weightless
  util 91.96% at sr=0 → 77.11% at sr=0.8). `use_v2_seed` kept keyed on real
  weight/load constraints (Phase 7a basin protected).
- ⏳ Items 3–6 (input-validation gate, hard deadline, `max_pallets` leak,
  `Box.group`) — pending.

> **⚠️ Research-narrative flag discovered during the D2 fix.** Because support was
> never enforced on weightless data, the published BR utilization numbers (and
> the "beats 2013 SOTA on BR3/5/7" claim in `29_final_audit.md`) are **measured
> without support enforcement**. If the literature baselines (G&R 2013, Lim 2013)
> assume full support — as the `geometric_only_config` docstring asserts
> Bischoff–Ratcliff does — then our BR util is **inflated** (BR1#1 drops 91.96% →
> 77.11% once support_ratio=0.8 is enforced). The academic comparison may be
> apples-to-oranges. This does **not** affect the service (which enforces
> stability), but the SOTA claim needs an honest re-measurement before it's relied
> upon.

---

## 2. What is solid (the port did its job)

These dimensions were probed adversarially and held:

- **Cython ≡ Numba, bit-identical.** Geometric decoders: **4,368** comparisons
  (modes 0–5, N 1–150, varied pallets, tiny/oversize boxes, n_skus 1–N,
  max_pallets 0–10) → **0 real divergences**. Constraint-aware decoders:
  **2,670** comparisons exercising every guard (finite/inf weight cap, finite/inf
  mlot, rfs on/off, support_ratio {0,0.5,0.8,1.0}, centroid 0/1, CoG tight/loose,
  overhang 0/>0) → **0 mismatches**. The only geometric divergence is on
  degenerate *zero-dimension* boxes (Numba raises `ZeroDivisionError`; Cython
  degrades to a 1×1×1 block) — benign and unreachable on the production hot loop
  (which is 100% Cython batch).
- **Batch ≡ per-chromosome.** 187/189 sub-checks bit-identical across pop_size
  {1,2,500}, N 1–200, n_skus 1–N, degenerate dims, all constraint combos, and the
  mode-5 per-chromosome top-K path. (The 2 "misses" are the float fitness-formula
  gap below — not a batch bug.)
- **Determinism.** Bit-identical at fixed seed; `GLOBAL_SHA256` **identical under
  OMP_NUM_THREADS=1 and =8** (the prange static-schedule contract); n_restarts=2
  and multi-pallet constrained solves deterministic. (Time-bounded *un-converged*
  runs vary with the wall-clock budget — expected, not a race.)
- **Memory safety / static review.** 24/24 `.pyx` checks pass: no out-of-bounds,
  no scratch-array aliasing across prange threads, no div-by-zero trap, no
  uninitialized placement rows, no off-by-one in block/layer loops. The lone
  latent int64 overflow needs pallet dims > 1.5×10⁹ (unreachable at mm scale) and
  is shared with the Numba reference.
- **Weight cap.** 0 overruns across 26 adversarial cases; lands *exactly* on the
  cap (cap=200 → pallets at 200.0kg; sub-one-box cap → all unpacked).
- **Load-bearing (direct, integer dims).** 0 `max_load_on_top` violations across
  25 cases; fragile (mlot=0) boxes never carry weight.
- **Box conservation.** 0 violations across 66 runs — every box placed exactly
  once or unpacked; no duplication, no loss, no phantom boxes.
- **Polish (LS/PR/LNS + block-aware ops).** Never produces invalid output, never
  regresses fitness (accept-only), deterministic.
- **max_overhang.** Correctly bounded (100 boxes, 0 violations); overhang-off
  correctly ignores `pallet.max_overhang`.

---

## 3. Confirmed defects (deduplicated by root cause)

All independently reproduced by a second adversarial agent and (for the two
blockers) by hand by the orchestrator.

### 🔴 BLOCKER 1 — Integer-grid decode vs float-geometry validation

The decoders quantize every dimension to `int(round(...))` (box dims →
`int64 dims_all`, `brkga_v3_fast.py:337-339`; pallet dims → `int(round(...))`,
`driver.py:176-178`, `dispatch.py:179-181`). The reconstructed `Placement`
stores the **original float `Box`**, and the independent validator recomputes
geometry from those floats. So the engine packs at integer pitch while the world
is float: butted neighbors overlap by exactly the rounding delta (≤0.5 units).

**Independently reproduced (orchestrator, Docker OMP=1):**

| Input | Validator errors | Worst overlap |
|---|---:|---:|
| integer 100³ × 64 | **0** | 0.000u |
| fractional 100.49³ × 64 | **210** | **0.490u** |
| realistic 333.33 × 250.7 × 124.9 × 33 | **38** | 0.330u |

Across the agent's 21 fractional-input solves, **85.7% (18/21) failed the
validator**; all 10 integer-rounded twins were clean (0%). Overlap penetration
(~0.49u) is ~500,000× the validator EPS — genuine geometry, not a tolerance
artifact. **Sub-defect:** any dimension < 0.5 rounds to integer **0** → boxes
collapse to zero-volume points (50 metre-scale 0.4³ boxes → 1,225 pairwise
overlaps). This is the dominant real-world blocker: real inputs (mm/cm with
decimals, inch→mm conversions, unit-mismatched JSON) are routinely non-integer.

### 🔴 BLOCKER 2 — Does not scale to 5,000 boxes; time budget is soft

The intended service limit is **5,000 boxes**; the largest previously tested
case was **200**.

- **Constrained full solve at N=5,000 never completes** — >60 min without
  finishing a single generation-0 decode (killed). Peak RSS flat ~353 MB (no OOM,
  no int64 overflow — the failure is *time*, not memory).
- **`time_limit_s` is a soft, per-generation budget** (`driver.py:411`). When one
  generation costs more than the budget, the deadline blows past unbounded:
  constrained **N=2,000 ran 334s against a 120s budget (+178%)**; geometric
  N=2,000 ran +14%; N=1,000 +9–29% depending on workload.
- **Root cause is mode/seed-dependent, not the Cython kernels.** Block modes
  (4/5) are cheap even at scale (`decode_chromosome` mode-4 @ N=5,000 = **0.006s**
  measured). The O(N²) cost lives in the **per-box modes (0/1/2/3)** and the
  **v2-seed `PalletPacker`** (auto-enabled on constrained workloads), plus a full
  `PackResult` rebuild per improving chromosome.

Realistic ceiling for the as-shipped multi-decoder driver within a sane budget is
**a few hundred boxes**, not 5,000.

### 🟠 HIGH — support_ratio / stability silently disabled on weightless input

`driver.py:190-191`: `support_ratio_value = config.support_ratio if
has_constraints else 0.0`. `has_constraints` is **False** for any workload with
no finite `max_load_on_top`, no `requires_full_support`, and infinite
`pallet.max_weight` — i.e. **every weightless input**, including the entire BR
benchmark. On that path the decoder gets `support_ratio=0.0` and places
partially-supported ("floating") boxes that the validator (running at
`config.support_ratio`=0.8–1.0) rejects. **BR1 n=10: 91 validator errors, 8/10
instances "dirty"** (worst BR1#2 = 34). A real user who submits boxes *without
weights* silently gets physically unstable stacks. (For academic BR scoring,
which counts only volume, this is harmless — which is why it was never caught.)

### 🟠 HIGH — Block decoders (modes 4/5) violate load-bearing & support on *any* constrained workload

**Sharpened by follow-up investigation — this is not a scale-only issue.** On a
realistic mixed constrained workload (8 SKUs, ~11% fragile, integer dims) the
default config (`n_modes=6`) produced validator errors at **every** N from 200 up:

| N | `n_modes=4` (no blocks) | `n_modes=6` (default) |
|---:|---|---|
| 200 | 9 pallets, **0 errs** | 4 pallets, **18 errs** (2 floating + 16 load-bearing) |
| 500 | 28 pallets, **0 errs** | 9 pallets, **46 errs** (8 floating + 38 load-bearing) |

Disabling the block modes (`n_modes≤4`) yields validator-clean output, isolating
the defect to **modes 4 & 5** (`decode_blocks_njit_mode_cstr` + the mode-5
delegation). The errors include the serious one: **fragile boxes
(`max_load_on_top=0`) carrying 12 kg** — physically crushed.

**Root cause (read from source):** `precompute_box_dims_and_sku`
(`precompute.py:43`) keys SKUs on `(length, width, height, weight, rotations)` —
**`max_load_on_top` and `requires_full_support` are *not* in the key**. So fragile
and sturdy boxes of identical size merge into one composite block and get stacked
into an n_x×n_y×n_z grid; the block enumerator caps neither stack height by load
capacity nor checks intra-block support against `support_ratio`.

**This cannot be fixed by disabling block modes** — they are the source of 2–3×
the packing density (N=500: 9 pallets vs 28). The fix is to make block-building
constraint-aware: put `mlot`/`rfs` in the SKU key (so fragile items form their
own non-stackable blocks) and enforce per-box load + support inside block
decomposition. The 10 industry cases stayed clean only because their specific SKU
geometries happened to avoid stackable-fragile collisions.

### 🟠 HIGH — `Box.group` co-location never implemented

`models.py:70-71` documents: *"boxes in the same group must end up on the same
pallet."* A grep across the entire `_brkga_core` package finds **zero reads of
`Box.group`** — no decoder, dispatch, driver, precompute, or chromosome init
encodes it. IND6 ("LTL groupage") silently splits groups across pallets (group of
18 spanning 2 pallets even with unlimited pallets). The validator is also blind
to it. A documented feature is non-functional.

### 🟡 MEDIUM — `max_pallets` function-arg leaked on v2-seed path

`brkga_pack_v35(max_pallets=K)` is not propagated to the v2 seed packer
(`driver.py:259-261` builds `PalletPacker(pallet, config)`, which honors only
`config.max_pallets`). When the v2 result wins the fitness comparison or is the
fallback, the cap is breached: IND2 with `max_pallets=1/2/3` all returned **4
pallets**. Only bites on constrained workloads (where v2-seed auto-enables) when
`config.max_pallets` is looser than the function arg.

### 🟡 MEDIUM — Malformed input: silent-wrong / no boundary validation

`Box`/`Pallet` perform no validation. Found: **NaN `pallet.max_weight` silently
disables the weight cap** (`math.isinf(NaN)` is False → cap becomes NaN → every
comparison False; a 999,999 kg box packs validator-clean). **Negative-dimension
boxes** produce validator-clean but physically out-of-bounds placements
(box spanning x∈[−30,0]). **NaN/inf dims** crash with a raw error. For a public
API taking arbitrary JSON this needs an input-validation gate.

### ⚪ LOW / documented limitations

- **Load model is direct-supporter-only, not recursive/cumulative** despite the
  `PackerConfig` docstring. 15/25 stacking families had *cumulative* physical
  load exceeding `max_load_on_top` while direct contact stayed within — a
  real-world tall-stack stability gap.
- **CoG envelope has a first-box/new-bin exemption** (a single heavy box can land
  off-envelope above the load gate). Judged faithful to the canonical algorithm;
  `validate()` does not check CoG at all, so breaches are invisible. A stability
  consideration for real loads.
- **Polish accept criterion is volume-only (constraint-blind)** — latent; does
  not manifest through the driver path.
- **Zero-dimension box is "placed" rather than rejected** — benign (Cython
  degrades gracefully to a 1×1×1 block).

---

## 4. Timing & throughput (controlled, OMP=8 unless noted)

**Batch decode throughput** (pop=600, n≈100–120):

| Mode | decodes/sec |
|---|---:|
| mode 4 (dynamic blocks) | **121,400** |
| mode 5 (precomputed blocks) | 56,700 |
| mode 3 (layer — per-box) | 1,900 |

**OpenMP scaling** (layer batch, pop=600, n=120) — near-linear:

| OMP threads | layer batch | speedup | cstr0 batch | speedup |
|---:|---:|---:|---:|---:|
| 1 | 1813 ms | 1.00× | 818 ms | 1.00× |
| 2 | 958 ms | 1.89× | 423 ms | 1.93× |
| 4 | 571 ms | 3.17× | 248 ms | 3.30× |
| 8 | 305 ms | **5.95×** | 139 ms | **5.88×** |

**Hot-loop generation time** (`decode_population_fitness`, multi-decoder, pop=100):
N=50 → 13.8 ms (7,227 dec/s); N=200 → 236 ms (423); N=2,000 → 389 ms (257).
Array setup negligible (6.6 ms @ N=2,000).

**Scale ceiling** (full driver):

| N | workload | wall (budget) | overrun | placed | validator errs |
|---:|---|---|---:|---:|---:|
| 1,000 | geometric | 21.8s (20s) | +9% | 1000/1000 | 14 (floating) |
| 2,000 | geometric | 22.7s (20s) | +14% | 2000/2000 | 47 (floating) |
| 1,000 | constrained | 116s (90s) | +29% | 1000/1000 | 0 |
| 2,000 | constrained | 334s (120s) | +178% | 2000/2000 | 3 (floating) |
| 5,000 | constrained | >3600s | ∞ | — | did not complete |

**Real-world per-instance wall-clock + validity** (15s budget, OMP=8):

*BR (geometric, single pallet)* — utilization is competitive, but the validator
flags floating placements on **11 of 12** instances (defect: weightless ⇒
support dropped):

| Set | util range | validator errs (per inst.) | clean |
|---|---|---|---:|
| BR1 (#1–3) | 88.1–92.6% | 3, 34, 0 | 1/3 |
| BR3 (#1–3) | 92.8–95.2% | 18, 31, 24 | 0/3 |
| BR5 (#1–3) | 92.7–93.5% | 17, 19, 28 | 0/3 |
| BR7 (#1–3) | 92.1–92.6% | 24, 25, 26 | 0/3 |

*Industry (constrained, multi-pallet)* — **10/10 validator-clean, 0 unpacked**;
the constrained path is correct at these scales. One budget overrun:

| Case | N | wall (15s budget) | pallets | errs |
|---|---:|---:|---:|---:|
| IND1 E-commerce | 120 | 9.3s | 9 | 0 |
| IND2 Pharma (fragile) | 200 | **26.4s (+76%)** | 4 | 0 |
| IND3 Furniture | 15 | 4.8s | 8 | 0 |
| IND4 Beverage | 44 | 4.9s | 1 | 0 |
| IND5 Retail | 150 | 9.6s | 2 | 0 |
| IND6 LTL groupage | 43 | 5.0s | 1 | 0¹ |
| IND7 Electronics | 150 | 12.7s | 8 | 0 |
| IND8 Automotive | 60 | 15.0s | 4 | 0 |
| IND9 Document | 100 | 7.0s | 2 | 0 |
| IND10 Cold-chain | 80 | 6.1s | 2 | 0 |

¹ IND6 is validator-clean but its `group` constraint is silently ignored (HIGH
defect above); the validator does not check grouping. IND2 (+76%) shows the soft
budget bites even at N=200 once the v2-seed packer runs.


The throughput claims from the prior audit hold (parallel decode ~6× on 8 cores).
The new finding is that **throughput does not translate to scalability**: the
per-box modes and v2-seed are O(N²), and the soft time budget means a service
cannot rely on `time_limit_s` as an SLA for N≥1,000.

---

## 5. Implications for the planned public service

The service plan (open, no-login, async job + polling, Redis, **5,000-box
limit**, time budget "much bigger than 60s", multi-worker) needs these resolved
*before* build:

1. **Cap input far below 5,000** (≈300–500 with current code) **or** invest in
   sub-quadratic placement + drop v2-seed at scale + block-only mode for large N.
   5,000 is not achievable as-is.
2. **Fix the integer/float gap** — quantize all inputs to a common integer grid
   *and validate on that grid* (or make the engine float-native). Without this,
   real customer dimensions yield overlapping packings.
3. **Make `time_limit_s` a hard deadline** (check budget inside the decode/build
   loop), so the worker's job timeout is meaningful.
4. **Enforce stability regardless of weights** — decouple `support_ratio` from
   `has_constraints`, or the service produces unstable packs for weightless input.
5. **Validate input at the API boundary** — reject non-positive / non-finite /
   zero dims and NaN caps with a 4xx, not a silent-wrong 200.
6. **Decide on `group` and `max_pallets`** — implement group co-location and fix
   the v2-path cap leak, or document them as unsupported and remove from the API.

None of these are Cython-port regressions — the port is a faithful reproduction.
They are pre-existing algorithm/engine gaps that the academic + small-integer
test surface never exposed, and they matter precisely because a public service
takes arbitrary real-world input.

---

## 6. Reproduction

All probes live in `scripts/_verify/` (one per dimension, `verify_*.py`, plus the
adversarial `recheck_*.py` counter-probes and `timing_suite.py`). Each runs as:

```
docker run --rm -e OMP_NUM_THREADS=1 -e PYTHONHASHSEED=0 \
    -v "$(pwd)":/app -w /app pallet-packer:dev python scripts/_verify/<probe>.py
```

Headline repros: `verify_float_int_dims.py` (blocker 1), `verify_scale_5000.py`
(blocker 2), `verify_edge_geom.py` (weightless stability),
`verify_multipallet_conservation.py` (group / max_pallets).
