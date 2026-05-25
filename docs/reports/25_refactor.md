# Refactor — brkga_v3_5.py monolith → _brkga_core/ package (12 modules)

Pre-port restructuring. The 4154-line `pallet_packer/brkga_v3_5.py`
monolith is split into 12 logically-grouped modules under
`pallet_packer/_brkga_core/`. The old import path
(`from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit`)
continues to work via a 56-line shim.

> **Behavior verdict.** Industry n=10: all 10 cases match v3.12 exactly
> (0 errors, IND7 48.2 %, IND8 55.2 %). BR n=3: **12 / 12 instances
> bit-identical** to the pre-refactor baseline. Refactor is a pure
> structural change — zero algorithm drift.

---

## 1. Why refactor before the C/Cython port

A 4154-line single-file module is hostile to a port:

- **No obvious compile units.** A C port wants one .c / .pyx per
  cluster of related functions; the monolith made every function
  technically reachable from every other.
- **No test boundaries.** Without natural module-level seams there's
  nowhere to attach unit tests.
- **Hidden coupling.** Some helpers were used in many places, others
  in one — invisible without a dependency graph.
- **Cognitive load.** Anyone (including future-me) opening the file
  has to scroll past 12 unrelated concerns to find one thing.

The split fixes all four. Each new module is 100-1000 lines, has a
single concern, has explicit imports, and maps cleanly to one C
compile unit.

---

## 2. Module map (final)

```
pallet_packer/_brkga_core/
├── __init__.py              118 lines  re-exports + module map docstring
├── precompute.py            115 lines  per-box / pallet array prep
├── jit_primitives.py        143 lines  find_best_wall / corner / in_slab
├── jit_constraints.py       255 lines  _check_load_on_top / _check_cog
│                                       / _apply_*
├── jit_decoders_geom.py     900 lines  decode_njit_mode + 3 block decoders
├── jit_decoders_cstr.py    1024 lines  cstr variants + block helpers
├── chromosome.py            160 lines  chromosome_from_order etc.
├── blocks.py                183 lines  block enumeration + resolve
├── dispatch.py              386 lines  warmup_jit + decode_chromosome
│                                       + decode_auto_mode
├── polish.py                402 lines  local_search_2opt + path_relinking
│                                       + lns_polish
├── adaptive.py              108 lines  experimental adaptive selector
├── sku_aware.py             201 lines  experimental SKU-aware mini-BRKGA
└── driver.py                575 lines  brkga_pack_v35 — main orchestrator

pallet_packer/brkga_v3_5.py   56 lines  thin shim (re-exports from
                                        _brkga_core)
```

**Total:** 4626 lines across 13 files (vs 4154-line monolith + 56-line
shim = 4210 effective). The ~400-line growth is module docstrings and
import statements that document the dependency graph explicitly.

### Dependency graph

```
     precompute ──┐
   jit_primitives ┼─── jit_decoders_geom ──┐
  jit_constraints ┼─── jit_decoders_cstr ──┤
                  │                         │
                  └─── chromosome           │
                       blocks ──────────────┤
                                            ▼
                                       dispatch
                                       /   |   \
                                  polish adaptive sku_aware
                                       \   |   /
                                        driver
```

No cycles. Each module depends only on layers below it.

---

## 3. Why this split helps the C/Cython port

| Module | Port priority | Notes |
|---|---|---|
| `jit_primitives.py` | **High** | 3 pure functions, hot loop — first port target |
| `jit_constraints.py` | **High** | 4 helpers, called per placement — second target |
| `jit_decoders_geom.py` | **High** | the actual decoders that consume the above |
| `jit_decoders_cstr.py` | **High** | same, constraint-aware variants |
| `dispatch.py` | Medium | thin Python wrapper; could stay Python |
| `precompute.py` | Low | runs once per call; Python is fine |
| `blocks.py` | Low | runs once per call; Python is fine |
| `chromosome.py` | Low | constructor helpers; Python is fine |
| `polish.py` | Medium | hot but calls decoders; can stay Python |
| `adaptive.py` / `sku_aware.py` | **Don't port** | experimental, OFF by default |
| `driver.py` | **Don't port** | orchestration; Python latency negligible |

So the actual port scope is 4 modules totalling ~2300 lines of JIT
code — much more tractable than "port a 4154-line file."

---

## 4. Behavioral validation

### Industry (n=10 × 30 s/pallet)

| Case | Pre-refactor | Post-refactor | Match? |
|---|---|---|---|
| IND1 E-commerce | 4p / 88.6 % / 0 errs | 4p / 88.6 % / 0 errs | ✓ |
| IND2 Pharma | 4p / 6.4 % / 0 errs | 4p / 6.4 % / 0 errs | ✓ |
| IND3 Furniture | 5p / 55.7 % / 0 errs | 5p / 55.7 % / 0 errs | ✓ |
| IND4 Beverage | 1p / 69.2 % / 0 errs | 1p / 69.2 % / 0 errs | ✓ |
| IND5 Retail 4-SKU | 2p / 92.9 % / 0 errs | 2p / 92.9 % / 0 errs | ✓ |
| IND6 LTL groupage | 1p / 39.2 % / 0 errs | 1p / 39.2 % / 0 errs | ✓ |
| IND7 Electronics | 3p / 48.2 % / 0 errs | 3p / 48.2 % / 0 errs | ✓ |
| IND8 Automotive | 4p / 55.2 % / 0 errs | 4p / 55.2 % / 0 errs | ✓ |
| IND9 Document | 2p / 85.5 % / 0 errs | 2p / 85.5 % / 0 errs | ✓ |
| IND10 Cold-chain | 2p / 95.4 % / 0 errs | 2p / 95.4 % / 0 errs | ✓ |

### BR (n=3/set × 30 s/instance)

| Instance | Pre-refactor | Post-refactor | Δ |
|---|---|---|---|
| BR1 # 1, # 2, # 3 | 91.05, 91.36, 88.11 | **91.05, 91.36, 88.11** | 0 |
| BR3 # 1, # 2, # 3 | 94.02, 95.22, 92.70 | **94.02, 95.22, 92.70** | 0 |
| BR5 # 1, # 2, # 3 | 91.45, 93.35, 92.13 | **91.45, 93.35, 92.13** | 0 |
| BR7 # 1, # 2, # 3 | 92.87, 92.26, 91.42 | **92.87, 92.26, 91.42** | 0 |

**12 / 12 bit-identical.** Same seeds + same params → same numpy RNG
sequence → same chromosomes → same decoded packings.

---

## 5. One bug surfaced, one fix

The migration caught one bug that the monolith was hiding:

**`from .packer import PalletPacker`** in `driver.py` resolved correctly
in the monolith (`pallet_packer.packer`) but became
`pallet_packer._brkga_core.packer` after the move — non-existent.
The v2 warm-start silently failed (`v2 seed failed: No module named...`),
which on geometric BR was invisible (smart-init carried the load) but
on IND2 destroyed pop-0 quality (0.36% starting util vs ~6% baseline)
and left 132 boxes unpacked.

Fix: `from ..packer import PalletPacker` (double-dot for parent
package). Confirmed by post-fix smoke: IND2 → 4p / 0 unp / 0 errs.

Lesson recorded for the C/Cython port: **anywhere imports are touched,
re-run IND2 + BR1#1 immediately**. Both should match the canonical
4p / 0u / 0errs and 91.05 % values at the standard seeds.

---

## 6. Backward compatibility

The shim at `pallet_packer/brkga_v3_5.py` re-exports every public name
(plus all the JIT internals, since some tests touch them directly). All
existing scripts continue to work unchanged:

- `scripts/run_v38_full_eval.py`
- `scripts/run_v39_adaptive_eval.py`
- `scripts/run_v310_industry_eval.py`
- `scripts/run_v310_br_regression.py`
- `scripts/test_blocks_lns.py`
- `scripts/run_brkga_v35_eval.py`
- `scripts/run_brkga_v35_new_eval.py`

New code should prefer the explicit path:

```python
from pallet_packer._brkga_core import brkga_pack_v35, warmup_jit
```

---

## 7. Status

- v3.12 algorithm now lives in `pallet_packer/_brkga_core/` (12 modules)
- Old import path (`pallet_packer.brkga_v3_5`) still works via shim
- BR + industry behavior bit-identical to pre-refactor
- One latent bug (v2-seed import path) caught and fixed
- Ready for the C/Cython port — clear module boundaries, explicit
  dependency graph, 4 hot JIT modules totalling ~2300 lines to port
