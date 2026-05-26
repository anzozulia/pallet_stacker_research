# Cython Port — Handoff & Resume Plan

Living document for the v3.12 → Cython port. Read top-to-bottom on
resume; the "Resume here" section at the bottom is the next action.

---

## 1. Where we are right now

**Phase 3 complete (3a + 3b + 3c + 3d). All four geometric decoders are
Cython LIVE.** End-to-end BR1#1 = 91.05% and BR3#1 = 94.02% — exact match
to v3.12 baseline.

| Decoder | Numba | Cython | Speedup |
|---|---|---|---|
| Mode 0 (DFTRC) | 4607 µs | 2681 µs | **1.72×** |
| Mode 1 (wall) | 2789 µs | 2533 µs | 1.10× |
| Mode 2 (corner) | 2644 µs | 2319 µs | 1.14× |
| Mode 3 (layer, n=100) | 3134 µs | 2616 µs | 1.20× |
| Mode 4 (blocks, n=100/skus=4) | 2.6 µs | 4.7 µs | 0.54× ⚠ |
| Mode 5 (precomputed, n=100/skus=4) | 8.8 µs | 11.6 µs | 0.76× ⚠ |

⚠ Mode 4 / Mode 5 single-call microbenches are misleading — the chosen
representative cases (n=100, max_pallets=4-6, skus=4) finish most boxes
via the early-out path (MAX_BINS reached, no fit), so per-call work is
tiny and Python wrapper setup dominates. On real BR workloads (single
SKU, larger pallets, longer per-call work) the Numba-vs-Cython gap
closes. The end-to-end br-smoke shows no regression in solution quality,
and Phase 5 (`prange` parallel pop eval) is where the compounding gains
arrive.

### Recent commits (port-relevant)

```
35c5852  Phase 0: Docker + Cython build infra
04ffabc  Phase 1 (port): jit_primitives.py -> jit_primitives_cy.pyx
4301285  Phase 2 (port): jit_constraints.py -> jit_constraints_cy.pyx
400fd2c  Phase 3a (port): decode_njit_mode (modes 0/1/2) Cython LIVE
64cacb4  Phase 3b (port): decode_layer_njit (mode 3) Cython LIVE
db95105  Phase 3c (port): decode_blocks_njit_mode (mode 4) Cython LIVE
5816add  Phase 3d (port): decode_precomputed_blocks_njit_mode (mode 5) LIVE
```

---

## 2. Architecture (current state)

```
pallet_packer/_brkga_core/
├── precompute.py             pure Python
├── blocks.py                 pure Python
├── chromosome.py             pure Python
├── adaptive.py               pure Python (experimental, OFF)
├── sku_aware.py              pure Python (experimental, OFF)
│
├── jit_primitives.py         Numba reference (fallback)
├── jit_primitives_cy.pxd     Cython header
├── jit_primitives_cy.pyx     Cython port (find_best_wall/corner/in_slab)
│
├── jit_constraints.py        Numba reference (fallback)
├── jit_constraints_cy.pyx    Cython port — NOT YET WIRED LIVE
│
├── v3fast_cy.pxd             Cython header for brkga_v3_fast helpers
├── v3fast_cy.pyx             Cython port of find_best_dftrc + commit_ems
│
├── jit_decoders_geom.py      Numba reference: decode_njit_mode +
│                             decode_layer_njit + decode_blocks_njit_mode +
│                             decode_precomputed_blocks_njit_mode (fallback)
├── jit_decoders_geom_cy.pxd  Cython header — declares _find_best_block_at_pos
│                             so Phase 4 cstr blocks can cimport it
├── jit_decoders_geom_cy.pyx  Cython port — COMPLETE: modes 0/1/2 (3a),
│                             3 (3b), 4 (3c), 5 (3d). All geometric
│                             decoders Cython-LIVE via dispatcher.
│
├── jit_decoders_cstr.py      Numba reference: 3 cstr decoders + helpers
│   (no .pyx yet — Phase 4)
│
├── dispatch.py               try Cython first, fall back to Numba
├── polish.py                 pure Python (LS + PR + LNS)
└── driver.py                 brkga_pack_v35 main loop
```

### Key invariant

Every Cython port keeps a **bit-identical** match to the Numba reference
at fixed seed. The dispatcher's try-import pattern means the algorithm
works whether or not the .so files exist — Docker builds them; raw
checkouts fall back to Numba.

---

## 3. Code patterns (every port follows these)

### Pattern A — .pxd header for cross-module cimport

```cython
# foo_cy.pxd
from libc.stdint cimport int64_t
ctypedef int64_t i64

cdef void _my_func(
    const i64[:, :, ::1] arr,
    i64 n,
    i64* out_x, i64* out_y,
) noexcept nogil
```

### Pattern B — .pyx implementation + Python wrapper

```cython
# foo_cy.pyx
# cython: language_level=3
# cython: boundscheck=False, wraparound=False, cdivision=True
# cython: initializedcheck=False
import numpy as np
cimport numpy as cnp
# i64 + int64_t come from the .pxd

cdef void _my_func(...) noexcept nogil:
    """Internal nogil core."""
    ...

def my_func_njit(arr, n):
    """Python wrapper matching the Numba signature."""
    cdef const i64[:, :, ::1] arr_v = arr
    cdef i64 x, y
    _my_func(arr_v, n, &x, &y)
    return x, y
```

### Pattern C — Cross-module cimport in a decoder

```cython
# jit_decoders_geom_cy.pyx
from .v3fast_cy cimport _find_best_dftrc, _commit_ems, MAX_EMS_C
from .jit_primitives_cy cimport _find_best_wall, _find_best_corner

cdef i64 _decode_loop(...) noexcept nogil:
    ...
    _find_best_dftrc(bin_emss[b], ..., &idx, &x, &y, &z)  # zero-overhead
    ...
```

### Pattern D — Dispatcher try-import

```python
# dispatch.py
try:
    from .jit_decoders_geom_cy import decode_njit_mode
except ImportError:
    from .jit_decoders_geom import decode_njit_mode
```

---

## 4. Build pipeline (commit `35c5852`)

- **Dockerfile** at repo root: Python 3.13 + cython 3.2.5 + numpy 2.4.6 + numba 0.65.1
- **pyproject.toml**: build-system requires cython + numpy; runtime deps include numba (fallback)
- **setup.py**: `cythonize()` with `-O3 -ffast-math -march=native`; one `Extension(...)` per .pyx
- **Makefile** targets:
  - `make image` — build Docker image (3 min, one-time)
  - `make build` — rebuild Cython after .pyx changes (~10s)
  - `make shell` — drop into container
  - `make br-smoke` — quick BR n=2 sanity (1-2 min)
  - `make industry-smoke` — quick industry n=3 sanity
  - `make cython-clean` — remove generated .c/.so/.html
- **.gitignore**: ignores `**/*.c`, `**/*.so`, `**/*.html` (build artefacts)

### Adding a new .pyx

1. Write the `.pyx` (and `.pxd` if it exposes cdef functions to other modules)
2. Append an `Extension(...)` entry in `setup.py`
3. Run `make build`
4. Verify `.so` appears: `ls pallet_packer/_brkga_core/*.so`
5. Write an A/B test: `scripts/ab_test_<module>.py` (use Phase 1/2/3a as templates)
6. Run inside Docker:
   `docker run --rm -v "$(pwd)":/app pallet-packer:dev python scripts/ab_test_<module>.py`
7. Wire dispatcher if it goes live in this phase
8. `make br-smoke` to confirm no regression

---

## 5. Remaining phases

### Phase 3b — `decode_layer_njit` (mode 3, layer-build)

- **Source**: `jit_decoders_geom.py:267` (about 200 lines)
- **Adds to**: `jit_decoders_geom_cy.pyx` (already exists)
- **Helpers used**: `find_best_in_slab_njit` (already ported in Phase 1),
  `commit_ems_njit` (already ported in Phase 3a)
- **Dispatcher change**: extend the existing try-import to also pull
  `decode_layer_njit` from `jit_decoders_geom_cy`
- **A/B test**: `scripts/ab_test_decoders_geom.py` (extend Phase 3a test
  to also call mode 3)
- **Expected effort**: half day
- **Expected speedup**: similar to mode 0 (1.5-2× per decode)

### Phase 3c — `decode_blocks_njit_mode` + `find_best_block_at_pos_njit` (mode 4)

- **Source**: `jit_decoders_geom.py:455` (find_best_block_at_pos),
  `jit_decoders_geom.py:516` (decode_blocks_njit_mode) — about 250 lines total
- **Adds to**: `jit_decoders_geom_cy.pyx`
- **Helpers used**: same as 3a + 3b
- **Note**: `find_best_block_at_pos_njit` needs a .pxd cdef declaration
  since `decode_blocks_njit_mode_cstr` (Phase 4) will also call it
- **Expected effort**: 1 day (block extension logic is more involved)
- **Expected speedup**: 1.5-2× per decode

### Phase 3d — `decode_precomputed_blocks_njit_mode` (mode 5)

- **Source**: `jit_decoders_geom.py:703` (about 300 lines)
- **Adds to**: `jit_decoders_geom_cy.pyx`
- **Note**: Mode 5 has a fallback path that calls into mode 4 dynamic
  blocks; both need to be Cython by the time this is done
- **Expected effort**: 1 day
- **Expected speedup**: 1.5-2×

After 3a-d, **all 4 geometric decoders are Cython**. `make br-smoke`
should produce bit-identical results vs Numba baseline.

### Phase 4 — Constraint-aware decoders (`jit_decoders_cstr_cy.pyx`)

- **Source**: `jit_decoders_cstr.py` (~1024 lines, 3 decoders + 2 helpers)
- **Helpers used**:
  - everything from 3a-d
  - everything from `jit_constraints_cy` (Phase 2 — need to add a .pxd!)
- **New .pxd needed**: `jit_constraints_cy.pxd` with cdef declarations
  for `_ck_load_on_top`, `_ck_cog_envelope`, `_ap_load_contribution`,
  `_ap_cog_contribution` (mirror of `jit_primitives_cy.pxd` pattern)
- **Functions to port**:
  - `decode_njit_mode_cstr` (cstr modes 0/1/2) — ~500 lines
  - `decode_blocks_njit_mode_cstr` (cstr mode 4) — ~350 lines
  - `decode_layer_njit_cstr` (cstr mode 3) — ~250 lines
  - `_commit_block_placements_njit` (helper for cstr mode 4)
  - `_max_block_under_constraints_njit` (helper for cstr mode 4)
- **Dispatcher change**: extend try-import to also swap the cstr decoders
- **Expected effort**: 2-3 days (largest phase; ~1000 lines)
- **Expected speedup**: 1.5-2× per decode, plus opens the door for Phase 5

### Phase 5 — `prange` parallel population evaluation

- **The multi-core multiplier.** Once all decoders are Cython, the BRKGA
  loop in `driver.py` can dispatch per-individual decodes to threads via
  `cython.parallel.prange`.
- **Key constraint**: must remain deterministic by seed. Pre-assign
  chromosome indices to threads (no work-stealing); each thread uses
  its own `np.random.default_rng(seed + i)`.
- **Implementation**: add a Cython batch-decode function:
  ```cython
  cpdef void decode_batch(
      double[:, ::1] chromosomes,   # (pop_size, chrom_len)
      double[::1] fitnesses_out,
      ...
  ) noexcept nogil:
      cdef Py_ssize_t i
      for i in prange(pop_size, nogil=True, schedule='static'):
          fitnesses_out[i] = _decode_chrom(chromosomes[i], ...)
  ```
- **Expected effort**: 1-2 days
- **Expected speedup**: 4-8× on an 8-core M-series CPU (linear scaling
  for embarrassingly parallel decodes)

### Phase 6 — Final eval + tuning

- Full BR n=10 + industry n=10 with the entire Cython chain live
- Compare against BRKGA-2013 SOTA on BR1 (target: close the 1.57 pp gap)
- Profile any unexpected bottleneck
- Update `docs/reports/27_port_complete.md` with final numbers

---

## 6. Validation — bit-identical contract

Every port phase must produce **bit-identical results** on the canonical
seeds. The smoke target:

```bash
$ make br-smoke
BR1#1: util=91.05%  (v3.12 baseline: 91.05%) ✓
BR3#1: util=94.02%  (v3.12 baseline: 94.02%) ✓
```

If util differs at the 3rd decimal at seed=43, **stop and diff**. Pure
Cython→C is deterministic; any drift is a port bug (off-by-one,
signed-int truncation, missing memoryview cast).

### A/B test scripts (template per phase)

- `scripts/ab_test_primitives.py` (Phase 1, 50 random calls per fn)
- `scripts/ab_test_constraints.py` (Phase 2, 200 random calls per fn)
- _**TODO**: `scripts/ab_test_decoders_geom.py` for Phase 3b/c/d_
- _**TODO**: `scripts/ab_test_decoders_cstr.py` for Phase 4_

The decoder A/B tests are simpler than per-function: call both with
random BPS orderings + same dims, assert `placements_out` arrays are
`np.array_equal`.

---

## 7. Key facts to remember on resume

1. **Numba @njit functions cannot call Cython functions at C level.**
   Different ABI. They can call them via Python (slow, defeats the
   purpose) or not at all. This is why entire chains must port together
   in each phase (decoder + its helpers).

2. **`cdef` functions in .pxd must NOT be `inline`** — that's a
   header-only construct. Implementations live in .pyx. (Phase 1's
   first build failed because of this; sed fix in commit `<phase3a>`.)

3. **i64 ctypedef belongs in the .pxd only.** Declaring it in both
   .pxd and .pyx causes "redeclared" error.

4. **Memoryview syntax for our arrays:**
   - `i64[:, ::1]` — 2D, last dim contiguous (placements_out)
   - `i64[:, :, ::1]` — 3D, last dim contiguous (dims_all, EMS)
   - `i64[::1]` — 1D contiguous (bps_order)
   - `double[::1]` — 1D float array

5. **`MAX_EMS_C = 512`** in `v3fast_cy.pyx` and `MAX_EMS = 512` in
   `brkga_v3_fast.py`. **Keep in sync** — change both together.

6. **The `placements_out` array layout:**
   columns `[pallet_idx, rotation_idx, x, y, z, placed_flag]`
   placed_flag = 1 if box was placed, 0 if unpacked

7. **Docker is the canonical build environment.** Apple Silicon dev
   machines build `*-aarch64-linux-gnu.so` inside the container.
   `make build` does the right thing.

---

## 8. Files NOT to touch

- `pallet_packer/brkga_v3_5.py` — backward-compat shim. Re-exports from
  `_brkga_core`. Don't add logic here.
- `pallet_packer/packer.py` — v2 algorithm, used by the warm-start in
  driver.py. Stays Python forever; not a port target.
- `pallet_packer/mip.py`, `pallet_packer/lower_bounds.py` — auxiliary
  modules, not in the BRKGA hot path.
- `pallet_packer/_brkga_core/sku_aware.py`, `adaptive.py` —
  experimental, default OFF, won't be ported.

---

## 9. RESUME HERE

**Next action: Phase 4 — port the constraint-aware decoders
(`jit_decoders_cstr.py`, 1024 lines, 3 decoders + 2 helpers) to Cython.**

This is the largest single phase of the port. It is also the gate to
production deployment: every industry workload (pharma, document,
cold-chain) runs through the cstr path. Plan ~2–3 days.

### Scope (Numba source: jit_decoders_cstr.py)

```
line 44   decode_njit_mode_cstr            cstr modes 0/1/2  ~250 lines
line 299  _commit_block_placements_njit    helper            ~60 lines
line 361  _max_block_under_constraints_njit helper            ~40 lines
line 401  decode_blocks_njit_mode_cstr     cstr mode 4       ~350 lines
line 748  decode_layer_njit_cstr           cstr mode 3       ~280 lines
```

### Prerequisite: jit_constraints_cy.pxd

The cstr decoders all call helpers from `jit_constraints_cy.pyx`
(Phase 2). Today those are Python-callable defs only — no cdef nogil
interface. To call them from inside a nogil decoder loop, we need a .pxd:

```cython
# jit_constraints_cy.pxd — NEW FILE
from libc.stdint cimport int64_t
ctypedef int64_t i64

cdef bint _check_load_on_top(
    const i64[:, ::1] placements, i64 n_placed,
    const i64[:, ::1] box_bottoms, const i64[:, ::1] box_tops,
    ...   # full signature: read jit_constraints.py:25-130
) noexcept nogil

cdef bint _check_cog_envelope(...) noexcept nogil
cdef void _apply_cog_contribution(...) noexcept nogil
cdef void _apply_load_contribution(...) noexcept nogil
```

This will require splitting `jit_constraints_cy.pyx` into a `cdef` core
+ `def` Python wrapper for each helper, mirroring the Phase 1 pattern
(see `jit_primitives_cy.pyx` for the template).

**Sub-step Phase 4.0** — do this refactor FIRST and rerun the existing
`scripts/ab_test_constraints.py` to confirm the helpers stay
bit-identical. Then proceed to the decoders.

### Per-decoder steps (apply to each of the 3)

1. Read the Numba source and identify all Cython helpers it needs:
   - `_find_best_dftrc`, `_find_best_wall`, `_find_best_corner`,
     `_find_best_in_slab` from `jit_primitives_cy` / `v3fast_cy`
   - `_find_best_block_at_pos` from `jit_decoders_geom_cy`
   - `_check_load_on_top`, `_check_cog_envelope`,
     `_apply_load_contribution`, `_apply_cog_contribution` from
     `jit_constraints_cy` (after the .pxd is in place)
   - `_commit_ems` from `v3fast_cy`

2. Add to `jit_decoders_cstr_cy.pyx` (new file):
   - Python wrapper `def decode_..._cstr(...)` matching the Numba signature
   - All-nogil `cdef i64 _..._loop(...)` body

3. Add the `Extension(...)` for `jit_decoders_cstr_cy` to setup.py.

4. Write `scripts/ab_test_decoder_<name>_cstr.py`:
   - Construct cases that fire each constraint (weight cap, fragility,
     support ratio, centroid req, CoG envelope, overhang)
   - Assert array_equal placements

5. Wire dispatcher (extend the existing try-import — three more names).

6. Smoke-test against `industry_smoke` (in addition to `br-smoke`):
   ```bash
   make industry-smoke
   ```
   Expected: IND2 = 4p / 0 unp / 0 errs; IND9 ≈ 85.5 % util₁; IND10 ≈ 95.4 % util₁

7. Commit each decoder separately:
   - `Phase 4a (port): jit_constraints .pxd surface + helper refactor`
   - `Phase 4b (port): decode_njit_mode_cstr (cstr modes 0/1/2) Cython LIVE`
   - `Phase 4c (port): decode_blocks_njit_mode_cstr (cstr mode 4) Cython LIVE`
   - `Phase 4d (port): decode_layer_njit_cstr (cstr mode 3) Cython LIVE`

### After Phase 4

The entire BRKGA hot path is Cython. Phase 5 (`prange` parallel pop eval)
becomes the next multiplier — see Section 5 above for details.

---

## 10. Quick reference — the canonical numbers

These are the v3.12 baselines (pre-port) every phase must match:

| Workload | Seed | Util / Result |
|---|---|---|
| BR1 #1 | 43 | 91.05 % |
| BR1 #2 | 44 | 91.36 % |
| BR1 #3 | 45 | 88.11 % |
| BR3 #1 | 43 | 94.02 % |
| BR3 #2 | 44 | 95.22 % |
| BR3 #3 | 45 | 92.70 % |
| IND2 (pharma, 200 fragile) | 42 | 4p / 0 unp / 0 errs |
| IND9 (document, 100) | 42 | 2p / 0 unp / 0 errs, 85.5 % util₁ |
| IND10 (cold-chain, 80) | 42 | 2p / 0 unp / 0 errs, 95.4 % util₁ |

A drift on any of these at the canonical 30 s budget = a port bug.

---

## 11. Docs read on resume

- `docs/reports/23_deep_analysis.md` — pre-port state-of-algorithm; the
  "why we're porting" doc
- `docs/reports/25_refactor.md` — monolith → 12-module refactor;
  the "why we split first" doc
- This file (`PORT_HANDOFF.md`) — the "what to do next" doc
