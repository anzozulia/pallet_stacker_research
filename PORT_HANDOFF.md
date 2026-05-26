# Cython Port — Handoff & Resume Plan

Living document for the v3.12 → Cython port. Read top-to-bottom on
resume; the "Resume here" section at the bottom is the next action.

---

## 1. Where we are right now

**Phase 3 + Phase 4 + Phase 5 complete. The entire BRKGA hot path is
Cython AND parallel.** End-to-end BR1#1 = 91.05% / BR3#1 = 94.02% /
IND2 = 4p 0unp 0errs / IND9 = u1=85.5% / IND10 = u1=95.4% — all
bit-identical to the v3.12 baseline.

| Decoder | Numba | Cython | Speedup |
|---|---|---|---|
| Geom mode 0 (DFTRC) | 4607 µs | 2681 µs | **1.72×** |
| Geom mode 1 (wall) | 2789 µs | 2533 µs | 1.10× |
| Geom mode 2 (corner) | 2644 µs | 2319 µs | 1.14× |
| Geom mode 3 (layer, n=100) | 3134 µs | 2616 µs | 1.20× |
| Geom mode 4 (blocks, small case) | 2.6 µs | 4.7 µs | 0.54× ⚠ |
| Geom mode 5 (precomputed, small case) | 8.8 µs | 11.6 µs | 0.76× ⚠ |
| Cstr modes 0/1/2 (mode 0, n=100) | 1272 µs | 1163 µs | 1.09× |
| Cstr mode 4 (blocks, small case) | 12.1 µs | 15.9 µs | 0.76× ⚠ |
| Cstr mode 3 (layer, n=100) | 653 µs | 497 µs | **1.31×** |

⚠ Small-case microbenches are misleading — Python wrapper setup
dominates per-call work when MAX_BINS is small and most boxes early-out.
End-to-end br-smoke + industry-smoke confirm no quality regression
across all eight industry workloads. Phase 5 (`prange` parallel pop
eval) is where the multi-core multiplier kicks in.

### Recent commits (port-relevant)

```
35c5852  Phase 0: Docker + Cython build infra
04ffabc  Phase 1 (port): jit_primitives_cy
4301285  Phase 2 (port): jit_constraints_cy
400fd2c  Phase 3a: decode_njit_mode (modes 0/1/2) LIVE
64cacb4  Phase 3b: decode_layer_njit (mode 3) LIVE
db95105  Phase 3c: decode_blocks_njit_mode (mode 4) LIVE
5816add  Phase 3d: decode_precomputed_blocks_njit_mode (mode 5) LIVE
794a9a9  Phase 4a: jit_constraints_cy.pxd surface
8f496f8  Phase 4b: decode_njit_mode_cstr (cstr 0/1/2) LIVE
4bc96ba  Phase 4c: decode_blocks_njit_mode_cstr (cstr 4) LIVE
5a0664b  Phase 4d: decode_layer_njit_cstr      (cstr 3) LIVE
03b54ca  Phase 5a: decode_batch_njit_mode prange (7.85× per-call)
4c799d9  Phase 5b: batch entries for all remaining decoders
555158f  Phase 5c: driver.py uses decode_population_fitness
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
├── jit_constraints_cy.pxd    Cython header — cdef nogil surface for cstr
├── jit_constraints_cy.pyx    Cython port — used by Phase 4 cstr decoders
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
├── jit_decoders_cstr.py      Numba reference (fallback)
├── jit_decoders_cstr_cy.pyx  Cython port — COMPLETE: all 3 cstr decoders
│                             (mode 0/1/2, mode 3, mode 4) Cython-LIVE
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

**Next action: Phase 6 — end-to-end final eval + report.**

The port is functionally complete:
- Cython for every decoder (Phase 3 + 4).
- Parallel batch decode wired into the BRKGA inner loop (Phase 5).
- All canonical seeds bit-identical to v3.12 baseline.

What remains is measuring the actual end-to-end wall-clock gain and
documenting where the algorithm stands vs the literature.

### Suggested tasks for Phase 6

1. **Wall-clock benchmark**:
   - Re-run `make br-smoke` with verbose timing; compare generations/sec
     to a baseline checkout pre-Phase-5 (`git stash` the driver change
     or check out 5a4e6e8). Expect 2-5× more generations in the same
     30s budget on multi-core machines.
   - Re-run `scripts/run_v310_industry_eval.py` at 30s budget per pallet
     for a side-by-side time vs quality comparison.

2. **Full BR n=10 at the canonical 30s budget**:
   - `python scripts/run_v310_br_regression.py --n 10 --time 30 \
        --sets thpack1 thpack2 thpack3 thpack4 thpack5 thpack6 thpack7`
   - Compare against Gonçalves-Resende 2013 SOTA on BR1 (~92.62%, our
     pre-Phase-5 mean was 91.04%). The extra generations from parallel
     pop eval should close some of the gap.

3. **Industry workloads at production budget**:
   - 30s/pallet across IND1-10, full n=10 seeds per case. Report
     unpacked count, pallet count, per-constraint error count, u1/utT.

4. **Profile any unexpected bottleneck**:
   - `python -m cProfile -o pr.out scripts/run_v310_br_regression.py ...`
   - `snakeviz pr.out` to find what's left to optimise. Likely
     candidates: the v2-seed PalletPacker (still pure Python), local
     search polish, path relinking inner loop.

5. **Write `docs/reports/27_port_complete.md`** with:
   - Architecture summary (Cython + prange + Docker pipeline)
   - Per-phase commit log and per-decoder speedup table (from Section 1)
   - Final BR results vs literature
   - Final industry results
   - Known limitations + suggested follow-ons (polish polish parallel,
     v2 packer port if needed)

### Possible follow-ons (post Phase 6)

These are all OPTIONAL — the port is production-ready as it stands.

- **Per-thread scratch reuse**: each call to a `decode_batch_*` entry
  allocates ~MAX_BINS × MAX_EMS_C × pop_size × 48 bytes of int64. For
  pop_size=80 and max_pallets=8 that's ~16 MB per generation, which
  shows up in profile. Stash the scratch in a thread-local cache and
  zero it instead of reallocating; ~10-15 ms saved per generation.
- **prange the polish step**: local-search 2-opt evaluates many
  neighbours sequentially. Can run in parallel.
- **JIT vs AOT path-relinking**: PR re-decodes ~50 intermediate
  chromosomes per call. Wrap in a batch entry.

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
