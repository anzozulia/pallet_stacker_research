# Report 33 — Floor-first block shape: stop corner-tower stacking on under-filled pallets

## Symptom (reported against the live API)

A few identical boxes on a large/tall pallet were stacked into a tall, narrow
corner tower instead of being spread across the pallet floor. Reported case:

- 10 identical boxes (400×300×250) on a 1200×800×1800 pallet, `support_ratio=0.8`.
- Result: every box at `x=0`, two across the width, stacked **5 layers high**
  (`z = 0,250,500,750,1000`) — 25 % floor use, a 1000–1250 mm top-heavy column,
  when a flat 4×2 layer (8 boxes) + 2 on top is trivially available.
- `items_unpacked = 0` — the packing was *valid*, just badly shaped.
- The 8-box "perfect single layer" case (Test B) also towered (2×4).

## Root cause

The problem was **not** orientation, support, or our service options (an 8-box
single-layer case and a `support_ratio=0` case both towered identically). It was
the **block decoder's shape selection**.

`find_best_block_at_pos_njit` (shared by every block decoder — geometric and
constraint-aware, single and batch, Numba and Cython) chose the composite
`(k, l, m)` block that **maximised box count** `k·l·m`. On a tall pallet the
first `(k,l,m)` reaching the max count, in loop order, is a low-footprint
**tower**:

| boxes | old block | shape | height |
|------:|-----------|-------|-------:|
| 10 | `(1,2,5)` | 2 wide × 5 high | 1250 |
| 8  | `(1,2,4)` | 2 wide × 4 high | 1000 |

An earlier floor-first fix made the single-box finders (DFTRC / wall / corner /
slab) prefer the lowest `z`, which set the block's **anchor** to the floor — but
the block then **grew upward** into a tower. Geometric mode hid this (other modes
produced flat layouts that tied on volume and won); the constraint-aware (cstr)
path the API uses settled on the block tower.

## Fix — floor-first block shape (footprint before height)

`find_best_block_at_pos_njit` (+ Cython twin `_find_best_block_at_pos`, +
`blocks.py` `enumerate_*_block*` for mode 5) now prefer the **largest floor
footprint `k·l`**, then the tallest stack `m` that fits within that footprint and
the available count. This builds complete floor layers before stacking.

**Why it is density-neutral.** `max_m` (the stackable height) does not depend on
`(k, l)`. So whenever box count is *not* the binding limit (the dense / full-
container BR regime), the full footprint reaches the very same maximum count as
any tower — the chosen brick is identical, just described footprint-first. It
only trades count for floor-spread when boxes are *scarce* relative to the pallet
— exactly the under-filled case we want flat.

Unit check of the finder:

```
under-filled:  n=10 (300×400×250) → (4,2,1) flat 8   (was (1,2,5) tower)
               n=8                → (4,2,1) flat 8   (was (1,2,4) tower)
dense:         n=56  → (4,2,7)=56   n=48 → (4,2,6)=48   n=1000 → (4,2,7)=56
               (full brick — identical count to the old max-count choice)
```

## Verification (Cython backend, in-image build)

**Flatness** — the exact reported case through the production cstr path:

| case | before | after |
|------|--------|-------|
| 10×(400×300×250), 1200×800×1800, sr=0.8 (cstr) | **TOWER** top_z=1000, x∈{0} | **FLAT** top_z=250, x∈{0,300,600,900}, 8 on floor + 2 on top |
| 8-box Test B (geom) | FLAT | FLAT |
| sweep n=4/12/6, various pallets/seeds | FLAT | FLAT |

**BR density (block-building v2, 8 instances/set)** — bit-identical, every
per-instance utilisation unchanged:

| set | before | after |
|-----|-------:|------:|
| BR1 | 83.31 % ± 5.83 | **83.31 % ± 5.83** |
| BR3 | 82.21 % ± 2.31 | **82.21 % ± 2.31** |
| BR5 | 82.94 % ± 3.74 | **82.94 % ± 3.74** |
| BR7 | 81.37 % ± 1.52 | **81.37 % ± 1.52** |

## Files

- `pallet_packer/_brkga_core/jit_decoders_geom.py` — `find_best_block_at_pos_njit`
- `pallet_packer/_brkga_core/jit_decoders_geom_cy.pyx` — `_find_best_block_at_pos`
  (Numba/Cython twins kept logic-identical → bit-equivalence preserved)
- `pallet_packer/_brkga_core/blocks.py` — `enumerate_top_k_blocks_per_sku`,
  `enumerate_best_block_per_sku` (mode-5 precomputed blocks)

All three synced verbatim into the API service's vendored core
(`pallet_packer_api_service/core/pallet_packer/…`). Regression harness:
`scripts/verify_floor_first_fix.py` (run from repo root inside the dev image).
