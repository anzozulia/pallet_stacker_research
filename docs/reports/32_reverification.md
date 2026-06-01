# Re-verification — All Session Changes

A full, extra-precise re-verification after this session's five fixes (block
decoder, always-on stability, input gate, max_pallets, Box.group) — confirming
**nothing regressed** and the fixes **hold**, with deep focus on edge cases and
the 10 real-world industry cases. 8 dimensions, each adversarially cross-checked.
All probes in `scripts/_verify/reverify_*.py`, run in the project Docker image.

> **Verdict: clean.** 7/8 dimensions passed first time; the 8th surfaced one
> **MEDIUM packing-quality** defect in the group wrapper (it never violated a
> safety invariant) — **now fixed and re-verified**. The geometric decode path is
> proven **bit-identical** to pre-session; backend equivalence, determinism,
> conservation, and all 10 industry cases hold. The session's work is safe to
> rely on.

---

## 1. Regression — nothing broke

| Check | Result |
|---|---|
| **Geometric BR bit-identical** to `cead45d` (pre-session) | ✅ util %, placement sha256, n_placed **all match** on BR1#1/#2, BR3#1, BR5#1, BR7#1; SKU partition identical (the mlot/rfs key change is a no-op for weightless data) |
| **Cython ≡ Numba** after the block-decoder change (highest risk) | ✅ cstr mode-4 block decoder: 0 mismatches across 360 random + 70 edge + 800 ε-boundary + 500 per-box-stress comparisons; both backends implement the identical per-box loop |
| **Batch ≡ serial** (shared `_decode_cstr_blocks_loop`) | ✅ `cstr blocks (mode 4): bit-identical`; 187/189 placement checks (2 "fails" = pre-existing non-integer float-fitness gap, untouched this session) |
| **Determinism + thread-invariance** after driver changes | ✅ `GLOBAL_SHA256` identical OMP=1 vs 8; constrained, D2-weightless, and group multi-pallet solves all bit-identical 3× and thread-invariant |
| **`use_v2_seed` stays off for weightless** (Phase 7a) | ✅ `has_constraints=False ⇒ v2-seed off` confirmed |

## 2. The five fixes still hold

`test_partA_fix` (0 load/support errs), `verify_fix_comprehensive` (constrained
ladder + industry 0 errs, geometric bit-identical), `test_d2_stability` (weightless
support enforced), `test_input_gate` (21/21 malformed caught), `test_group_colocation`
(14/14) — **all pass**.

## 3. Real-world industry cases (deep)

All 10 cases re-checked across **7 dimensions** — validator, conservation,
max_pallets, group co-location, weight cap, fragile-uncrushed, bounds — with
**zero violations**:

| | IND1 | IND2 | IND3 | IND4 | IND5 | IND6 | IND7 | IND8 | IND9 | IND10 |
|---|---|---|---|---|---|---|---|---|---|---|
| pallets | 9 | 4 | 8 | 1 | 2 | 1 | 8 | 4 | 2 | 2 |
| placed/unpkd | 120/0 | 200/0 | 15/0 | 44/0 | 150/0 | 43/0 | 150/0 | 60/0 | 100/0 | 80/0 |
| violations | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Exact match to `30_verification.md`. IND6's 3 groups co-located (0 splits).
**Fragile non-vacuity** proven: the engine genuinely stacks deep (IND7: 24
z-levels, 109 load-bearing boxes) yet **0** of the placed fragile (`mlot=0`)
boxes carry any load.

## 4. Edge cases (extra attention)

`reverify_edge_comprehensive` (38/38) + `reverify_edge_group_industry` (14/14) +
re-run of both original edge probes. New interactions all hold: empty list with
`max_pallets>1`; single grouped/ungrouped box; **a group larger than one pallet
→ never split** (overflow → unpacked); all-one-group; weightless single box under
D2; `max_pallets=1` with groups (base case). The two degenerate **silent-wrong**
inputs from the original audit (all-zero-dim box, NaN `max_weight`) are now
**rejected by the input gate**.

## 5. Static review of the session diff

`cead45d..HEAD`, all 7 changed files, 25/25 checks: per-box block decoder cy==nb;
SKU partition unchanged for geometric; group-pack conservation + co-location +
re-id + base-case; `use_cstr_path` distinct from `has_constraints` (v2-seed
correctly stays off for stability-only); `_is_cap` NaN/inf handling. No defect.

---

## 6. The one finding — and its fix

**MEDIUM (quality, not safety): the group wrapper over-packed early pallets.**
`_pack_with_groups` assigned whole groups to pallets first-fit-decreasing on a
**volume-only** budget (`vol × 0.85`). That over-estimates capacity when **weight
binds** (heavy low-volume groups) or boxes are **non-stackable** (fragile,
`mlot=0` → single layer → they consume floor area, not full-height volume). Groups
piled onto too few pallets; the per-pallet solve then spilled the excess to
`unpacked` while allowed pallets sat empty. Severe but **never unsafe** — across
7 adversarial combined-constraint cases the hard invariants always held (0
validator errors, 0 weight overruns, 0 crushed fragile, **0 group splits**,
conservation exact).

**Fix** (`driver.py _pack_with_groups`): the assignment now tracks **three**
budgets per pallet — volume, **weight** (`pallet.max_weight`), and **floor area**
(footprint of unstackable boxes) — and a bundle is placed only if all three fit.

**Re-verified** (`reverify_realworld_combos.py`, 63/63):

| Case | before | after |
|---|---|---|
| C6 (84 boxes, weight-binding) | 57 placed / 27 unpacked | **84 / 0** |
| C7 (90 boxes, 5 heavy + 50% fragile) | 25 placed / 65 unpacked | **90 / 0** (beats ungrouped) |
| C5 (300 boxes) | grouped under ungrouped | grouped places **+30 more** |

`test_group_colocation` still 14/14 (the stackable constructed case is unchanged,
confirming the new budgets only bind on weight/non-stackable loads).

---

## 7. Bottom line

Every change made this session is verified: no regression, all fixes hold, 10/10
industry cases and the edge surface clean, and the one quality defect found is
fixed and re-confirmed. The geometric benchmark path is bit-identical to before
the session. The packer is correct and robust for real-world integer-dimension
input.
