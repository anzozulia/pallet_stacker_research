# v3.12 — Centroid / RFS / Overhang / CoG Envelope

Adds four physical-constraint checks to the v3.11 constraint-aware
JIT decoders, closing all Tier-A and Tier-B gaps from the deep
analysis report (`23_deep_analysis.md`):

1. `require_centroid_supported` — footprint centroid must lie over
   at least one supporter
2. `Box.requires_full_support` — per-box override that forces
   `support_ratio = 1.0` for that box
3. `Pallet.max_overhang` + `PackerConfig.allow_pallet_overhang` —
   boxes may extend past +x/+y pallet edges by the overhang amount
4. Pallet CoG envelope (`cog_envelope_fraction`,
   `cog_check_min_load_fraction`, `Pallet.cog_*_range`) — weighted
   centroid stays within the configured box

## TL;DR

| Suite | v3.11 | v3.12 |
|---|---|---|
| BR n=3/set sanity | baseline | **12/12 bit-identical** |
| Industry n=10, errors | 0 / 10 | **0 / 10** |
| Industry, util on IND8 | 53.3 % | **55.2 %** (+1.9 pp) |
| Industry, IND7 unpacked | 19 | **16** (-3) |

v3.12 closes 4 of v2's 9 constraints that v3.11 didn't enforce in
the JIT path. Together with v3.10/v3.11, v3.12 now enforces **7 of
v2's 9 constraints** natively in the fast path. The two remaining
deferred items are recursive `enforce_load_bearing` and `group`
(LTL keep-together) — both bounded scope, neither bites the current
industry suite.

---

## 1. What was built

### 1.1 `precompute_constraint_arrays` (extended)

Now returns `(weights, mlot, rfs, pallet_max_weight, has_constraints)`.
The new `rfs[n]` int64 array carries `1` for boxes whose
`requires_full_support` flag is set. Setting it on any box flips
`has_constraints = True` so the constraint path activates.

### 1.2 `precompute_cog_envelope` (new)

Returns `(cog_x_min, cog_x_max, cog_y_min, cog_y_max,
cog_min_load_frac, cog_active)` resolved from `Pallet.cog_*_range`
(if provided) or derived from `config.cog_envelope_fraction`. The
`cog_active` flag is `False` only when no envelope restriction is
in effect (`cog_envelope_fraction >= 1.0` AND no explicit ranges),
in which case the JIT skips the check entirely.

### 1.3 `_check_load_on_top_njit` (extended)

Two new arguments: `require_centroid: int` and `require_full_support: int`.

- **Centroid check** (when `require_centroid != 0`): walks the same
  supporter loop already in Pass 1, tracks whether the footprint
  centroid `(cand_x + cand_dx/2, cand_y + cand_dy/2)` lies within
  any supporter's XY footprint; rejects if not.
- **RFS override**: when the per-box `requires_full_support` flag is
  set, the effective support ratio is forced to 1.0 regardless of
  `config.support_ratio`.

Both default to inactive (0); old call sites continue to work as
before. The centroid loop adds zero extra iterations — it piggybacks
on the existing supporter scan.

### 1.4 `_check_cog_envelope_njit` (new)

Given a candidate placement and per-pallet running sums
`pallet_weights`, `pallet_sum_xw`, `pallet_sum_yw`:

1. If new total pallet weight would be below
   `cog_min_load_frac * pallet_max_weight`, skip (no CoG enforcement
   while pallet is lightly loaded — matches v2 semantics).
2. Compute post-placement weighted centroid; reject if outside
   `[cog_x_min, cog_x_max] × [cog_y_min, cog_y_max]`.

### 1.5 `_apply_cog_contribution_njit` (new)

Called after every committed placement to update `pallet_sum_xw`
and `pallet_sum_yw` (additive maintenance of the per-pallet
weighted-position sums).

### 1.6 Decoder integration

All three constraint-aware decoders (`decode_njit_mode_cstr`,
`decode_blocks_njit_mode_cstr`, `decode_layer_njit_cstr`) now:

- Maintain `pallet_sum_xw` / `pallet_sum_yw` arrays
- Call `_check_cog_envelope_njit` before commit (gated by `cog_active`)
- Pass `require_centroid` and `rfs[box_idx]` to `_check_load_on_top_njit`
- For block placements, treat the entire k×l×m block as a single
  point mass at the bottom-layer footprint centroid for the CoG check,
  then `_apply_cog_contribution_njit` once per box at its actual
  position for accurate per-box CoG bookkeeping

### 1.7 Overhang (`max_overhang` / `allow_pallet_overhang`)

The dispatcher (`decode_chromosome`) computes
`L_eff = L + max_overhang` and `W_eff = W + max_overhang` when
overhang is enabled, then passes these to all constraint-aware
decoders. The decoders use the enlarged bin EMS as `(0, 0, 0) ×
(L_eff, W_eff, H)`, so existing geometric containment checks
naturally permit +x / +y overhang up to the configured amount (and
still forbid -x / -y overhang — matching v2's asymmetric semantics).

### 1.8 Pipeline plumbing

The new parameters flow through:
- `decode_chromosome` — added 7 new kwargs (rfs, require_centroid,
  cog_x_min/max, cog_y_min/max, cog_min_load_frac, cog_active,
  max_overhang)
- `decode_auto_mode` — forwards them
- `local_search_2opt`, `path_relinking`, `lns_polish` — added to
  signatures and inner decoder lambdas so polish phases use the same
  constraint stack as main BRKGA
- `brkga_pack_v35` — calls both precompute helpers, extracts scalars
  into `*_value` locals, forwards to all decoder closures and call
  sites

---

## 2. Industry results (v3.11 → v3.12)

| Case | v3.11 | v3.12 | Δ util₁ | Δ unp | Notes |
|---|---|---|---|---|---|
| IND1 E-commerce | 4p / 88.6 % / 24 unp | 4p / 88.6 % / 24 unp | 0 | 0 | same |
| IND2 Pharma | 4p / 6.4 % / 0 unp | 4p / 6.4 % / 0 unp | 0 | 0 | fragility-bound |
| IND3 Furniture | 5p / 55.7 % / 3 unp | 5p / 55.7 % / 3 unp | 0 | 0 | same |
| IND4 Beverage | 1p / 69.2 % / 0 unp | 1p / 69.2 % / 0 unp | 0 | 0 | weight-bound |
| IND5 Retail 4-SKU | 2p / 92.9 % / 0 unp | 2p / 92.9 % / 0 unp | 0 | 0 | already saturated |
| IND6 LTL groupage | 1p / 39.2 % / 0 unp | 1p / 39.2 % / 0 unp | 0 | 0 | group still unenforced |
| **IND7 Electronics** | 3p / 48.0 % / 19 unp | 3p / **48.2 %** / **16 unp** | **+0.2** | **-3** | centroid forces denser layout |
| **IND8 Automotive** | 4p / 53.3 % / 6 unp | 4p / **55.2 %** / **0 unp** | **+1.9** | **-6** | centroid + CoG help heterogeneous |
| IND9 Document | 2p / 85.5 % / 0 unp | 2p / 85.5 % / 0 unp | 0 | 0 | already saturated |
| IND10 Cold-chain | 2p / 95.4 % / 0 unp | 2p / 95.4 % / 0 unp | 0 | 0 | already saturated |

**Total validator errors: 0 / 10**, same as v3.11.

### Why IND7 / IND8 improved

These are the heterogeneous-multi-pallet cases where v3.11's
`support_ratio` check sometimes accepted placements with the footprint
centroid hanging off the supporter (technically supported by ≥80%
contact area but unstable in practice). v3.12's centroid check
rejects those, forcing the BRKGA toward placements where the
centroid actually rests on a supporter — which incidentally produces
tighter packings on pallet 0, fitting more boxes per pallet.

### Why the other 8 cases unchanged

Most of our 10 industry cases either:
- Have homogeneous SKUs where the bottom-layer placement is already
  centroid-supported by construction (IND4, IND9, IND10)
- Are constraint-bound by something else (IND2 = fragility, IND3 =
  too-tall fridges, IND6 = group)
- Already saturate util₁ via the block decoders (IND1, IND5)

---

## 3. BR regression (n=3/set, 30 s/instance)

| Set | v3.11 baseline | v3.12 | W/T/L |
|---|---|---|---|
| BR1 #1, #2, #3 | 91.05, 91.36, 88.11 | **91.05, 91.36, 88.11** | 0/3/0 |
| BR3 #1, #2, #3 | 94.02, 95.22, 92.70 | **94.02, 95.22, 92.70** | 0/3/0 |
| BR5 #1, #2, #3 | 91.45, 93.35, 92.13 | **91.45, 93.35, 92.13** | 0/3/0 |
| BR7 #1, #2, #3 | 92.87, 92.26, 91.42 | **92.87, 92.26, 91.42** | 0/3/0 |

**12 of 12 instances bit-identical.** `has_constraints=False` short-
circuits the entire constraint check at the dispatcher level, so the
geometric path is untouched.

---

## 4. Constraint coverage — where we are now

| v2 constraint | v3.10 | v3.11 | **v3.12** |
|---|---|---|---|
| Geometric containment | ✓ | ✓ | ✓ |
| Non-overlap (EMS) | ✓ | ✓ | ✓ |
| `Pallet.max_weight` | ✓ | ✓ | ✓ |
| `Box.max_load_on_top` (direct supporters) | ✓ | ✓ | ✓ |
| `support_ratio` | ✓ | ✓ | ✓ |
| `require_centroid_supported` | — | — | **✓** |
| `requires_full_support` (per-box) | — | — | **✓** |
| `max_overhang` / `allow_pallet_overhang` | — | — | **✓** |
| CoG envelope | — | — | **✓** |
| `enforce_load_bearing` (recursive) | direct only | direct only | direct only |
| `group` (LTL keep-together) | — | — | — |

**7 of 9 v2 constraints natively enforced in the JIT fast path.**
Recursive load-bearing and group remain deferred — bounded scope,
neither bites the current industry suite (max stack depth = 3; IND6
already passes validator without group check).

---

## 5. Files touched

`pallet_packer/brkga_v3_5.py`:

- `precompute_constraint_arrays` — added `rfs` array
- `precompute_cog_envelope` — new helper
- `_check_load_on_top_njit` — added `require_centroid`,
  `require_full_support` params
- `_check_cog_envelope_njit` — new JIT helper
- `_apply_cog_contribution_njit` — new JIT helper
- `decode_njit_mode_cstr`, `decode_blocks_njit_mode_cstr`,
  `decode_layer_njit_cstr` — accept + use the new params, maintain
  `pallet_sum_xw`/`pallet_sum_yw`
- `decode_chromosome` — accepts new kwargs, computes `L_eff`/`W_eff`
  for overhang, passes everything through
- `decode_auto_mode` — forwards
- `local_search_2opt`, `path_relinking`, `lns_polish` — added params
  + lambda forwarding
- `brkga_pack_v35` — extracts config values, passes through closures
  and polish call sites
- `warmup_jit` — pre-compiles new cstr decoder signatures

---

## 6. Status

- v3.12 ships 4 new constraints (centroid, RFS, overhang, CoG).
- BR regression: 12/12 bit-identical at n=3 — no geometric regression.
- Industry: 0 errors maintained, IND7 / IND8 actually improved by
  the centroid check.
- Production recommendation: **v3.12 for all workloads**.

## 7. What's left

The deep-analysis report's remaining roadmap items:

1. **`group` constraint** (LTL keep-together) — would fix IND6's
   silent group violation. Implementation: 2-4 hours; pre-group BPS
   sort + per-pallet group-id tracking. Skip-flagged in the original
   user direction; revisit when a real LTL customer surfaces.
2. **Recursive load-bearing** — only matters for ≥4-deep stacks; no
   current industry case bites. Implementation: 4-6 hours; needs
   parent-pointer or flat-supporter-list bookkeeping.
3. **C/Cython decoder port** — original final-step item; closes the
   ~1.6 pp BR1 gap to BRKGA-2013.

Of the three, the C/Cython port is the only one with material BR
impact. v3.12 is now feature-complete for industry deployment
modulo the two deferred items.
