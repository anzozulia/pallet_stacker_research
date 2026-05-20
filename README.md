# 3D Pallet Packer

A 3D bin packing / pallet loading service in pure Python that handles the
realistic physical constraints existing libraries miss: per-box weight,
configurable rotation, overhangs, anti-floating support, fragility /
load-bearing limits, and per-pallet centre-of-gravity envelope.

## Files

| File | Purpose |
|------|---------|
| `pallet_packer.py` | Core library: data models, extreme-point engine, constraint stack, multi-start search, JSON output, validator |
| `example.py` | The user's scenario (20 boxes 30×40×70 + 15 boxes 40×45×50 + 10 boxes 60×60×30) on a 120×100×100 pallet |
| `visualize.py` | Matplotlib 3D renderer that reads `packing_result.json` |

## Algorithm

The pipeline matches the design in the research report:

1. **Outer loop — multi-pallet assignment.** First-Fit-Decreasing (or
   Best-Fit) across already-open pallets; open a new pallet only when no
   existing one accepts the box.
2. **Inner loop — extreme-point placement** (Crainic, Perboli & Tadei,
   *INFORMS J. Computing* 20(3), 2008). Each pallet maintains a set of
   candidate corner positions; the algorithm tries every (extreme point,
   rotation) pair, scores feasible candidates, places the best.
3. **Constraint stack** runs on every candidate placement (order: cheap → expensive):
   - Geometry: inside pallet, no overlap with placed items.
   - Pallet weight budget.
   - **Support / no-floating** — fraction of base area resting on items
     below ≥ `support_ratio` (default 0.8; matches `py3dbp.support_surface_ratio`).
     Also: footprint centroid must be over a supporter (Junqueira, Morabito
     & Yamashita, *COR* 39(1), 2012).
   - **Load-bearing** — propagate weight through contact areas; reject if
     any direct supporter would exceed its `max_load_on_top` (Bischoff,
     *EJOR* 168(3), 2006).
   - **CoG envelope** — pallet centre-of-gravity must stay inside a
     centred box (default ±25 % of pallet length/width); only enforced
     once the pallet is meaningfully loaded.
   - **Rotation** — only orientations in `box.allowed_rotations`. Helper
     constants: `ALL_ROTATIONS`, `THIS_SIDE_UP` (rotation around Z only),
     `NO_ROTATION`.
4. **Multi-start search (BRKGA-lite).** Sweep the cartesian product of
   (box ordering × 4 placement-scoring strategies × {best-fit, first-fit}),
   pick the best result by lexicographic (num_pallets, num_unpacked,
   −utilisation). Number of trials controlled by
   `PackerConfig.multi_start_trials` (default 20).

## Usage

```python
from pallet_packer import Box, Pallet, PalletPacker, PackerConfig, THIS_SIDE_UP, save_json, validate

# Define cargo.
boxes = [
    Box(id="A-1", length=30, width=40, height=70, weight=5.0,
        allowed_rotations=THIS_SIDE_UP, max_load_on_top=50.0),
    # ...
]

# Define pallet.
pallet = Pallet(length=120, width=100, height=100, max_weight=500.0)

# Configure the packer.
config = PackerConfig(
    support_ratio=0.8,
    require_centroid_supported=True,
    cog_envelope_fraction=0.25,
    enforce_load_bearing=True,
    multi_start_trials=25,
    seed=42,
)

# Pack.
packer = PalletPacker(pallet, config)
result = packer.pack(boxes)

# Verify and save.
assert not validate(result, pallet, config), "constraint violated"
save_json(result, pallet, "packing_result.json")
```

## JSON output schema

```jsonc
{
  "input_summary": {
    "items_packed": 45,
    "items_unpacked": 0,
    "pallets_used": 5,
    "total_volume_utilisation": 0.685
  },
  "pallets": [
    {
      "pallet_id": "P001",
      "dimensions": { "L": 120, "W": 100, "H": 100, "max_weight": 500 },
      "utilisation": 0.91,
      "cog": { "x": 49.5, "y": 53.3, "z": 45.8 },
      "total_weight": 109.0,
      "items": [
        {
          "item_id": "B-08",
          "position":    { "x": 0,  "y": 0,  "z": 0 },
          "dimensions":  { "L": 40, "W": 45, "H": 50 },
          "orientation": { "perm": [0, 1, 2], "name": "LWH" },
          "weight": 8.0,
          "support_ratio": 1.0,
          "supported_by": ["floor"],
          "supports": ["B-02"]
        }
      ]
    }
  ],
  "unpacked_items": []
}
```

`orientation.perm` is a permutation of (0, 1, 2) mapping the box's original
(length, width, height) axes onto the pallet's (X, Y, Z) axes. Together with
`position` (back-left-bottom corner in pallet coordinates) and `dimensions`
(the *rotated* extents), any visualiser has everything needed to draw the
cuboid.

## Quality / limits

- All sanity-check scenarios pass: perfect fill, volume-forced multi-pallet,
  weight-forced multi-pallet, "this side up" honored, fragile-item
  no-load-on-top honored, oversized item rejected.
- On the headline scenario (45 mixed boxes / 120×100×100 pallet) the packer
  uses 5 pallets with the leading pallet at 91 % volume utilisation and an
  overall 68.5 % utilisation in ~3 s with 200 trials.
- The theoretical lower bound for the headline scenario is 4 pallets;
  closing that 1-pallet gap requires a true block-building decoder
  (Eley 2002, Bortfeldt 2000) or a full BRKGA with population recombination
  (Gonçalves & Resende 2013) — both straightforward extensions of this
  codebase.

## Extending

- **New constraint** → add a method on `PalletState`, call it from
  `feasible()`, and mirror it in `validate()` so the independent check
  catches engine bugs.
- **New placement strategy** → add a branch to `_score_placement()` and add
  its name to `PalletState.SCORING_STRATEGIES`; the multi-start loop will
  pick it up automatically.
- **True BRKGA upgrade** → replace `_pack_once` with the standard random-key
  decoder, add population-level crossover and elite carry-over per
  Gonçalves & Resende (*IJPE* 145, 2013). The constraint stack stays
  unchanged.
- **MIP polish** → for ≤ 30 items per pallet, feed the BRKGA-best into an
  OR-Tools CP-SAT model expressing the do Nascimento et al. (*COR* 128,
  2021) formulation as a final improvement pass.
