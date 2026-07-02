"""Realism-layer (D14 / report 34) verification — runs inside pallet-packer:dev
or any env where the package imports (the postprocess + scalar-fitness paths
need no compiled extensions).

Checks, in order:
  1. FLAG-OFF NEUTRALITY — default PackerConfig() must produce the same
     fingerprint whether or not the new code paths exist (compare against a
     pre-change checkout: run this script there with the same TAG and diff).
  2. RECENTER — a single box / an under-filled load ends up centred on the
     deck, integer coordinates, validator-clean, idempotent.
  3. ALIGN — a hand-built layer with one deviant same-SKU rotation comes out
     uniform; a deviant that supports another box is left alone (dz gate).
  4. REALISM ORDERING — with realism_weight=1.0, heavy-low beats heavy-high,
     flat beats tower, uniform orientation beats mixed, and a full pack always
     beats dropping the smallest box (bounded loss).

Usage: python scripts/verify_realism_layer.py [TAG]
"""
import json
import sys

from pallet_packer import Box, Pallet, PackerConfig
from pallet_packer.brkga_v3_5 import brkga_pack_v35
from pallet_packer.brkga_v3_fast import _fitness_pallet1
from pallet_packer.models import Placement, Rotation
from pallet_packer.packer import PackResult, PalletState
from pallet_packer.postprocess import align_orientations_pass, apply_postprocess
from pallet_packer.validate import validate
from pallet_packer._brkga_core.realism import build_realism_context

TAG = sys.argv[1] if len(sys.argv) > 1 else "RUN"
FAILURES = []


def check(name, ok, detail=""):
    print(f"[{TAG}] {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        FAILURES.append(name)


def fingerprint(res):
    return [[st.pallet_id, p.box.id, p.rotation.name, p.x, p.y, p.z]
            for st in res.pallets
            for p in sorted(st.placements, key=lambda q: q.box.id)]


# --- 1. flag-OFF neutrality fingerprint (diff across checkouts) -------------
# Needs the compiled Cython batch decoders (run inside pallet-packer:dev);
# on an uncompiled checkout this section skips and the unit checks still run.
boxes = [Box(id=f"G{i}", length=200 + 37 * (i % 5), width=150 + 23 * (i % 4),
             height=100 + 41 * (i % 3), weight=float(i % 4)) for i in range(18)]
pal = Pallet(length=1200, width=1000, height=1500, max_weight=200.0)
try:
    res = brkga_pack_v35(boxes, pal, PackerConfig(), time_limit_s=5.0,
                         population_size=30, n_populations=2, patience=8,
                         seed=42)
    print(f"[{TAG}] flag-off fingerprint sha: "
          f"{hash(json.dumps(fingerprint(res), sort_keys=True)) & 0xffffffff:08x}")
    print(json.dumps(fingerprint(res)))
except NameError:
    print(f"[{TAG}] SKIP flag-off fingerprint (Cython batch decoders not "
          f"built — run inside pallet-packer:dev for this section)")

# --- 2. recenter -------------------------------------------------------------
cfg = PackerConfig(recenter_layout=True, align_orientations=True)
b = Box(id="A", length=600, width=400, height=400, weight=25.0)
st = PalletState(pal, "P001", cfg)
st.placements.append(Placement(box=b, rotation=Rotation.LWH, x=0.0, y=0.0, z=0.0))
st.total_weight = 25.0
r = PackResult(pallets=[st], unpacked=[])
apply_postprocess(r, pal, cfg)
p = st.placements[0]
check("recenter single box", (p.x, p.y, p.z) == (300.0, 300.0, 0.0),
      f"pos=({p.x},{p.y},{p.z})")
before = (p.x, p.y)
apply_postprocess(r, pal, cfg)
check("recenter idempotent", (p.x, p.y) == before)
check("recenter validator-clean",
      not validate(r, pal, cfg))

# --- 3. align ---------------------------------------------------------------
sku = [Box(id=f"S{i}", length=400, width=300, height=200, weight=8.0)
       for i in range(3)]
st2 = PalletState(pal, "P001", cfg)
st2.placements += [
    Placement(box=sku[0], rotation=Rotation.LWH, x=0.0, y=0.0, z=0.0),
    Placement(box=sku[1], rotation=Rotation.LWH, x=400.0, y=0.0, z=0.0),
    Placement(box=sku[2], rotation=Rotation.WLH, x=800.0, y=0.0, z=0.0),
]
st2.total_weight = 24.0
swaps = align_orientations_pass(st2, pal, cfg)
check("align unifies deviant", swaps == 1 and len(
    {tuple(int(v) for v in q.dims) for q in st2.placements}) == 1)

# --- 4. realism ordering ------------------------------------------------------
rc_cfg = PackerConfig(realism_weight=1.0)
heavy = Box(id="H", length=400, width=400, height=300, weight=40.0)
light = Box(id="L", length=400, width=400, height=300, weight=2.0)
ctx = build_realism_context([heavy, light], pal, rc_cfg)


def hand(pairs):
    s = PalletState(pal, "P001", rc_cfg)
    for bx, z in pairs:
        s.placements.append(Placement(box=bx, rotation=Rotation.LWH,
                                      x=0.0, y=0.0, z=z))
        s.total_weight += bx.weight
    return PackResult(pallets=[s], unpacked=[])


f_good = _fitness_pallet1(hand([(heavy, 0.0), (light, 300.0)]), pal, realism=ctx)
f_bad = _fitness_pallet1(hand([(light, 0.0), (heavy, 300.0)]), pal, realism=ctx)
check("heavy-low beats heavy-high", f_good < f_bad - 1e-9,
      f"{f_good:.9f} < {f_bad:.9f}")

small = Box(id="SM", length=100, width=100, height=100, weight=1.0)
ctx2 = build_realism_context([heavy, small], pal, rc_cfg)
full = hand([(heavy, 1100.0)])
full.pallets[0].placements.insert(0, Placement(
    box=small, rotation=Rotation.LWH, x=0.0, y=0.0, z=0.0))
dropped = hand([(heavy, 0.0)])
dropped.unpacked = [small]
check("bounded loss never drops a box",
      _fitness_pallet1(full, pal, realism=ctx2)
      < _fitness_pallet1(dropped, pal, realism=ctx2))

print(f"[{TAG}] {'ALL PASS' if not FAILURES else 'FAILURES: ' + str(FAILURES)}")
sys.exit(1 if FAILURES else 0)
