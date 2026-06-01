import os, sys, hashlib
sys.path.insert(0, "/app")
from pallet_packer import validate, Box, Pallet, PackerConfig, THIS_SIDE_UP
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core import driver as drv

warmup_jit()

# Instrument: confirm _pack_with_groups is actually invoked.
orig = drv._pack_with_groups
calls = {"n": 0}
def spy(*a, **k):
    calls["n"] += 1
    return orig(*a, **k)
drv._pack_with_groups = spy

# Force MANY boxes so groups MUST span multiple pallets.
boxes = []
for g in ("A", "B", "C"):
    for i in range(40):
        boxes.append(Box(id=f"{g}-{i:02d}", length=400, width=350, height=300,
                         weight=2.0, allowed_rotations=THIS_SIDE_UP, group=g))
pallet = Pallet(length=1200, width=1000, height=1500, max_weight=1000)
cfg = PackerConfig()

def sigstr(r):
    rows=[]
    for pi,st in enumerate(r.pallets):
        for p in st.placements:
            rows.append((pi,str(p.box.id),p.rotation.name,round(float(p.x),6),round(float(p.y),6),round(float(p.z),6)))
    rows.sort()
    return "\n".join("|".join(map(str,row)) for row in rows)

def grp_pallets(r):
    g={}
    for pi,st in enumerate(r.pallets):
        for p in st.placements:
            g.setdefault(str(p.box.id).split("-")[0], set()).add(pi)
    return g

sigs=[]
for seed in (5,5,5,5):  # repeat same seed 4x
    r = brkga_pack_v35(boxes, pallet, cfg, time_limit_s=600.0, max_pallets=20,
                       population_size=60, n_populations=2, patience=8,
                       seed=seed, verbose=False, n_modes=6)
    sigs.append(sigstr(r))
last=r
g=grp_pallets(r)
split={k:sorted(v) for k,v in g.items() if len(v)>1}
errs=validate(r, pallet, cfg)
print("pack_with_groups_calls=", calls["n"])
print("distinct_sigs=", len(set(sigs)), " pallets=", len(r.pallets), " unp=", len(r.unpacked))
print("group_pallets=", {k:sorted(v) for k,v in g.items()})
print("split_across_pallets=", split, " (empty=co-located)")
print("validator_errors=", len(errs))
print("RESULT=", "PASS" if (len(set(sigs))==1 and len(split)==0 and len(errs)==0 and calls["n"]>=4) else "FAIL")
