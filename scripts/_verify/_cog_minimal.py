"""Minimal deterministic repro of the CoG above-gate breach via full solve."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from pallet_packer import Box, Pallet, PackerConfig, ALL_ROTATIONS, validate
from pallet_packer.brkga_v3_5 import brkga_pack_v35, warmup_jit
from pallet_packer._brkga_core.precompute import precompute_cog_envelope
ANY=list(ALL_ROTATIONS)
warmup_jit()

def rc(pls):
    t=sx=sy=0.0
    for p in pls: w=p.box.weight;t+=w;sx+=w*(p.x+p.dx/2);sy+=w*(p.y+p.dy/2)
    return (sx/t,sy/t,t) if t>0 else (None,None,0.0)

# Minimal: one dominant heavy box + a handful of light. Tight envelope.
# finite max_weight so CoG IS engine-enforced. gate=0.2*600=120.
pallet=Pallet(length=1000,width=1000,height=1000,max_weight=600)
boxes=[Box(id="MEGA",length=400,width=400,height=400,weight=200.0,allowed_rotations=ANY)]
for i in range(8):
    boxes.append(Box(id=f"f{i}",length=200,width=200,height=200,weight=10.0,allowed_rotations=ANY))
cfg=PackerConfig(cog_envelope_fraction=0.1, cog_check_min_load_fraction=0.2,
                 support_ratio=0.0, require_centroid_supported=False, enforce_load_bearing=False)
xmn,xmx,ymn,ymx,mlf,act=precompute_cog_envelope(pallet,cfg); gate=mlf*pallet.max_weight
print(f"env x[{xmn},{xmx}] y[{ymn},{ymx}] gate={gate}")
breach=False
for seed in [1,2,3,7,42]:
    r=brkga_pack_v35(boxes,pallet,cfg,time_limit_s=3.0,max_pallets=5,seed=seed,
                     population_size=100,n_populations=2,patience=80,
                     local_search_budget_s=0.5,verbose=False,n_modes=6)
    errs=validate(r,pallet,cfg)
    for st in r.pallets:
        cx,cy,t=rc(st.placements)
        if cx is None: continue
        above=t>=gate-1e-6; inside=(xmn-1e-4<=cx<=xmx+1e-4) and (ymn-1e-4<=cy<=ymx+1e-4)
        if above and not inside:
            breach=True
            print(f"  seed={seed} {st.pallet_id}: CoG=({cx:.1f},{cy:.1f}) tot={t:.0f} "
                  f"n={len(st.placements)} above_gate=True inside=False validator_errs={len(errs)} <-- BREACH")
print(f"BREACH_FOUND={breach}")
