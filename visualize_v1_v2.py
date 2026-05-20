"""Side-by-side 3D visualization comparing v1 vs v2 packing for F3.

The F3 case is where v2's block-building + BRKGA actually helps: v1 packs
the 12 boxes (6 big + 6 small) into 2 pallets at 45% utilisation, but v2
finds the interlocking layout that fits all 12 onto 1 pallet at 90% util.
"""
from __future__ import annotations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from pallet_packer import Pallet, PackerConfig, PalletPacker
from failure_cases import case_F3_interlock_pattern

# Reproduce the same packings the comparison harness uses.
case = case_F3_interlock_pattern()

cfg_v1 = PackerConfig(seed=42, multi_start_trials=case.config.multi_start_trials)
cfg_v2 = PackerConfig(seed=42, multi_start_trials=case.config.multi_start_trials,
                     use_block_building=True, use_brkga=True,
                     brkga_population_size=16, brkga_generations=4)
v1_result = PalletPacker(case.pallet, cfg_v1).pack(case.boxes_factory())
v2_result = PalletPacker(case.pallet, cfg_v2).pack(case.boxes_factory())

print(f"v1: {v1_result.num_pallets} pallets, "
      f"util={v1_result.total_volume_utilisation:.1%}")
print(f"v2: {v2_result.num_pallets} pallets, "
      f"util={v2_result.total_volume_utilisation:.1%}")


def _box_faces(x, y, z, dx, dy, dz):
    """Return the 6 quadrilateral faces of an axis-aligned box."""
    p = np.array([
        [x, y, z], [x + dx, y, z], [x + dx, y + dy, z], [x, y + dy, z],
        [x, y, z + dz], [x + dx, y, z + dz], [x + dx, y + dy, z + dz],
        [x, y + dy, z + dz],
    ])
    return [
        [p[0], p[1], p[2], p[3]],  # bottom
        [p[4], p[5], p[6], p[7]],  # top
        [p[0], p[1], p[5], p[4]],  # front
        [p[2], p[3], p[7], p[6]],  # back
        [p[1], p[2], p[6], p[5]],  # right
        [p[0], p[3], p[7], p[4]],  # left
    ]


def _draw_pallet(ax, pallet, placements, title):
    L, W, H = pallet.length, pallet.width, pallet.height
    # Pallet outline
    for x0, y0, x1, y1, z0, z1 in [
        (0, 0, L, 0, 0, 0), (0, W, L, W, 0, 0),
        (0, 0, 0, W, 0, 0), (L, 0, L, W, 0, 0),
    ]:
        ax.plot([x0, x1], [y0, y1], [z0, z1], color="black", linewidth=0.8)

    # Boxes — colour big boxes blue, small boxes orange (matches F3 names)
    for p in placements:
        is_big = p.box.id.startswith("L-")
        color = "#4A90E2" if is_big else "#F5A623"
        faces = _box_faces(p.x, p.y, p.z, p.dx, p.dy, p.dz)
        poly = Poly3DCollection(faces, facecolor=color, edgecolor="black",
                               linewidths=0.5, alpha=0.85)
        ax.add_collection3d(poly)

    ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
    ax.set_box_aspect([L, W, H])
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.set_zlabel("Z (mm)")
    ax.set_title(title, fontsize=11, pad=10)
    ax.view_init(elev=20, azim=-50)


# Layout: v1 = 2 pallets side-by-side, v2 = 1 pallet
fig = plt.figure(figsize=(15, 6))

# v1: 2 pallets
ax1 = fig.add_subplot(1, 3, 1, projection="3d")
_draw_pallet(ax1, case.pallet, v1_result.pallets[0].placements,
             f"v1 — pallet 1/2\n{len(v1_result.pallets[0].placements)} boxes")
ax2 = fig.add_subplot(1, 3, 2, projection="3d")
_draw_pallet(ax2, case.pallet, v1_result.pallets[1].placements,
             f"v1 — pallet 2/2\n{len(v1_result.pallets[1].placements)} boxes")
# v2: 1 pallet
ax3 = fig.add_subplot(1, 3, 3, projection="3d")
_draw_pallet(ax3, case.pallet, v2_result.pallets[0].placements,
             f"v2 — pallet 1/1\n{len(v2_result.pallets[0].placements)} boxes (block + BRKGA)")

plt.suptitle(
    f"F3 interlock: v1 uses {v1_result.num_pallets} pallets at "
    f"{v1_result.total_volume_utilisation:.0%} util; "
    f"v2 finds the interlock in {v2_result.num_pallets} pallet at "
    f"{v2_result.total_volume_utilisation:.0%} util",
    fontsize=12, y=0.98,
)
plt.tight_layout()
plt.savefig("/home/claude/packer/f3_v1_vs_v2.png", dpi=110, bbox_inches="tight")
print("Wrote f3_v1_vs_v2.png")
