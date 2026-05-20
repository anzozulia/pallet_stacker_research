"""
visualize_cases.py - Render representative benchmark cases as a multi-panel figure.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from pallet_packer import (
    Box, Pallet, PalletPacker, PackerConfig,
    ALL_ROTATIONS, THIS_SIDE_UP,
)
from benchmark import (
    mixed_classic, bischoff_ratcliff_lite, pareto_distribution_boxes,
)


def cuboid_faces(x, y, z, dx, dy, dz):
    return [
        [(x, y, z), (x+dx, y, z), (x+dx, y+dy, z), (x, y+dy, z)],
        [(x, y, z+dz), (x+dx, y, z+dz), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],
        [(x, y, z), (x+dx, y, z), (x+dx, y, z+dz), (x, y, z+dz)],
        [(x, y+dy, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x, y+dy, z+dz)],
        [(x, y, z), (x, y+dy, z), (x, y+dy, z+dz), (x, y, z+dz)],
        [(x+dx, y, z), (x+dx, y+dy, z), (x+dx, y+dy, z+dz), (x+dx, y, z+dz)],
    ]


def color_for(item_id, palette):
    family = item_id.split("-")[0]
    if family not in palette:
        names = list(mcolors.TABLEAU_COLORS.values())
        palette[family] = names[len(palette) % len(names)]
    return palette[family]


def plot_one(ax, result, pallet, title):
    palette = {}
    if result.pallets:
        # Use only the first pallet for the visual; show summary in title.
        st = result.pallets[0]
        outline = cuboid_faces(0, 0, 0, pallet.length, pallet.width, pallet.height)
        ax.add_collection3d(Poly3DCollection(
            outline, facecolors=(0, 0, 0, 0), edgecolors="grey",
            linewidths=0.5, linestyles="--"))
        for p in st.placements:
            faces = cuboid_faces(p.x, p.y, p.z, p.dx, p.dy, p.dz)
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=color_for(p.box.id, palette),
                edgecolors="black", linewidths=0.3, alpha=0.85))
    ax.set_xlim(0, pallet.length)
    ax.set_ylim(0, pallet.width)
    ax.set_zlim(0, pallet.height)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((pallet.length, pallet.width, pallet.height))
    except Exception:
        pass
    ax.set_title(title, fontsize=9)


def main():
    PALLET = Pallet(120, 100, 100, max_weight=500)

    cases = [
        # Known-optimal
        ("B1: 8×(60×50×50) — perfect 100% fill",
         [Box(f"K-{i}", 60, 50, 50, weight=5) for i in range(8)],
         PALLET, PackerConfig(multi_start_trials=10)),
        # Headline mix
        ("C1: headline 45 mixed",
         mixed_classic(), PALLET, PackerConfig(multi_start_trials=15)),
        # Pareto
        ("C3: Pareto 30 (heterogeneous)",
         pareto_distribution_boxes(30), PALLET,
         PackerConfig(multi_start_trials=15)),
        # All-fragile
        ("E2: all-fragile (no stacking)",
         [Box(f"F-{i}", 40, 30, 25, weight=2, max_load_on_top=0) for i in range(12)],
         PALLET, PackerConfig(multi_start_trials=10)),
        # Weight-limited
        ("E3: weight-limited (≤2/pallet)",
         [Box(f"H-{i}", 40, 40, 40, weight=120) for i in range(6)],
         Pallet(120, 100, 100, max_weight=300), PackerConfig(multi_start_trials=10)),
        # Almost-fits
        ("E4: 'needle eye' 110×95×95",
         [Box(f"N-{i}", 110, 95, 95, weight=20) for i in range(6)], PALLET,
         PackerConfig(multi_start_trials=5)),
    ]

    fig = plt.figure(figsize=(15, 9))
    for i, (title, boxes, pallet, cfg) in enumerate(cases, start=1):
        packer = PalletPacker(pallet, cfg)
        result = packer.pack(boxes)
        ax = fig.add_subplot(2, 3, i, projection="3d")
        summary = (f"{result.num_pallets} pallets, "
                   f"util {result.total_volume_utilisation:.0%}")
        plot_one(ax, result, pallet, f"{title}\n→ {summary}")

    plt.suptitle("Representative cases (showing pallet #1 of each result)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("case_gallery.png", dpi=120, bbox_inches="tight")
    print("Wrote case_gallery.png")


if __name__ == "__main__":
    main()
