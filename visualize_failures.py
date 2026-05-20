"""
visualize_failures.py - Render the three confirmed-failure cases side by side.
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from pallet_packer import PalletPacker
from failure_cases import (
    case_F1_pareto_continuum,
    case_F3_interlock_pattern,
    case_F12_strongly_heterogeneous_at_scale,
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


def plot_pallet(ax, pallet, placements, title, palette):
    outline = cuboid_faces(0, 0, 0, pallet.length, pallet.width, pallet.height)
    ax.add_collection3d(Poly3DCollection(
        outline, facecolors=(0, 0, 0, 0), edgecolors="grey",
        linewidths=0.6, linestyles="--"))
    for p in placements:
        fam = p.box.id.split("-")[0]
        if fam not in palette:
            names = list(mcolors.TABLEAU_COLORS.values())
            palette[fam] = names[len(palette) % len(names)]
        faces = cuboid_faces(p.x, p.y, p.z, p.dx, p.dy, p.dz)
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors=palette[fam], edgecolors="black",
            linewidths=0.3, alpha=0.85))
    ax.set_xlim(0, pallet.length); ax.set_ylim(0, pallet.width); ax.set_zlim(0, pallet.height)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    try:
        ax.set_box_aspect((pallet.length, pallet.width, pallet.height))
    except Exception:
        pass
    ax.set_title(title, fontsize=8)


def render_failure(c, fig, base_idx, n_cols, label):
    """Render all pallets of a failure case in a row."""
    boxes = c.boxes_factory()
    packer = PalletPacker(c.pallet, c.config)
    result = packer.pack(boxes)
    palette = {}
    n_pallets = result.num_pallets
    for i, st in enumerate(result.pallets):
        ax = fig.add_subplot(3, n_cols, base_idx + i, projection="3d")
        util = sum(p.box.volume for p in st.placements) / (c.pallet.length * c.pallet.width * c.pallet.height)
        plot_pallet(ax, c.pallet, st.placements,
                    f"{label}\n{st.pallet_id}: {len(st.placements)} boxes, "
                    f"util {util:.0%}", palette)
    return n_pallets


def main():
    # Each case gets one row; allocate enough cols for the worst.
    cases = [
        (case_F1_pareto_continuum(), "F1 Pareto continuum (gap +1)"),
        (case_F3_interlock_pattern(), "F3 interlock (gap +1)"),
        (case_F12_strongly_heterogeneous_at_scale(), "F12 strongly heterogeneous (gap +1)"),
    ]

    # First do dry runs to find max pallets per case
    max_pallets = []
    for c, _ in cases:
        boxes = c.boxes_factory()
        packer = PalletPacker(c.pallet, c.config)
        result = packer.pack(boxes)
        max_pallets.append(result.num_pallets)
    n_cols = max(max_pallets)

    fig = plt.figure(figsize=(4 * n_cols, 12))
    for row, (c, label) in enumerate(cases):
        boxes = c.boxes_factory()
        packer = PalletPacker(c.pallet, c.config)
        result = packer.pack(boxes)
        palette = {}
        for col, st in enumerate(result.pallets):
            ax = fig.add_subplot(len(cases), n_cols, row * n_cols + col + 1,
                                 projection="3d")
            util = sum(p.box.volume for p in st.placements) / (
                c.pallet.length * c.pallet.width * c.pallet.height)
            title = (f"{label}\n{st.pallet_id}: "
                     f"{len(st.placements)} boxes, util {util:.0%}")
            plot_pallet(ax, c.pallet, st.placements, title, palette)

    plt.suptitle("Confirmed failure cases (gap = +1 pallet to volume LB)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig("failure_gallery.png", dpi=110, bbox_inches="tight")
    print("Wrote failure_gallery.png")


if __name__ == "__main__":
    main()
