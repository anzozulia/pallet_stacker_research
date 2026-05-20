"""
visualize.py - Render a packing JSON as 3D figures (one per pallet).
"""
import json
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors


def cuboid_faces(x, y, z, dx, dy, dz):
    """Return the six rectangular faces of a cuboid as polygon vertex lists."""
    return [
        # bottom (z)
        [(x, y, z), (x + dx, y, z), (x + dx, y + dy, z), (x, y + dy, z)],
        # top (z + dz)
        [(x, y, z + dz), (x + dx, y, z + dz),
         (x + dx, y + dy, z + dz), (x, y + dy, z + dz)],
        # front (y)
        [(x, y, z), (x + dx, y, z), (x + dx, y, z + dz), (x, y, z + dz)],
        # back (y + dy)
        [(x, y + dy, z), (x + dx, y + dy, z),
         (x + dx, y + dy, z + dz), (x, y + dy, z + dz)],
        # left (x)
        [(x, y, z), (x, y + dy, z), (x, y + dy, z + dz), (x, y, z + dz)],
        # right (x + dx)
        [(x + dx, y, z), (x + dx, y + dy, z),
         (x + dx, y + dy, z + dz), (x + dx, y, z + dz)],
    ]


def color_for(item_id, palette):
    """Stable color per SKU family (the letter prefix before '-')."""
    family = item_id.split("-")[0] if "-" in item_id else item_id
    if family not in palette:
        names = list(mcolors.TABLEAU_COLORS.values())
        palette[family] = names[len(palette) % len(names)]
    return palette[family]


def render(json_path: str, out_path: str = "packing_visualization.png"):
    with open(json_path) as f:
        data = json.load(f)

    pallets = data["pallets"]
    n = len(pallets)
    cols = min(3, n) if n > 0 else 1
    rows = (n + cols - 1) // cols if n > 0 else 1

    fig = plt.figure(figsize=(6 * cols, 5 * rows))
    palette = {}

    for i, p in enumerate(pallets, start=1):
        ax = fig.add_subplot(rows, cols, i, projection="3d")
        L = p["dimensions"]["L"]
        W = p["dimensions"]["W"]
        H = p["dimensions"]["H"]

        # Pallet outline (transparent)
        outline = cuboid_faces(0, 0, 0, L, W, H)
        ax.add_collection3d(Poly3DCollection(
            outline, facecolors=(0, 0, 0, 0), edgecolors="grey",
            linewidths=0.5, linestyles="--"))

        for item in p["items"]:
            pos = item["position"]
            dims = item["dimensions"]
            faces = cuboid_faces(pos["x"], pos["y"], pos["z"],
                                 dims["L"], dims["W"], dims["H"])
            c = color_for(item["item_id"], palette)
            ax.add_collection3d(Poly3DCollection(
                faces, facecolors=c, edgecolors="black",
                linewidths=0.4, alpha=0.85))

        ax.set_xlim(0, L); ax.set_ylim(0, W); ax.set_zlim(0, H)
        ax.set_xlabel("X (length)"); ax.set_ylabel("Y (width)"); ax.set_zlabel("Z (height)")
        ax.set_title(f"{p['pallet_id']}: {len(p['items'])} boxes, "
                     f"{p['utilisation']*100:.0f}%, {p['total_weight']:.0f} kg")
        try:
            ax.set_box_aspect((L, W, H))
        except Exception:
            pass

    plt.suptitle(f"Packing result: {n} pallets, "
                 f"overall utilisation {data['input_summary']['total_volume_utilisation']*100:.1f}%",
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "packing_result.json"
    out = sys.argv[2] if len(sys.argv) > 2 else "packing_visualization.png"
    render(src, out)
