"""Save a bounded-cost 3D debug preview; independent of the dashboard."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def save_mesh_preview(mesh, output_path, *, max_faces=20_000):
    """Render at most max_faces triangles without changing the source mesh."""
    if max_faces < 1:
        raise ValueError("max_faces must be positive")
    output_path = Path(output_path)
    face_count = len(mesh.faces)
    indices = np.linspace(0, face_count - 1, min(face_count, max_faces), dtype=int)
    triangles = mesh.vertices[mesh.faces[indices]]
    bounds = mesh.bounds
    dimensions = bounds[1] - bounds[0]
    padding = max(float(dimensions.max()) * 0.01, 1e-6)

    fig = plt.figure(figsize=(9, 7))
    try:
        ax = fig.add_subplot(111, projection="3d")
        ax.add_collection3d(Poly3DCollection(
            triangles, facecolor="#6796b8", edgecolor="none", alpha=0.9
        ))
        for setter, index in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
            setter(bounds[0, index] - padding, bounds[1, index] + padding)
        ax.set_box_aspect(np.maximum(dimensions, padding))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Room model geometry (Z up)")
        fig.tight_layout()
        fig.savefig(output_path, dpi=120)
    finally:
        plt.close(fig)
    return output_path
