"""Load converted room geometry without depending on a web framework."""

from pathlib import Path

import numpy as np
import trimesh


class GeometryError(ValueError):
    """An OBJ has no usable room geometry."""


def load_obj_mesh(obj_path):
    """Return one mesh with all OBJ scene-instance transforms baked in."""
    path = Path(obj_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise GeometryError(f"OBJ is missing or empty: {path}")
    try:
        scene = trimesh.load(path, force="scene", process=False)
        mesh = scene.to_mesh()
    except Exception as exc:
        raise GeometryError(f"Could not load OBJ geometry at {path}: {exc}") from exc
    if len(mesh.vertices) == 0:
        raise GeometryError(f"OBJ contains no vertices: {path}")
    if len(mesh.faces) == 0:
        raise GeometryError(f"OBJ contains no faces: {path}")
    if not np.isfinite(mesh.vertices).all():
        raise GeometryError(f"OBJ contains non-finite vertices: {path}")
    return mesh


def mesh_metadata(mesh):
    """Return JSON-compatible counts and axis-aligned geometry in model units."""
    bounds = mesh.bounds
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds": bounds.tolist(),
        "dimensions": (bounds[1] - bounds[0]).tolist(),
        "center": bounds.mean(axis=0).tolist(),
    }
