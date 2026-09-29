"""One-call USDZ processing entry point for a future upload endpoint."""

from pathlib import Path
import shutil
from uuid import uuid4

from .converter import ConversionError, convert_usdz_to_obj
from .geometry import GeometryError, load_obj_mesh, mesh_metadata
from .viewer import save_mesh_preview


class RoomModelError(RuntimeError):
    """A room model could not be processed."""


DEFAULT_STORAGE_ROOT = Path(__file__).resolve().parent.parent / "storage" / "room_models"


def process_room_model(usdz_path, *, storage_root=DEFAULT_STORAGE_ROOT):
    """Convert a saved USDZ to OBJ, validate it, and render a PNG preview."""
    source = Path(usdz_path).expanduser().resolve()
    if source.suffix.lower() != ".usdz":
        raise RoomModelError(f"Expected a .usdz file: {source}")
    if not source.is_file() or source.stat().st_size == 0:
        raise RoomModelError(f"USDZ is missing or empty: {source}")

    model_id = uuid4().hex
    model_dir = Path(storage_root).expanduser().resolve() / model_id
    model_dir.mkdir(parents=True, exist_ok=False)
    obj_path = model_dir / "model.obj"
    preview_path = model_dir / "preview.png"
    try:
        convert_usdz_to_obj(source, obj_path)
        mesh = load_obj_mesh(obj_path)
        metadata = mesh_metadata(mesh)
        save_mesh_preview(mesh, preview_path)
    except (ConversionError, GeometryError, OSError, ValueError, RuntimeError) as exc:
        shutil.rmtree(model_dir)
        raise RoomModelError(f"Room model processing failed: {exc}") from exc

    return {
        "model_id": model_id,
        "source_path": str(source),
        "obj_path": str(obj_path),
        "preview_path": str(preview_path),
        "mesh": metadata,
    }
