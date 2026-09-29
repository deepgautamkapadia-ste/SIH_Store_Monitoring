"""Run inside Blender: --python usdz_to_obj.py -- input.usdz output.obj."""

from pathlib import Path
import sys

import bpy


def main():
    if "--" not in sys.argv or len(sys.argv[sys.argv.index("--") + 1:]) != 2:
        raise SystemExit("Expected -- input.usdz output.obj")
    source, target = map(Path, sys.argv[sys.argv.index("--") + 1:])

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    if "FINISHED" not in bpy.ops.wm.usd_import(filepath=str(source)):
        raise RuntimeError(f"USDZ import failed: {source}")

    for obj in list(bpy.context.scene.objects):
        if obj.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # Keep only actual mesh objects; parent empties may carry transforms, so
    # leave them in the scene and simply select the meshes for export.
    bpy.ops.object.select_all(action="DESELECT")
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("USDZ scene contains no mesh objects")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]

    if not hasattr(bpy.ops.wm, "obj_export"):
        raise RuntimeError("Blender 4.0 or newer is required for built-in OBJ export")
    if "FINISHED" not in bpy.ops.wm.obj_export(
        filepath=str(target),
        export_selected_objects=True,
        apply_modifiers=True,
        apply_transform=True,
        export_triangulated_mesh=True,
        export_uv=False,
        export_materials=False,
        export_normals=True,
        forward_axis="Y",
        up_axis="Z",
    ):
        raise RuntimeError(f"OBJ export failed: {target}")


if __name__ == "__main__":
    main()
