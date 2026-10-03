"""Launch Blender in the background to convert USDZ geometry to OBJ."""

import os
from pathlib import Path
import shutil
import subprocess


class ConversionError(RuntimeError):
    """Blender could not convert the input scene."""


def find_blender():
    """Resolve Blender from an override, PATH, or common Windows installs."""
    override = os.environ.get("BLENDER_EXECUTABLE")
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        on_path = shutil.which(override)
        if on_path:
            return on_path
        raise ConversionError(
            f"BLENDER_EXECUTABLE points to missing Blender executable: {override}"
        )

    on_path = shutil.which("blender")
    if on_path:
        return on_path

    if os.name == "nt":
        for variable in ("ProgramFiles", "ProgramFiles(x86)"):
            base = os.environ.get(variable)
            if not base:
                continue
            foundation = Path(base) / "Blender Foundation"
            for candidate in sorted(foundation.glob("Blender */blender.exe"), reverse=True):
                if candidate.is_file():
                    return str(candidate)

    raise ConversionError(
        "Blender was not found. Install Blender and add 'blender' to PATH, "
        "or set BLENDER_EXECUTABLE to the full executable path."
    )


def convert_usdz_to_obj(usdz_path, obj_path, *, timeout=600):
    """Run the bundled Blender script and return the generated OBJ path."""
    blender = find_blender()
    script = Path(__file__).parent / "blender_scripts" / "usdz_to_obj.py"
    command = [
        blender, "--background", "--factory-startup", "--disable-autoexec",
        "--python-exit-code", "1",
        "--python", str(script), "--", str(usdz_path), str(obj_path),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, errors="replace",
            timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ConversionError(f"Blender conversion timed out after {timeout} seconds") from exc
    except OSError as exc:
        raise ConversionError(f"Could not start Blender: {exc}") from exc

    output = "\n".join(part for part in (result.stdout, result.stderr) if part)
    if result.returncode != 0:
        raise ConversionError(
            f"Blender conversion failed (exit {result.returncode}). "
            f"Output:\n{output[-4000:]}"
        )
    obj_path = Path(obj_path)
    if not obj_path.is_file() or obj_path.stat().st_size == 0:
        raise ConversionError(
            f"Blender finished without a non-empty OBJ at {obj_path}. "
            f"Output:\n{output[-4000:]}"
        )
    return obj_path
