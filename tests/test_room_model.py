"""Focused tests for OBJ handling and the one-call processing contract."""

from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from room_model import RoomModelError, process_room_model
from room_model.converter import ConversionError, convert_usdz_to_obj, find_blender
from room_model.geometry import GeometryError, load_obj_mesh, mesh_metadata


# Two separate OBJ objects, one displaced along X.
TWO_TRIANGLES = """o first
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
o second
v 10 0 0
v 11 0 0
v 10 1 0
f 4 5 6
"""


class RoomModelTests(unittest.TestCase):
    def test_multi_object_obj_and_metadata(self):
        with TemporaryDirectory() as temp:
            obj = Path(temp) / "two.obj"
            obj.write_text(TWO_TRIANGLES, encoding="utf-8")
            mesh = load_obj_mesh(obj)
            info = mesh_metadata(mesh)
            self.assertEqual(info["faces"], 2)
            self.assertEqual(info["vertices"], 6)
            self.assertEqual(info["bounds"], [[0.0, 0.0, 0.0], [11.0, 1.0, 0.0]])
            self.assertEqual(info["center"], [5.5, 0.5, 0.0])

    def test_empty_obj_rejected(self):
        with TemporaryDirectory() as temp:
            obj = Path(temp) / "empty.obj"
            obj.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(GeometryError, "missing or empty"):
                load_obj_mesh(obj)

    def test_pipeline_produces_unique_outputs_and_png(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scan.usdz"
            source.write_bytes(b"placeholder for mocked converter")

            def fake_convert(_source, obj_path):
                obj_path.write_text(TWO_TRIANGLES, encoding="utf-8")
                return obj_path

            with patch("room_model.pipeline.convert_usdz_to_obj", side_effect=fake_convert):
                first = process_room_model(source, storage_root=root / "models")
                second = process_room_model(source, storage_root=root / "models")

            self.assertNotEqual(first["model_id"], second["model_id"])
            for result in (first, second):
                self.assertEqual(result["mesh"]["faces"], 2)
                self.assertTrue(Path(result["obj_path"]).is_file())
                self.assertEqual(Path(result["preview_path"]).read_bytes()[:8],
                                 b"\x89PNG\r\n\x1a\n")

    def test_invalid_source_rejected_before_conversion(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            wrong = root / "scan.obj"
            wrong.write_bytes(b"data")
            with self.assertRaisesRegex(RoomModelError, "Expected a .usdz"):
                process_room_model(wrong, storage_root=root / "models")
            self.assertFalse((root / "models").exists())

    def test_failure_removes_partial_model(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "scan.usdz"
            source.write_bytes(b"placeholder for mocked converter")
            with patch("room_model.pipeline.convert_usdz_to_obj",
                       side_effect=RuntimeError("conversion failed")):
                with self.assertRaisesRegex(RoomModelError, "conversion failed"):
                    process_room_model(source, storage_root=root / "models")
            self.assertEqual(list((root / "models").iterdir()), [])

    def test_blender_command_is_argument_array_and_errors_are_reported(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "room scan.usdz"
            output = Path(temp) / "room model.obj"
            with patch("room_model.converter.find_blender", return_value="blender"), \
                 patch("room_model.converter.subprocess.run") as run:
                run.return_value.returncode = 7
                run.return_value.stdout = ""
                run.return_value.stderr = "bad USD scene"
                with self.assertRaisesRegex(ConversionError, "bad USD scene"):
                    convert_usdz_to_obj(source, output)
                command = run.call_args.args[0]
                self.assertEqual(command[0], "blender")
                self.assertIn("--disable-autoexec", command)
                self.assertEqual(command[-2:], [str(source), str(output)])
                self.assertEqual(run.call_args.kwargs["timeout"], 600)

    def test_blender_override_error(self):
        with patch.dict("room_model.converter.os.environ",
                        {"BLENDER_EXECUTABLE": "missing/blender"}):
            with patch("room_model.converter.shutil.which", return_value=None):
                with self.assertRaisesRegex(ConversionError, "BLENDER_EXECUTABLE"):
                    find_blender()

    def test_blender_timeout_and_missing_output(self):
        with TemporaryDirectory() as temp:
            output = Path(temp) / "model.obj"
            with patch("room_model.converter.find_blender", return_value="blender"), \
                 patch("room_model.converter.subprocess.run") as run:
                run.side_effect = subprocess.TimeoutExpired(["blender"], 3)
                with self.assertRaisesRegex(ConversionError, "timed out after 3"):
                    convert_usdz_to_obj("room.usdz", output, timeout=3)
                run.side_effect = None
                run.return_value.returncode = 0
                run.return_value.stdout = "finished"
                run.return_value.stderr = ""
                with self.assertRaisesRegex(ConversionError, "without a non-empty OBJ"):
                    convert_usdz_to_obj("room.usdz", output)


if __name__ == "__main__":
    unittest.main()
