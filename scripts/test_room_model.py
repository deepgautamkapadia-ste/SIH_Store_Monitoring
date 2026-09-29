"""Manual end-to-end check with a real USDZ file and Blender."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from room_model import RoomModelError, process_room_model


def main():
    parser = argparse.ArgumentParser(description="Convert and inspect a USDZ room model")
    parser.add_argument("usdz_path", help="path to a non-empty .usdz file")
    args = parser.parse_args()
    try:
        result = process_room_model(args.usdz_path)
    except RoomModelError as exc:
        parser.exit(1, f"Conversion failed: {exc}\n")
    print("Conversion status: success")
    print(f"OBJ: {result['obj_path']}")
    print(f"Preview: {result['preview_path']}")
    print(f"Vertices: {result['mesh']['vertices']}")
    print(f"Faces: {result['mesh']['faces']}")
    print(f"Dimensions: {result['mesh']['dimensions']}")


if __name__ == "__main__":
    main()
