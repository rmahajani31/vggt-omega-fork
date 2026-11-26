"""Convert MVS-Synth (GTAV_540) sequences to the unified layout.

Mapping (see MAPPING.md, verified by reprojection on real pairs):
  * one scene per sequence directory <seq>/ with images/XXXX.png, depths/XXXX.exr,
    poses/XXXX.json; records are associated by the shared 4-digit frame id;
  * ``extrinsic`` is a 4x4 camera-from-world matrix in OpenCV camera axes (x right, y down,
    z forward), but its 3x3 block has det = -1 because GTA's world frame is left-handed. A
    world-axis reflection G = diag(1, 1, -1) is applied (cam_from_world = E @ G), which makes
    the rotations proper and leaves every camera-frame quantity (and hence all relative
    poses, depth and projections) unchanged;
  * depth EXR is planar z in metres (GTA units); inf (sky) and non-positive values -> 0;
  * intrinsics are pinhole f_x, f_y, c_x, c_y in pixels of the 960 x 540 image;
  * images are copied byte-for-byte; output frames are renamed 00000.png, 00001.png, ...
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from convert_common import REPOSITORY, base_parser, manifest_rows, run, select

VERSION = "mvssynth-1"
WORLD_REFLECTION = np.diag([1.0, 1.0, -1.0, 1.0])


def cam_from_world(extrinsic) -> np.ndarray:
    E = np.asarray(extrinsic, dtype=np.float64)
    if E.shape != (4, 4) or not np.allclose(E[3], [0, 0, 0, 1]):
        raise ValueError(f"Unexpected extrinsic {E.tolist()}")
    return (E @ WORLD_REFLECTION)[:3]


def build(scene: str, args: dict):
    import cv2
    from unified_io import Frame

    root = Path(args["source"]) / scene
    ids = sorted(p.stem for p in (root / "images").glob("*.png"))
    frames, skipped = [], []
    for i, frame_id in enumerate(ids):
        depth_path, pose_path = root / "depths" / f"{frame_id}.exr", root / "poses" / f"{frame_id}.json"
        if not depth_path.is_file() or not pose_path.is_file():
            skipped.append((frame_id, "missing depth or pose"))
            continue
        pose = json.loads(pose_path.read_text())
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if depth is None or depth.ndim != 2:
            skipped.append((frame_id, "undecodable depth"))
            continue
        K = np.array([[pose["f_x"], 0, pose["c_x"]], [0, pose["f_y"], pose["c_y"]], [0, 0, 1]], dtype=np.float64)
        frames.append(Frame(name=f"{len(frames):05d}.png", cam_from_world=cam_from_world(pose["extrinsic"]),
                            K=K, depth=depth, image_path=root / "images" / f"{frame_id}.png",
                            source_id=frame_id))
    return frames, skipped


def main() -> None:
    parser = base_parser(__doc__, "mvssynth", REPOSITORY / "data" / "mvs-synth")
    args = parser.parse_args()
    scenes = select(manifest_rows("MVS-Synth", args.manifest), args)
    run("mvssynth", build, scenes, args, VERSION, {"source": str(Path(args.source).resolve())},
        {"dataset": "MVS-Synth GTAV_540", "url": "https://huggingface.co/datasets/phuang17/MVS-Synth"})


if __name__ == "__main__":
    main()
