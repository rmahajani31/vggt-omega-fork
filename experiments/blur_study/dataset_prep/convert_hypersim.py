"""Convert Hypersim camera trajectories (<scene>_<cam>) to the unified layout.

Mapping (see MAPPING.md and hypersim_io.py, verified by reprojection on every camera type):
  * one unified scene per (scene, camera trajectory), named like the authors' valid_seqs
    (e.g. ai_001_001_cam_00); frames are associated by the 4-digit frame id shared by
    final_preview/frame.XXXX.tonemap.jpg, geometry_hdf5/frame.XXXX.depth_meters.hdf5 and the
    camera keyframe arrays (camera_keyframe_frame_indices);
  * RGB is Hypersim's tone-mapped preview JPEG, copied byte-for-byte (resampled to PNG only for
    the few skewed tilt-shift cameras);
  * depth_meters is Euclidean range in metres; it is converted to planar z of the (possibly
    tilt-rotated) OpenCV camera; NaN -> 0;
  * camera poses: R_world_from_cam and positions * meters_per_asset_unit, Hypersim camera axes
    (x right, y up, z back) -> OpenCV, then the tilt rotation from the RQ decomposition of the
    scene's M_cam_from_uv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from convert_common import REPOSITORY, base_parser, manifest_rows, run, select

VERSION = "hypersim-1"


def build(scene: str, args: dict):
    import cv2

    import hypersim_io as h
    from unified_io import Frame

    source = Path(args["source"])
    row = args["catalog"][scene]
    scene_dir, camera_name = source / row["group_id"], row["camera_trajectory"]
    camera = h.opencv_camera(h.camera_parameters(str(source / "metadata_camera_parameters.csv"))[row["group_id"]])
    mpau = h.meters_per_asset_unit(scene_dir)
    _, orientations, positions = h.load_trajectory(scene_dir, camera_name)
    records, skipped = h.frame_records(scene_dir, camera_name)
    frames = []
    for frame_id, jpg, depth_path, k in records:
        range_map = h.read_range(depth_path)
        if range_map.shape != (camera.height, camera.width):
            skipped.append((frame_id, f"depth shape {range_map.shape}"))
            continue
        depth = h.planar_depth(range_map, camera)
        pose = h.cam_from_world(orientations[k], positions[k] * mpau, camera.R_tilt)
        if camera.warp is None:
            frames.append(Frame(name=f"{frame_id:05d}.jpg", cam_from_world=pose, K=camera.K, depth=depth,
                                image_path=jpg, source_id=jpg.name))
        else:
            image = cv2.imread(str(jpg), cv2.IMREAD_COLOR)[:, :, ::-1]
            image, depth = h.warp_to_pinhole(image, depth, camera)
            frames.append(Frame(name=f"{frame_id:05d}.png", cam_from_world=pose, K=camera.K, depth=depth,
                                image=np.ascontiguousarray(image), source_id=jpg.name))
    return frames, skipped


def main() -> None:
    parser = base_parser(__doc__, "hypersim", REPOSITORY / "data" / "hypersim")
    args = parser.parse_args()
    rows = manifest_rows("Hypersim", args.manifest)
    scenes = select(rows, args)
    catalog = {r["catalog_id"]: {"group_id": r["group_id"], "camera_trajectory": r["camera_trajectory"]}
               for r in rows}
    run("hypersim", build, scenes, args, VERSION,
        {"source": str(Path(args.source).resolve()), "catalog": catalog},
        {"dataset": "Hypersim v1", "url": "https://github.com/apple/ml-hypersim",
         "rgb": "final_preview tonemap.jpg", "depth": "geometry_hdf5 depth_meters (range -> planar z)"})


if __name__ == "__main__":
    main()
