# Blur-study dataset mapping notes

This note covers how the three training sources are converted to the VGGT-Ω unified layout. The conversion follows the authors' `training/dataset_preparation/download_and_convert.md`.

- **Scene names** are the manifest `catalog_id`s, which equal the names in the authors' `training/valid_seqs/*.txt`.
- **Output location:** converted data is written to `data/unified/<dataset>/`.
- **Helper directories:** `_staging`, `_records`, `_logs` and `_reports` sit next to the dataset roots, never inside them.

Each convention below was checked by depth reprojection between frames 4 apart, using `geometry.py` and `validate_unified.py`. The scores are inlier fractions at 1% relative depth error. Every competing interpretation was scored on the same pairs, as the guide requires.

## uCO3D: the authors' re-annotation

| | |
|---|---|
| Source | `huggingface.co/datasets/facebook/uco3d`, `vggt_omega_anno/*.tar` (228 uncompressed shards, 7.6 TB). |
| Layout | Each shard is a tar of per-sequence tars `<seq>-<md5>.tar`. Each one's root `<seq>-<md5>/` is already in the unified layout: `images/`, `depths/*.exr`, `cam_from_worlds.npy`, `intrinsics.npy`, `image_names.json`. |
| Naming | Manifest/valid_seqs id `<seq>_rgb_video_<md5>` corresponds to member `<seq>-<md5>.tar`, which sits in shard `<md5[:2]>` (large prefixes are split as `<xx>-00`, `<xx>-01`, …). |
| Transfer | The study shard index records the byte range for each of the 250 selected sequences. `fetch_uco3d.py fetch` streams just the 250 manifest members and extracts them safely: no absolute paths, no `..`, regular files and directories only. Each scene is validated in staging, then renamed to its `catalog_id`. |
| Conversion | None. These are the authors' own unified files, so they are only validated. |
| Not used | The local raw release in `data/uco3d` (`rgb_video.mp4`, `depth_maps.h5`, `metadata.sqlite`). Its depth is DepthAnythingV2 and its cameras are the original uCO3D ones, not what the checkpoint was trained on. |
| Loader entry | The authors' `uco3d` block, unchanged (`frame_window 256`, `max_interval 64`, `max_depth 256`, `sample_by_index true`). |
| Validation (250/250 scenes, 44,279 frames) | Structural checks: 0 failures. Loader probe: 0 errors. |
| Geometry (inliers at 1%, 20 scenes × 6 pairs) | The documented convention scores best at every gap. The `validate_unified.py` gate ("documented > 2× the best control") still reports **fail**. These are slow object orbits (~200 frames per video), so a focal error and a one-frame shift move points only slightly, and those controls stay high. See the table below. |
| Inverted-pose control | Fails at every gap, so the pose direction is confirmed. |
| Reports | `data/unified/_reports/uco3d_validation*.json`. |

uCO3D geometry inlier fractions by frame gap:

| Frame gap | Documented | Inverted pose | Focal +10% | One-frame shift |
|---|---:|---:|---:|---:|
| 4 | 0.99 | 0.12 | 0.92 | 0.55 |
| 16 | 0.93 | 0.03 | 0.60 | 0.53 |
| 32 | 0.84 | 0.03 | 0.43 | 0.49 |

The gap between the documented convention and the controls widens as the baseline grows, which is the expected pattern for correct calibration. No conversion was applied, so these numbers only check the authors' annotation.

## Hypersim

| | |
|---|---|
| Scene | One per (scene, camera trajectory), e.g. `ai_001_001_cam_00`. Frames are associated by the shared 4-digit frame id in `final_preview/frame.XXXX.tonemap.jpg`, `geometry_hdf5/frame.XXXX.depth_meters.hdf5` and `_detail/<cam>/camera_keyframe_frame_indices.hdf5`. Frames missing any modality are skipped and logged. |
| RGB | Tone-mapped preview JPEG, copied byte-for-byte (`00042.jpg`). |
| Depth | `depth_meters` is Euclidean range in metres, NaN where there is no geometry. Planar z of the OpenCV camera is `z = r · (R_t a)_z / ‖a‖` with `a = A p`. Invalid values become 0. |
| Pose | `R_world_from_cam` comes from `camera_keyframe_orientations` and the centre from `camera_keyframe_positions × meters_per_asset_unit`. Hypersim camera axes (x right, y up, z back) map to OpenCV via `D = diag(1,−1,−1)`: `cam_from_world = R_t · [D Rᵀ | −D Rᵀ c]`. |
| Intrinsics | From the official `metadata_camera_parameters.csv` `M_cam_from_uv`. `A = D M N` maps pixel centres to OpenCV rays, and RQ decomposition gives `A⁻¹ = K_s R_t`. |
| Camera types (140 manifest scenes) | 85 centred, 3 lens-shifted, 49 tilt-shift with zero skew (image plane tilted ≤ 11°, handled exactly by `R_t` with no resampling), and 3 tilt-shift with skew up to 12 px. The skewed ones are resampled with the homography `K K_s⁻¹` (cubic for RGB, nearest for depth, written as PNG). |
| Evidence (gap 4) | Documented conversion 0.75–0.98. Unflipped axes ≤ 0.05, range used as z ≤ 0.30, orientation transposed ≤ 0.03. Ignoring tilt/shift drops to 0.52–0.69 on tilted or shifted scenes. Focal +10%: 0.15. |
| Loader entry | The authors' EDEN (synthetic) block, with `sample_by_index: true`. |

## MVS-Synth (GTAV_540)

| | |
|---|---|
| Scene | One per sequence directory. `images/XXXX.png`, `depths/XXXX.exr` and `poses/XXXX.json` share the frame id. All 110 manifest sequences have 100 frames. |
| RGB | PNG at 960×540, copied byte-for-byte (`00000.png`, …). |
| Depth | Planar z in GTA metres. Sky is `inf` (~12% of pixels), and it and any other invalid value become 0. Depth is not clipped (no `max_depth`, as in the authors' EDEN entry). |
| Pose | `extrinsic` is a 4×4 camera-from-world matrix in OpenCV camera axes. Its rotation has det = −1 because GTA's world frame is left-handed. A world reflection `G = diag(1,1,−1,1)` makes the rotations proper (`cam_from_world = E·G`) without changing any camera-frame quantity. |
| Intrinsics | `f_x, f_y, c_x = 480, c_y = 270` (pixel-centre convention). |
| Evidence | Documented convention 0.91–0.97 (gap 1–4). Inverted pose ≤ 0.04, camera-axis flips ≤ 0.28, depth treated as range ≤ 0.11, focal +10% 0.28 (gap 4). |
| Observation for later cleaning | Consecutive frames sometimes share an identical pose; these are kept. |
| Loader entry | The authors' EDEN (synthetic) block, with `sample_by_index: true`. |
