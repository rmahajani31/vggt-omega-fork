# Blur-study dataset mapping notes

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
