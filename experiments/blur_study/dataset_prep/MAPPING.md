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
