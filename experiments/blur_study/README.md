# VGGT-Ω motion-blur fine-tuning study: how to run it

Use the supplied `data/vggt_omega_train_validation_v2.csv` and `dataset_prep/uco3d_shard_index.json`. The manifest records 102 uCO3D substitutions in its `replaces` column. Each replacement retains its split and cohort. The shard index locates the 250 selected recordings.

Run from `vggt-omega/`.

```bash
python experiments/blur_study/dataset_prep/fetch_uco3d.py fetch
```

## Convert Hypersim

Place the raw Hypersim scenes and `metadata_camera_parameters.csv` under `data/hypersim/`. Install `h5py` and `scipy` in the study environment. Run the converter from the repository root.

```bash
python experiments/blur_study/dataset_prep/convert_hypersim.py --workers 6
```

Each selected camera trajectory is written under `data/unified/hypersim/`. Use `--scenes ai_001_001_cam_00` to convert a single recording. See [the Hypersim mapping](dataset_prep/MAPPING.md#hypersim) for camera and depth conventions.
