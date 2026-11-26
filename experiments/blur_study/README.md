# VGGT-Ω motion-blur fine-tuning study: how to run it

Use the supplied `data/vggt_omega_train_validation_v2.csv` and `dataset_prep/uco3d_shard_index.json`. The manifest records 102 uCO3D substitutions in its `replaces` column. Each replacement retains its split and cohort. The shard index locates the 250 selected recordings.

Run from `vggt-omega/`.

```bash
python experiments/blur_study/dataset_prep/fetch_uco3d.py fetch
```
