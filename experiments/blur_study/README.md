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

## Convert MVS-Synth

Place the raw GTAV_540 recording directories under `data/mvs-synth/`. Run from `vggt-omega/`.

```bash
python experiments/blur_study/dataset_prep/convert_mvssynth.py --workers 6
```

The converter writes selected recordings under `data/unified/mvssynth/`. Use `--scenes 0000` to convert a single recording. See the [MVS-Synth mapping](dataset_prep/MAPPING.md#mvs-synth-gtav_540).

## Build the shared schedule and prepare source validation

```bash
test -f experiments/blur_study/schedule.json || python experiments/blur_study/build_schedule.py
python eval/prepare_source_val.py                    # -> data/source_val/, eval/frames/source_val.json
```

The schedule fixes all of the following for every recipe:

- **Passes:** 5 passes over the 450 training recordings. Each pass visits every recording once, in 150 batches of 3 sequences.
- **Batch shape:** a discrete aspect ratio per batch, with the 416² pixel budget.
- **Per sample:**
  - 10 frame ids, drawn with each dataset's native `UnifiedDataset` sampler
  - an augmentation seed
  - a permutation of views 1–9
  - each recipe's blur count:
    - fixed: 2
    - variable: exactly 150 each of 1/2/3 per pass
    - clean-or-4: exactly 225 each of 0/4 per pass

  A recipe blurs the first *count* views of that permutation.

## 3. Train (Linux, NCCL, one recipe per GPU)

```bash
bash experiments/blur_study/run_linux.sh     # 2 GPUs: 2 recipes at a time; 1 GPU: sequential
```

Single recipe: `cd training && torchrun --standalone --nproc_per_node=1 launch.py --config blur_study recipe=fixed`.

The config is `training/config/blur_study.yaml`. It uses the authors' benchmark recipe:
- a 416² budget with the authors' augmentation block
- AdamW at lr 2e-4, weight decay 0.05, betas (0.9, 0.95)
- 5% warmup, then cosine decay
- gradient clipping at 1.0, bf16, gradient checkpointing
- loss weights from the released code
- 3 sequences × 10 views per step, from `FRAME_COUNT_TO_BATCH_SIZE`

The study adds:
- LoRA (r=16, α=32) on the aggregator frame and global blocks and the camera head
- 5 epochs × 150 steps
- a start from `vggt_omega_1b_416_reproduce.pt`, loaded strictly

On the local Ada, peak memory at batch 3 was 16 GB.

Each run writes `training/logs/blur_study_<recipe>/checkpoints/model_merged.pt`. `export_merged.py` rebuilds merged weights from checkpoint.pt, verifies strict loading, and writes them to `checkpoint/blur_study/<recipe>.pt`.
