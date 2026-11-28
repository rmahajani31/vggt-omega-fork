# VGGT-Ω motion-blur fine-tuning study: how to run it

This study fine-tunes VGGT-Ω with LoRA under six blur recipes. All six see an identical sequence of samples and differ only in which views are blurred and how strongly. The blur is a horizontal box blur applied to the source image, with view 0 always clean.

- Four recipes use a single width of 31 px at an 846 px reference width: clean, fixed-2, variable 1/2/3 and clean-or-4.
- Two recipes, `variable_random_width` and `ramped_random_width`, give each blurred view its own width of 2–10% of the image width (section 6).

Each recipe is evaluated alongside the original checkpoint on three cohorts: 50 held-out source recordings, Sintel and ETH3D. Inference uses `max_size` 512.

All commands below run from `vggt-omega/`.

## 0. Environment

```bash
pip install -r requirements.txt -r eval/requirements.txt && pip install -e ".[train]" && pip install h5py scipy requests
```

## 1. Data (`dataset_prep/`, see `dataset_prep/MAPPING.md`)

Use the supplied `data/vggt_omega_train_validation_v2.csv` and `dataset_prep/uco3d_shard_index.json`. The manifest records 102 uCO3D substitutions in its `replaces` column. Each replacement retains its split and cohort. The shard index locates the 250 selected recordings.

For preparation from source, place Hypersim under `data/hypersim` and MVS-Synth under `data/mvs-synth`. For an existing prepared dataset, start with the validation and split commands.

```bash
python experiments/blur_study/dataset_prep/fetch_uco3d.py fetch
python experiments/blur_study/dataset_prep/convert_hypersim.py --workers 6
python experiments/blur_study/dataset_prep/convert_mvssynth.py --workers 6
for d in hypersim mvssynth uco3d; do python experiments/blur_study/dataset_prep/validate_unified.py "$d" --full; done
python experiments/blur_study/prepare_splits.py --check-data
```

The saved uCO3D report fails the geometry comparison despite complete coverage and successful loader probes. See `dataset_prep/MAPPING.md` for the measured controls at three frame gaps.

Converted data is written to `data/unified/{uco3d,hypersim,mvssynth}/`. Copy or rsync that folder to the Linux box, or point `BLUR_STUDY_DATA` at it.

## 2. Shared schedule and validation cohort

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
    - variable_random_width: the same counts as variable
    - ramped_random_width: exactly 45/90/135/180 with 0/1/2/3 per pass
  - for the two random-width recipes, a width per blurred view (`width_fraction_by_view`)

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

## 4. Evaluate and summarize

The evaluation wrapper reads prepared Sintel and ETH3D scenes. If those folders are missing, run `python eval/prepare.py sintel --output data/sintel` and `python eval/prepare.py eth3d --output data/eth3d` first.

```bash
bash eval/run_study_eval.sh     # original + 4 recipes x {source_val, sintel, eth3d}, 31 px @ 846
# -> outputs/blur_study/summary/{source_val,sintel,eth3d}.{md,csv}
```

## 5. Save and inspect a Linux run

Each recipe writes its console output to `training/logs/blur_study_<recipe>/stdout.log`, its training log to `logs/log.txt` inside that directory, its TensorBoard events to `tensorboard/`, and its weights and optimizer checkpoint to `checkpoints/`. The training wrapper opens a fresh `stdout.log` on each invocation. The evaluation wrapper replaces each selected model and cohort's results directory. To retain a run, archive its outputs before starting another run in the same directories.

After a completed study, these commands archive the run artifacts and record the Git revision. Run from the repository root using the default output paths.

```bash
run_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
tar -cf "../blur-study-${run_id}.tar" \
  training/logs/blur_study_* checkpoint/blur_study outputs/blur_study \
  experiments/blur_study/schedule.json data/vggt_omega_train_validation_v2.csv \
  data/source_val/selection.json data/source_val/manifest_with_source_val_frames.csv
```

After extracting the archive, run `tensorboard --logdir training/logs` from the extraction directory. Compare the training objective and component losses across steps and recipes. The trainer records pre-clipping gradient norms by parameter group, skipping the first batch of each epoch for those norms. The held-out evaluation scores are in `outputs/blur_study/summary/`.

## 6. Random-width recipes `variable_random_width` and `ramped_random_width`

These two recipes replace the single 31 px width with a width per blurred view. Each width is drawn uniformly from 2–10% of that view's source image width, about 17–85 px at 846 px. `build_schedule.py` adds them to `schedule.json` next to the four single-width recipes. They use their own seeded generators, so adding them leaves every frame, augmentation seed, view order, aspect ratio and single-width blur count unchanged.

| Recipe | Samples with 0 / 1 / 2 / 3 blurred views per pass | Compared with | Isolates |
|---|---:|---|---|
| `variable_random_width` | 0 / 150 / 150 / 150 (exactly the views `variable` blurs) | `variable` | the width distribution |
| `ramped_random_width` | 45 / 90 / 135 / 180 (the same 900 widths per pass) | `variable_random_width` | how blurred views are spread over samples |

`ramped_random_width` keeps `variable_random_width`'s width on every view both recipes blur. It reassigns the 45 widths it frees to the 45 views it newly blurs. `validate_schedule` checks these invariants on load, and the random-width recipes need a multiple of 30 recordings per pass, which matters for `--limit` smoke schedules.

Train them like any other recipe, for example `torchrun --standalone --nproc_per_node=1 launch.py --config blur_study recipe=variable_random_width`. They are part of `run_linux.sh`'s default list. `ScheduledBlurDataset` passes each sample's `width_fraction_by_view[recipe]` as `blur_cfg={"width_fractions": ...}`, and `UnifiedDataset.get_data` blurs view *i* with a kernel of `fraction[i] × source width` px, rounded to the nearest odd number.

For evaluation, `eval/motion_blur.py` takes two options:

- `--blur-px` sets a single width at the 846 px reference, for example 16.92 / 33.84 / 50.76 / 67.68 / 84.6 for 2 / 4 / 6 / 8 / 10%.
- `--mixed-severity` sets the `mixed_uniform` condition. Each blurred frame gets its own uniform 2–10% width, seeded by frame name.

**Earlier names.** The 2026-11-27 run, in `outputs/randomized-blur-width2to10/20261127T185344Z`, called these recipes `width2to10_a` (now `variable_random_width`) and `width2to10_b` (now `ramped_random_width`). The current `schedule.json` reproduces that run's blur counts and widths exactly.
