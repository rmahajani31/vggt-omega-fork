# Evaluation

Evaluation code for VGGT-Omega on ETH3D and Sintel. This codebase is an agent-assisted
cleanup of our original evaluation implementation. If you encounter any problems, please open a
[GitHub issue](https://github.com/facebookresearch/vggt-omega/issues).

Exact results can vary across machines and
software environments, for example, AUC may differ by up to two points, while the relative
trends between methods remain consistent.

## Install

```bash
pip install -r requirements.txt
pip install -e .
pip install -r eval/requirements.txt
```

## Prepare data

```bash
python eval/prepare.py eth3d --output data/eth3d
python eval/prepare.py sintel --output data/sintel
```

Preparation uses local archives when available, otherwise tries the official dataset
server and then a pinned Hugging Face mirror. Use `--source official`, `--source hf`, or
`--source local --archive-dir /path/to/archives` to choose explicitly. To convert an
already extracted official dataset, pass `--raw-root`.

Prepared data has the same layout for both datasets:

```text
data/<dataset>/<scene>/
├── frames.txt
├── cameras.npz
├── images/
└── depths/
```

## Run

```bash
python eval/evaluate.py \
  --dataset eth3d \
  --data-root data/eth3d \
  --checkpoint /path/to/vggt_omega_1b_416_reproduce.pt \
  --output outputs/eth3d.json
```

Change `eth3d` to `sintel` for Sintel.

## Output

The output JSON contains `AUC@3`, `AUC@30`, `delta125`, and `AbsRel`, together with
per-scene values, frame names, dataset counts, and runtime versions. Use
`--max-scenes 1` for a quick smoke run.

## Controlled motion blur on Sintel and ETH3D

From the VGGT-Omega repository root, use the VGGT lab interpreter. Prepare
each dataset once. The sweeps use the ten views per scene specified in the
released frame lists for all 16 Sintel sequences and 13 ETH3D scenes.

```bash
../.venv/Scripts/python.exe eval/prepare.py sintel --raw-root data/sintel_raw --output data/sintel
../.venv/Scripts/python.exe eval/prepare.py eth3d --output data/eth3d
../.venv/Scripts/python.exe eval/motion_blur.py --dataset sintel --seed 42
../.venv/Scripts/python.exe eval/motion_blur.py --dataset eth3d --seed 42
```

Each run scores the same ten benchmark views and 45 camera pairs. The baseline
uses ten clean images. Subsequent runs blur 1, 2, 3, 4, or 5 images, keeping the
first view clean. The one-blurred-view split tests each of the nine eligible
views once. The other counts use ten distinct seeded selections each. For a shared
seed the selections are nested across counts. This gives 50 inferences per
scene. Use `--scenes alley_1` to inspect a single scene.

Every invocation of `motion_blur.py` clears its entire output directory before
processing the first scene, including when using `--scenes`. The default is
the dataset-specific folder. Use `--output-dir` to select a separate directory
for a debug run.

The horizontal box filter spans 61 pixels at the prepared Sintel width of 846
pixels. On ETH3D it spans the same fraction of the prepared image width,
rounded to the nearest odd pixel count. Filtering precedes predictor cropping
and resizing.

Under `outputs/motion_blur/`, each dataset folder contains
`<scene>/results.json` with the baseline, trial scores, and means by split.
Split labels give clean and blurred counts, so `7_3` means 30% blurred.
The runner prints absolute AUC@30, AUC@3, delta125, and AbsRel for each
coverage level. Each completed run writes `coverage_metrics.json` with the
mean for each split across the scenes selected for that run. Top-level split
keys hold those averages. Under `per_scene`, each scene name maps to its own
scores for the same splits.
AUC and delta125 are percentages (higher is better). Lower AbsRel is better.

In the dataset averages, trials have equal weight within each blur level
and each scene has equal weight.
