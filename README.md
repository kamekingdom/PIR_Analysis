# PIR Analysis

PIR sensor time-series to Theia3D skeleton-coordinate regression.

This repository contains trimmed PIR/Theia3D data, preprocessing scripts, baseline ridge regression, GPU sequence models, model comparison utilities, ensembling, and a browser-based 3D prediction viewer.

## Data Layout

- `PIR_gpu/data/<trial>/pir/pir_data_192_168_11_x.csv`
  - PIR modules, 5 channels per module.
- `PIR_gpu/data/<trial>/skelton/Theia_Sub0.csv`
  - Theia3D global joint coordinates at 25 Hz.
- `PIR_gpu/data/<trial>/skelton/<trial>.tsv`
  - Theia metadata including frequency and timestamp.

The project intentionally keeps the existing `skelton` spelling used by the source data.

## Main Results

Held-out split:

- train: trials `002`, `003`, `004`
- test: trial `005`

Best result from prediction averaging ensemble:

| model | RMSE [mm] | MAE [mm] | mean joint error [mm] | R2 |
|---|---:|---:|---:|---:|
| Ensemble_TCN2_TCN4_TCN6_GRU6 | 240.1 | 138.0 | 312.4 | 0.963 |
| TCN_4s | 260.5 | 153.4 | 347.9 | 0.957 |
| Ridge baseline | 1184.1 | 829.7 | 1889.9 | 0.103 |

The full comparison table is in:

- `PIR_gpu/model_output/model_comparison/experiment_report.md`
- `PIR_gpu/model_output/model_comparison/model_comparison.csv`

## Setup

```bash
python -m pip install numpy torch
```

CUDA was tested with an RTX 4070 Ti SUPER. In this environment, PyTorch GPU access required running outside the restricted sandbox.

## Reproduce Baseline And Aligned Dataset

```bash
python PIR_gpu/data/train_pir_to_global_bones.py \
  --trimmed-dir PIR_gpu/data \
  --output-dir PIR_gpu/model_output/baseline_ridge_v3 \
  --pir-time-source auto \
  --sync-mode timestamps
```

This creates the 25 Hz aligned dataset:

```text
PIR_gpu/model_output/baseline_ridge_v3/aligned_dataset.npz
```

Large generated artifacts such as aligned datasets, model weights, prediction CSVs, and viewer JSON are ignored by Git. They can be regenerated with the scripts below.

## Train Sequence Models

Example best single model:

```bash
python PIR_gpu/data/train_pir_sequence_model.py \
  --dataset PIR_gpu/model_output/baseline_ridge_v3/aligned_dataset.npz \
  --output-dir PIR_gpu/model_output/sequence_tcn4s_gpu \
  --model tcn \
  --test-trials 005 \
  --context-sec 4.0 \
  --epochs 120 \
  --patience 16 \
  --batch-size 192 \
  --hidden-dim 192 \
  --layers 6 \
  --dropout 0.1 \
  --lr 0.002 \
  --weight-decay 0.0001 \
  --device cuda
```

Supported model types:

- `tcn`
- `mlp`
- `gru`
- `transformer`
- `convtransformer`

## Ensemble

```bash
python PIR_gpu/data/ensemble_prediction_csvs.py \
  --inputs \
    PIR_gpu/model_output/sequence_tcn_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_tcn6s_gpu/test_predictions.csv \
    PIR_gpu/model_output/sequence_gru6s_gpu/test_predictions.csv \
  --names tcn2s tcn4s tcn6s gru6s \
  --output-dir PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6
```

## Model Comparison Report

```bash
python PIR_gpu/data/summarize_model_comparison.py \
  --output-dir PIR_gpu/model_output/model_comparison \
  --model Ridge=PIR_gpu/model_output/baseline_ridge_v3 \
  --model TCN_4s=PIR_gpu/model_output/sequence_tcn4s_gpu \
  --model Ensemble_TCN2_TCN4_TCN6_GRU6=PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6
```

## 3D Viewer

Single model:

```bash
python PIR_gpu/data/create_prediction_viewer.py \
  --input PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
  --output-dir PIR_gpu/model_output/sequence_tcn4s_gpu/prediction_viewer
```

Multi-model comparison viewer:

```bash
python PIR_gpu/data/create_prediction_viewer.py \
  --output-dir PIR_gpu/model_output/comparison_viewer \
  --model Ridge=PIR_gpu/model_output/baseline_ridge_v3/test_predictions.csv \
  --model TCN_4s=PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
  --model Ensemble_TCN2_TCN4_TCN6_GRU6=PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6/test_predictions.csv
```

Serve the viewer:

```bash
cd PIR_gpu/model_output/comparison_viewer
python -m http.server 8766
```

Then open:

```text
http://127.0.0.1:8766/index.html
```

The viewer shows global and pelvis-relative skeletons, supports model switching, and exports the current view as WebM.

## Notes

- Raw/trimmed data is tracked.
- Large generated artifacts are ignored and should be regenerated locally.
- The current evaluation is a single held-out-trial experiment. For publication-grade claims, add trial-wise cross-validation and additional recording sessions.
