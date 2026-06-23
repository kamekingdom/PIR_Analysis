# PIR解析

PIRセンサ時系列からTheia3Dの3D骨格座標を推定する解析プロジェクトです。

このリポジトリには、trim済みのPIR/Theia3Dデータ、前処理、Ridgeベースライン、GPU対応の時系列深層学習モデル、モデル比較、アンサンブル、ブラウザ上の3D可視化ビューアが含まれています。

## データ構成

- `PIR_gpu/data/<trial>/pir/pir_data_192_168_11_x.csv`
  - PIRモジュールごとのCSVです。1モジュール5chです。
- `PIR_gpu/data/<trial>/skelton/Theia_Sub0.csv`
  - Theia3Dのglobal joint座標です。25Hzです。
- `PIR_gpu/data/<trial>/skelton/<trial>.tsv`
  - Theia側のメタデータです。周波数やタイムスタンプを含みます。

元データのフォルダ名に合わせて、`skeleton` ではなく `skelton` という綴りをそのまま使っています。

## 主な結果

評価分割:

- 学習: trial `002`, `003`, `004`
- テスト: trial `005`

最良結果は、複数モデルの予測平均アンサンブルでした。

| モデル | RMSE [mm] | MAE [mm] | 平均関節誤差 [mm] | R2 |
|---|---:|---:|---:|---:|
| Ensemble_TCN2_TCN4_TCN6_GRU6 | 240.1 | 138.0 | 312.4 | 0.963 |
| TCN_4s | 260.5 | 153.4 | 347.9 | 0.957 |
| Ridgeベースライン | 1184.1 | 829.7 | 1889.9 | 0.103 |

全モデルの比較表:

- `PIR_gpu/model_output/model_comparison/experiment_report.md`
- `PIR_gpu/model_output/model_comparison/model_comparison.csv`

## セットアップ

```bash
python -m pip install numpy torch
```

CUDA実行はRTX 4070 Ti SUPERで確認しました。この環境では、PyTorchからGPUを使うには制限付きsandboxの外で実行する必要がありました。

## ベースラインと25Hz整列データの再生成

```bash
python PIR_gpu/data/train_pir_to_global_bones.py \
  --trimmed-dir PIR_gpu/data \
  --output-dir PIR_gpu/model_output/baseline_ridge_v3 \
  --pir-time-source auto \
  --sync-mode timestamps
```

このコマンドで、Theiaの25Hzフレーム時刻にPIR特徴量を補間した整列済みデータが作成されます。

```text
PIR_gpu/model_output/baseline_ridge_v3/aligned_dataset.npz
```

`aligned_dataset.npz`、`model.pt`、予測CSV、viewer用JSONなどの大きい生成物はGit管理から外しています。必要に応じて各スクリプトで再生成してください。

## 時系列モデルの学習

最良単体モデルの例:

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

対応モデル:

- `tcn`
- `mlp`
- `gru`
- `transformer`
- `convtransformer`

## アンサンブル

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

## モデル比較レポート

```bash
python PIR_gpu/data/summarize_model_comparison.py \
  --output-dir PIR_gpu/model_output/model_comparison \
  --model Ridge=PIR_gpu/model_output/baseline_ridge_v3 \
  --model TCN_4s=PIR_gpu/model_output/sequence_tcn4s_gpu \
  --model Ensemble_TCN2_TCN4_TCN6_GRU6=PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6
```

## 3Dビューア

単一モデル:

```bash
python PIR_gpu/data/create_prediction_viewer.py \
  --input PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
  --output-dir PIR_gpu/model_output/sequence_tcn4s_gpu/prediction_viewer
```

複数モデル比較ビューア:

```bash
python PIR_gpu/data/create_prediction_viewer.py \
  --output-dir PIR_gpu/model_output/comparison_viewer \
  --model Ridge=PIR_gpu/model_output/baseline_ridge_v3/test_predictions.csv \
  --model TCN_4s=PIR_gpu/model_output/sequence_tcn4s_gpu/test_predictions.csv \
  --model Ensemble_TCN2_TCN4_TCN6_GRU6=PIR_gpu/model_output/ensemble_tcn2_tcn4_tcn6_gru6/test_predictions.csv
```

ビューアの起動:

```bash
cd PIR_gpu/model_output/comparison_viewer
python -m http.server 8766
```

ブラウザで以下を開きます。

```text
http://127.0.0.1:8766/index.html
```

ビューアではglobal座標とpelvis基準の相対座標を並べて表示できます。モデル切り替えと、現在の画角でのWebM動画書き出しにも対応しています。

## 注意

- trim済みのraw dataはGitで管理しています。
- 大きい生成物はGit管理外です。ローカルで再生成してください。
- 現在の評価は単一の未使用テストtrialです。論文レベルの主張には、trial単位の交差検証と別日・別条件の追加収録が必要です。
