# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: 002, 003, 004
- テストtrial: 005
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 100.0
- テストRMSE: 1184.1 mm
- テストMAE: 829.7 mm
- テスト平均関節誤差: 1889.9 mm
- テストR2: 0.103
- 同期モード: timestamps
- PIR時刻ソース: timestamp_unix

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 2043.5 |
| head | 2036.6 |
| l_foot | 2064.0 |
| l_hand | 2070.9 |
| l_larm | 2065.8 |
| l_shank | 2059.3 |
| l_thigh | 2051.2 |
| l_toes | 2066.9 |
| l_uarm | 2056.9 |
| neck | 2038.0 |
| pelvis | 2043.5 |
| r_foot | 2060.8 |
| r_hand | 2042.9 |
| r_larm | 2038.3 |
| r_shank | 2052.7 |
| r_thigh | 2038.8 |
| r_toes | 2063.0 |
| r_uarm | 2034.9 |
| thorax | 2038.0 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標