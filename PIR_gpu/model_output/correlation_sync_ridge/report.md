# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: 002, 003, 004
- テストtrial: 005
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 10000.0
- テストRMSE: 1244.5 mm
- テストMAE: 905.5 mm
- テスト平均関節誤差: 2037.8 mm
- テストR2: 0.010
- 同期モード: correlation
- PIR時刻ソース: timestamp_unix

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 2149.9 |
| head | 2141.2 |
| l_foot | 2169.6 |
| l_hand | 2167.4 |
| l_larm | 2166.3 |
| l_shank | 2159.6 |
| l_thigh | 2153.5 |
| l_toes | 2166.3 |
| l_uarm | 2158.1 |
| neck | 2142.4 |
| pelvis | 2149.9 |
| r_foot | 2169.1 |
| r_hand | 2151.1 |
| r_larm | 2150.3 |
| r_shank | 2158.0 |
| r_thigh | 2146.6 |
| r_toes | 2166.3 |
| r_uarm | 2146.0 |
| thorax | 2142.4 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標