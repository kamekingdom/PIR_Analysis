# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: 002, 003, 004
- テストtrial: 005
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 100000.0
- テストRMSE: 1251.5 mm
- テストMAE: 910.9 mm
- テスト平均関節誤差: 2052.6 mm
- テストR2: -0.001
- 同期モード: correlation
- PIR時刻ソース: time_sec

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 2162.3 |
| head | 2154.3 |
| l_foot | 2176.7 |
| l_hand | 2164.3 |
| l_larm | 2163.5 |
| l_shank | 2165.0 |
| l_thigh | 2159.2 |
| l_toes | 2172.1 |
| l_uarm | 2158.9 |
| neck | 2155.1 |
| pelvis | 2162.3 |
| r_foot | 2185.8 |
| r_hand | 2177.2 |
| r_larm | 2177.2 |
| r_shank | 2176.2 |
| r_thigh | 2165.4 |
| r_toes | 2184.1 |
| r_uarm | 2170.1 |
| thorax | 2155.1 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標