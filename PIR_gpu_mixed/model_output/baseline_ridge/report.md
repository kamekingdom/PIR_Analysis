# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- テストtrial: rw_005, sit_008
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 1000.0
- テストRMSE: 1126.6 mm
- テストMAE: 820.9 mm
- テスト平均関節誤差: 1812.7 mm
- テストR2: 0.015
- 同期モード: timestamps
- PIR時刻ソース: timestamp_unix

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 1942.0 |
| head | 1939.4 |
| l_foot | 1958.9 |
| l_hand | 1973.8 |
| l_larm | 1965.4 |
| l_shank | 1954.8 |
| l_thigh | 1946.9 |
| l_toes | 1961.0 |
| l_uarm | 1955.3 |
| neck | 1940.9 |
| pelvis | 1942.0 |
| r_foot | 1959.9 |
| r_hand | 1951.2 |
| r_larm | 1946.5 |
| r_shank | 1953.4 |
| r_thigh | 1939.7 |
| r_toes | 1961.3 |
| r_uarm | 1941.8 |
| thorax | 1940.9 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標