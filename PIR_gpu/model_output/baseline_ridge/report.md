# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: 002, 003, 004
- テストtrial: 005
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 100.0
- テストRMSE: 1183.6 mm
- テストMAE: 829.4 mm
- テスト平均関節誤差: 未算出
- テストR2: 0.104
- 同期モード: 未記録
- PIR時刻ソース: 未記録

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 2042.7 |
| head | 2035.8 |
| l_foot | 2063.0 |
| l_hand | 2070.3 |
| l_larm | 2065.0 |
| l_shank | 2058.6 |
| l_thigh | 2050.4 |
| l_toes | 2066.0 |
| l_uarm | 2056.1 |
| neck | 2037.3 |
| pelvis | 2042.7 |
| r_foot | 2060.1 |
| r_hand | 2042.1 |
| r_larm | 2037.5 |
| r_shank | 2052.0 |
| r_thigh | 2038.0 |
| r_toes | 2062.3 |
| r_uarm | 2034.2 |
| thorax | 2037.3 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標