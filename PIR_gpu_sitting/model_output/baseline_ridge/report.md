# 20260623 PIRからTheia3D Global Bone座標を推定するモデル

## 概要

- 学習trial: 003, 004, 005, 006, 007
- テストtrial: 008
- 特徴量数: 375
- 予測対象数: 57個のglobal座標値
- Ridge alpha: 1000.0
- テストRMSE: 996.1 mm
- テストMAE: 735.1 mm
- テスト平均関節誤差: 1591.5 mm
- テストR2: 0.030
- 同期モード: timestamps
- PIR時刻ソース: timestamp_unix

予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 1710.9 |
| head | 1715.0 |
| l_foot | 1723.6 |
| l_hand | 1753.2 |
| l_larm | 1735.8 |
| l_shank | 1727.1 |
| l_thigh | 1714.9 |
| l_toes | 1734.3 |
| l_uarm | 1724.7 |
| neck | 1717.1 |
| pelvis | 1710.9 |
| r_foot | 1730.4 |
| r_hand | 1737.3 |
| r_larm | 1725.8 |
| r_shank | 1730.8 |
| r_thigh | 1712.8 |
| r_toes | 1739.9 |
| r_uarm | 1718.9 |
| thorax | 1717.1 |

## 出力ファイル

- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ
- `test_predictions.csv`: 未使用テストtrialの予測結果
- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差
- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標
- `metrics.json`: 全評価指標