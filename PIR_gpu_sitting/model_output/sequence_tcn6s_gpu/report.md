# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 6.0秒 (151フレーム)
- テストRMSE: 409.1 mm
- テストMAE: 240.9 mm
- テスト平均関節誤差: 553.7 mm
- テストR2: 0.837

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 692.8 |
| head | 694.2 |
| l_foot | 707.3 |
| l_hand | 743.9 |
| l_larm | 725.7 |
| l_shank | 711.2 |
| l_thigh | 703.2 |
| l_toes | 724.4 |
| l_uarm | 712.7 |
| neck | 696.9 |
| pelvis | 692.2 |
| r_foot | 707.1 |
| r_hand | 727.5 |
| r_larm | 707.3 |
| r_shank | 700.8 |
| r_thigh | 695.1 |
| r_toes | 725.0 |
| r_uarm | 696.1 |
| thorax | 696.5 |