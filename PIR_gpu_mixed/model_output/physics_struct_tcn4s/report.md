# PIR時系列モデル

## 概要

- モデル: struct_tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 352.3 mm
- テストMAE: 198.4 mm
- テスト平均関節誤差: 456.6 mm
- テストR2: 0.904

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 588.4 |
| head | 593.7 |
| l_foot | 601.9 |
| l_hand | 637.2 |
| l_larm | 607.4 |
| l_shank | 614.1 |
| l_thigh | 595.7 |
| l_toes | 636.0 |
| l_uarm | 599.0 |
| neck | 595.8 |
| pelvis | 588.3 |
| r_foot | 614.1 |
| r_hand | 643.2 |
| r_larm | 612.1 |
| r_shank | 621.3 |
| r_thigh | 597.6 |
| r_toes | 649.8 |
| r_uarm | 598.5 |
| thorax | 595.9 |