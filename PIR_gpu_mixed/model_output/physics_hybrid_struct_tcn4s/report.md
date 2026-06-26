# PIR時系列モデル

## 概要

- モデル: hybrid_struct_tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 351.2 mm
- テストMAE: 206.8 mm
- テスト平均関節誤差: 476.5 mm
- テストR2: 0.904

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 588.3 |
| head | 592.6 |
| l_foot | 609.1 |
| l_hand | 633.2 |
| l_larm | 610.2 |
| l_shank | 608.8 |
| l_thigh | 593.6 |
| l_toes | 628.5 |
| l_uarm | 599.9 |
| neck | 593.9 |
| pelvis | 588.4 |
| r_foot | 612.2 |
| r_hand | 640.4 |
| r_larm | 615.9 |
| r_shank | 610.6 |
| r_thigh | 596.4 |
| r_toes | 634.8 |
| r_uarm | 602.8 |
| thorax | 593.4 |