# PIR時系列モデル

## 概要

- モデル: rootlocal_tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 338.2 mm
- テストMAE: 189.8 mm
- テスト平均関節誤差: 433.2 mm
- テストR2: 0.911

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 565.7 |
| head | 569.1 |
| l_foot | 592.3 |
| l_hand | 620.1 |
| l_larm | 597.2 |
| l_shank | 592.6 |
| l_thigh | 575.3 |
| l_toes | 614.8 |
| l_uarm | 584.8 |
| neck | 570.8 |
| pelvis | 563.9 |
| r_foot | 586.3 |
| r_hand | 608.7 |
| r_larm | 582.8 |
| r_shank | 582.0 |
| r_thigh | 568.5 |
| r_toes | 608.7 |
| r_uarm | 571.4 |
| thorax | 570.9 |