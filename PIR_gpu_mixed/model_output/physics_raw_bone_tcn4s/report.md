# PIR時系列モデル

## 概要

- モデル: raw_bone_tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 331.7 mm
- テストMAE: 190.9 mm
- テスト平均関節誤差: 434.3 mm
- テストR2: 0.915

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 554.7 |
| head | 561.5 |
| l_foot | 576.6 |
| l_hand | 611.5 |
| l_larm | 588.0 |
| l_shank | 575.9 |
| l_thigh | 563.3 |
| l_toes | 598.2 |
| l_uarm | 575.2 |
| neck | 560.9 |
| pelvis | 557.2 |
| r_foot | 577.6 |
| r_hand | 589.7 |
| r_larm | 569.1 |
| r_shank | 570.7 |
| r_thigh | 558.8 |
| r_toes | 599.6 |
| r_uarm | 562.2 |
| thorax | 561.0 |