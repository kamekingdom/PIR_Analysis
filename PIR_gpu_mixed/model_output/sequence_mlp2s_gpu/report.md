# PIR時系列モデル

## 概要

- モデル: mlp
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 377.1 mm
- テストMAE: 225.1 mm
- テスト平均関節誤差: 514.2 mm
- テストR2: 0.890

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 629.9 |
| head | 637.0 |
| l_foot | 647.9 |
| l_hand | 693.4 |
| l_larm | 665.1 |
| l_shank | 661.6 |
| l_thigh | 643.2 |
| l_toes | 681.4 |
| l_uarm | 653.6 |
| neck | 639.3 |
| pelvis | 629.9 |
| r_foot | 647.6 |
| r_hand | 678.9 |
| r_larm | 648.1 |
| r_shank | 655.0 |
| r_thigh | 634.3 |
| r_toes | 683.8 |
| r_uarm | 636.2 |
| thorax | 639.3 |