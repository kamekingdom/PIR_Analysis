# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 321.8 mm
- テストMAE: 183.5 mm
- テスト平均関節誤差: 418.4 mm
- テストR2: 0.920

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 538.5 |
| head | 540.9 |
| l_foot | 565.3 |
| l_hand | 591.7 |
| l_larm | 569.5 |
| l_shank | 559.6 |
| l_thigh | 547.7 |
| l_toes | 584.3 |
| l_uarm | 556.9 |
| neck | 542.0 |
| pelvis | 538.5 |
| r_foot | 560.2 |
| r_hand | 578.8 |
| r_larm | 555.2 |
| r_shank | 550.2 |
| r_thigh | 540.6 |
| r_toes | 580.8 |
| r_uarm | 543.7 |
| thorax | 542.0 |