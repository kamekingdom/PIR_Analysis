# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 318.5 mm
- テストMAE: 182.4 mm
- テスト平均関節誤差: 415.1 mm
- テストR2: 0.921

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 531.0 |
| head | 535.8 |
| l_foot | 554.2 |
| l_hand | 590.0 |
| l_larm | 565.5 |
| l_shank | 553.8 |
| l_thigh | 540.6 |
| l_toes | 575.6 |
| l_uarm | 551.8 |
| neck | 536.9 |
| pelvis | 531.0 |
| r_foot | 555.2 |
| r_hand | 571.4 |
| r_larm | 548.4 |
| r_shank | 547.4 |
| r_thigh | 534.0 |
| r_toes | 577.0 |
| r_uarm | 538.5 |
| thorax | 536.9 |