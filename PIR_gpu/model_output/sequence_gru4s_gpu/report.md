# PIR時系列モデル

## 概要

- モデル: gru
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 289.0 mm
- テストMAE: 177.2 mm
- テスト平均関節誤差: 409.7 mm
- テストR2: 0.946

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 464.3 |
| head | 480.4 |
| l_foot | 484.1 |
| l_hand | 555.5 |
| l_larm | 508.3 |
| l_shank | 505.5 |
| l_thigh | 480.1 |
| l_toes | 526.2 |
| l_uarm | 489.9 |
| neck | 476.5 |
| pelvis | 454.3 |
| r_foot | 490.3 |
| r_hand | 548.6 |
| r_larm | 510.1 |
| r_shank | 505.1 |
| r_thigh | 481.5 |
| r_toes | 556.5 |
| r_uarm | 497.5 |
| thorax | 480.6 |