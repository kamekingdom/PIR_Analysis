# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 260.5 mm
- テストMAE: 153.4 mm
- テスト平均関節誤差: 347.9 mm
- テストR2: 0.957

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 424.7 |
| head | 434.5 |
| l_foot | 452.6 |
| l_hand | 505.3 |
| l_larm | 467.4 |
| l_shank | 454.2 |
| l_thigh | 439.3 |
| l_toes | 487.2 |
| l_uarm | 453.6 |
| neck | 435.5 |
| pelvis | 425.8 |
| r_foot | 461.5 |
| r_hand | 460.3 |
| r_larm | 428.3 |
| r_shank | 452.2 |
| r_thigh | 425.9 |
| r_toes | 493.9 |
| r_uarm | 423.3 |
| thorax | 436.2 |