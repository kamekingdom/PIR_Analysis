# PIR時系列モデル

## 概要

- モデル: transformer
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 338.7 mm
- テストMAE: 199.7 mm
- テスト平均関節誤差: 459.0 mm
- テストR2: 0.927

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 565.1 |
| head | 572.6 |
| l_foot | 594.1 |
| l_hand | 630.7 |
| l_larm | 594.2 |
| l_shank | 598.1 |
| l_thigh | 576.1 |
| l_toes | 628.0 |
| l_uarm | 584.8 |
| neck | 568.8 |
| pelvis | 559.3 |
| r_foot | 584.3 |
| r_hand | 602.1 |
| r_larm | 563.7 |
| r_shank | 590.9 |
| r_thigh | 566.9 |
| r_toes | 624.4 |
| r_uarm | 557.7 |
| thorax | 575.1 |