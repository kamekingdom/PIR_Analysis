# PIR時系列モデル

## 概要

- モデル: convtransformer
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 355.0 mm
- テストMAE: 209.5 mm
- テスト平均関節誤差: 482.9 mm
- テストR2: 0.919

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 599.2 |
| head | 605.3 |
| l_foot | 611.8 |
| l_hand | 646.7 |
| l_larm | 619.4 |
| l_shank | 623.2 |
| l_thigh | 597.6 |
| l_toes | 648.5 |
| l_uarm | 614.2 |
| neck | 602.8 |
| pelvis | 589.6 |
| r_foot | 614.4 |
| r_hand | 634.4 |
| r_larm | 606.4 |
| r_shank | 617.7 |
| r_thigh | 595.9 |
| r_toes | 656.8 |
| r_uarm | 597.9 |
| thorax | 596.6 |