# PIR時系列モデル

## 概要

- モデル: mlp
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 409.0 mm
- テストMAE: 233.0 mm
- テスト平均関節誤差: 532.0 mm
- テストR2: 0.893

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 683.0 |
| head | 694.8 |
| l_foot | 706.1 |
| l_hand | 751.3 |
| l_larm | 714.5 |
| l_shank | 722.5 |
| l_thigh | 698.6 |
| l_toes | 752.2 |
| l_uarm | 706.4 |
| neck | 696.6 |
| pelvis | 683.0 |
| r_foot | 704.7 |
| r_hand | 723.8 |
| r_larm | 686.0 |
| r_shank | 715.6 |
| r_thigh | 685.4 |
| r_toes | 749.4 |
| r_uarm | 681.2 |
| thorax | 696.6 |