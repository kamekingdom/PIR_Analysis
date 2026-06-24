# PIR時系列モデル

## 概要

- モデル: convtransformer
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 370.2 mm
- テストMAE: 233.7 mm
- テスト平均関節誤差: 531.9 mm
- テストR2: 0.867

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 622.3 |
| head | 625.8 |
| l_foot | 638.0 |
| l_hand | 685.2 |
| l_larm | 658.8 |
| l_shank | 651.3 |
| l_thigh | 634.6 |
| l_toes | 665.8 |
| l_uarm | 645.6 |
| neck | 628.9 |
| pelvis | 620.6 |
| r_foot | 632.6 |
| r_hand | 660.2 |
| r_larm | 633.3 |
| r_shank | 639.7 |
| r_thigh | 623.2 |
| r_toes | 659.5 |
| r_uarm | 622.4 |
| thorax | 629.1 |