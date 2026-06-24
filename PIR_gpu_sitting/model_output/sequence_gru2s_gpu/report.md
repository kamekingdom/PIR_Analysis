# PIR時系列モデル

## 概要

- モデル: gru
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 373.1 mm
- テストMAE: 230.8 mm
- テスト平均関節誤差: 526.3 mm
- テストR2: 0.865

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 626.0 |
| head | 624.6 |
| l_foot | 641.8 |
| l_hand | 684.0 |
| l_larm | 655.9 |
| l_shank | 653.8 |
| l_thigh | 637.7 |
| l_toes | 673.0 |
| l_uarm | 645.7 |
| neck | 632.1 |
| pelvis | 626.0 |
| r_foot | 639.0 |
| r_hand | 668.2 |
| r_larm | 647.6 |
| r_shank | 645.4 |
| r_thigh | 634.4 |
| r_toes | 671.8 |
| r_uarm | 634.2 |
| thorax | 632.6 |