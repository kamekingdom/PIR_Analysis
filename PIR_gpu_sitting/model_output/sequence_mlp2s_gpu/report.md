# PIR時系列モデル

## 概要

- モデル: mlp
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 369.3 mm
- テストMAE: 236.6 mm
- テスト平均関節誤差: 533.2 mm
- テストR2: 0.867

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 617.1 |
| head | 625.8 |
| l_foot | 632.8 |
| l_hand | 680.7 |
| l_larm | 652.3 |
| l_shank | 650.5 |
| l_thigh | 629.4 |
| l_toes | 666.2 |
| l_uarm | 640.5 |
| neck | 628.5 |
| pelvis | 617.1 |
| r_foot | 634.8 |
| r_hand | 662.2 |
| r_larm | 632.9 |
| r_shank | 638.7 |
| r_thigh | 620.7 |
| r_toes | 667.0 |
| r_uarm | 622.4 |
| thorax | 628.5 |