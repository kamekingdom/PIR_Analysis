# PIR時系列モデル

## 概要

- モデル: gru
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 6.0秒 (151フレーム)
- テストRMSE: 382.3 mm
- テストMAE: 220.7 mm
- テスト平均関節誤差: 508.3 mm
- テストR2: 0.857

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 648.2 |
| head | 647.3 |
| l_foot | 659.8 |
| l_hand | 678.4 |
| l_larm | 661.7 |
| l_shank | 659.2 |
| l_thigh | 651.5 |
| l_toes | 678.3 |
| l_uarm | 651.9 |
| neck | 648.7 |
| pelvis | 647.0 |
| r_foot | 664.3 |
| r_hand | 690.6 |
| r_larm | 673.0 |
| r_shank | 663.9 |
| r_thigh | 658.2 |
| r_toes | 686.8 |
| r_uarm | 659.9 |
| thorax | 649.2 |