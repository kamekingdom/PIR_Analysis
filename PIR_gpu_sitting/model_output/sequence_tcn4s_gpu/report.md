# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 4.0秒 (101フレーム)
- テストRMSE: 401.3 mm
- テストMAE: 240.3 mm
- テスト平均関節誤差: 551.2 mm
- テストR2: 0.843

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 684.2 |
| head | 679.3 |
| l_foot | 688.6 |
| l_hand | 715.0 |
| l_larm | 701.7 |
| l_shank | 689.1 |
| l_thigh | 689.3 |
| l_toes | 702.7 |
| l_uarm | 691.2 |
| neck | 681.6 |
| pelvis | 684.1 |
| r_foot | 700.1 |
| r_hand | 719.3 |
| r_larm | 705.2 |
| r_shank | 689.4 |
| r_thigh | 691.1 |
| r_toes | 719.3 |
| r_uarm | 690.8 |
| thorax | 681.6 |