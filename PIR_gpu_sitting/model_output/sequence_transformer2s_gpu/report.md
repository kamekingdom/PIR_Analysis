# PIR時系列モデル

## 概要

- モデル: transformer
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 375.8 mm
- テストMAE: 241.1 mm
- テスト平均関節誤差: 544.4 mm
- テストR2: 0.863

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 622.5 |
| head | 631.7 |
| l_foot | 637.0 |
| l_hand | 702.7 |
| l_larm | 667.9 |
| l_shank | 659.4 |
| l_thigh | 642.7 |
| l_toes | 673.5 |
| l_uarm | 655.8 |
| neck | 637.2 |
| pelvis | 625.6 |
| r_foot | 636.6 |
| r_hand | 681.9 |
| r_larm | 644.0 |
| r_shank | 652.5 |
| r_thigh | 632.4 |
| r_toes | 683.5 |
| r_uarm | 635.4 |
| thorax | 639.6 |