# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 003, 004, 005, 006, 007
- 検証フレーム数: 5509
- テストtrial: 008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 340.0 mm
- テストMAE: 202.4 mm
- テスト平均関節誤差: 462.1 mm
- テストR2: 0.888

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 573.6 |
| head | 571.1 |
| l_foot | 590.2 |
| l_hand | 611.9 |
| l_larm | 592.3 |
| l_shank | 586.7 |
| l_thigh | 578.3 |
| l_toes | 609.4 |
| l_uarm | 581.4 |
| neck | 573.5 |
| pelvis | 573.5 |
| r_foot | 589.6 |
| r_hand | 619.7 |
| r_larm | 599.9 |
| r_shank | 586.5 |
| r_thigh | 581.5 |
| r_toes | 610.5 |
| r_uarm | 583.4 |
| thorax | 573.4 |