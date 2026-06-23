# PIR時系列モデル

## 概要

- モデル: gru
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 6.0秒 (151フレーム)
- テストRMSE: 266.6 mm
- テストMAE: 158.6 mm
- テスト平均関節誤差: 365.2 mm
- テストR2: 0.954

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 437.9 |
| head | 443.5 |
| l_foot | 466.5 |
| l_hand | 497.0 |
| l_larm | 468.8 |
| l_shank | 468.4 |
| l_thigh | 449.8 |
| l_toes | 500.7 |
| l_uarm | 460.2 |
| neck | 448.9 |
| pelvis | 446.8 |
| r_foot | 460.4 |
| r_hand | 479.6 |
| r_larm | 444.6 |
| r_shank | 465.6 |
| r_thigh | 439.7 |
| r_toes | 497.4 |
| r_uarm | 435.4 |
| thorax | 454.1 |