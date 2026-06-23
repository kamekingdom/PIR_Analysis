# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 6.0秒 (151フレーム)
- テストRMSE: 269.6 mm
- テストMAE: 156.5 mm
- テスト平均関節誤差: 355.3 mm
- テストR2: 0.953

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 444.8 |
| head | 449.4 |
| l_foot | 474.9 |
| l_hand | 505.7 |
| l_larm | 476.8 |
| l_shank | 470.9 |
| l_thigh | 456.0 |
| l_toes | 501.0 |
| l_uarm | 463.2 |
| neck | 451.0 |
| pelvis | 444.2 |
| r_foot | 479.2 |
| r_hand | 478.4 |
| r_larm | 452.0 |
| r_shank | 468.1 |
| r_thigh | 445.7 |
| r_toes | 505.3 |
| r_uarm | 446.8 |
| thorax | 451.1 |