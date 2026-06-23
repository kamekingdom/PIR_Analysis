# PIR時系列モデル

## 概要

- モデル: tcn
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 267.8 mm
- テストMAE: 156.2 mm
- テスト平均関節誤差: 357.3 mm
- テストR2: 0.954

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 440.0 |
| head | 446.7 |
| l_foot | 460.1 |
| l_hand | 497.5 |
| l_larm | 458.7 |
| l_shank | 462.9 |
| l_thigh | 448.8 |
| l_toes | 494.7 |
| l_uarm | 457.7 |
| neck | 445.7 |
| pelvis | 439.5 |
| r_foot | 471.5 |
| r_hand | 501.4 |
| r_larm | 467.2 |
| r_shank | 464.2 |
| r_thigh | 444.6 |
| r_toes | 503.2 |
| r_uarm | 454.1 |
| thorax | 447.2 |