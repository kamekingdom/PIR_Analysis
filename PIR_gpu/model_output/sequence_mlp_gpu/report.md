# PIR時系列モデル

## 概要

- モデル: mlp
- 実行デバイス: cuda
- 学習trial: 002, 003, 004
- 検証フレーム数: 2915
- テストtrial: 005
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 293.8 mm
- テストMAE: 169.7 mm
- テスト平均関節誤差: 389.0 mm
- テストR2: 0.945

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 482.1 |
| head | 494.0 |
| l_foot | 503.9 |
| l_hand | 550.5 |
| l_larm | 511.7 |
| l_shank | 514.4 |
| l_thigh | 494.4 |
| l_toes | 546.6 |
| l_uarm | 502.4 |
| neck | 495.5 |
| pelvis | 482.1 |
| r_foot | 514.0 |
| r_hand | 530.6 |
| r_larm | 494.3 |
| r_shank | 517.4 |
| r_thigh | 487.6 |
| r_toes | 556.0 |
| r_uarm | 487.8 |
| thorax | 495.5 |