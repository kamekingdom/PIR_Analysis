# PIR時系列モデル

## 概要

- モデル: convtransformer
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 326.7 mm
- テストMAE: 186.9 mm
- テスト平均関節誤差: 427.7 mm
- テストR2: 0.917

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 543.9 |
| head | 549.7 |
| l_foot | 566.1 |
| l_hand | 605.1 |
| l_larm | 576.5 |
| l_shank | 572.9 |
| l_thigh | 556.7 |
| l_toes | 598.9 |
| l_uarm | 565.4 |
| neck | 551.9 |
| pelvis | 543.9 |
| r_foot | 564.7 |
| r_hand | 585.4 |
| r_larm | 555.7 |
| r_shank | 567.1 |
| r_thigh | 547.1 |
| r_toes | 597.5 |
| r_uarm | 545.9 |
| thorax | 551.9 |