# PIR時系列モデル

## 概要

- モデル: gru
- 実行デバイス: cuda
- 学習trial: rw_002, rw_003, rw_004, sit_003, sit_004, sit_005, sit_006, sit_007
- 検証フレーム数: 8424
- テストtrial: rw_005, sit_008
- 文脈長: 2.0秒 (51フレーム)
- テストRMSE: 374.7 mm
- テストMAE: 223.2 mm
- テスト平均関節誤差: 515.3 mm
- テストR2: 0.891

## 関節別RMSE

| 関節 | RMSE [mm] |
|---|---:|
| abdomen | 630.9 |
| head | 633.1 |
| l_foot | 631.8 |
| l_hand | 686.9 |
| l_larm | 664.3 |
| l_shank | 649.3 |
| l_thigh | 644.6 |
| l_toes | 665.4 |
| l_uarm | 646.5 |
| neck | 640.3 |
| pelvis | 627.7 |
| r_foot | 648.7 |
| r_hand | 676.9 |
| r_larm | 655.9 |
| r_shank | 650.4 |
| r_thigh | 628.8 |
| r_toes | 675.9 |
| r_uarm | 633.2 |
| thorax | 637.4 |