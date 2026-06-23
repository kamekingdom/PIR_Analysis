# PIR Sequence Model

## Summary

- Model: transformer
- Device: cuda
- Train trials: 002, 003, 004
- Validation trials/frames: 2915
- Test trials: 005
- Context: 4.0 sec (101 frames)
- Test RMSE: 338.7 mm
- Test MAE: 199.7 mm
- Test mean joint error: 459.0 mm
- Test R2: 0.927

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 565.1 |
| head | 572.6 |
| l_foot | 594.1 |
| l_hand | 630.7 |
| l_larm | 594.2 |
| l_shank | 598.1 |
| l_thigh | 576.1 |
| l_toes | 628.0 |
| l_uarm | 584.8 |
| neck | 568.8 |
| pelvis | 559.3 |
| r_foot | 584.3 |
| r_hand | 602.1 |
| r_larm | 563.7 |
| r_shank | 590.9 |
| r_thigh | 566.9 |
| r_toes | 624.4 |
| r_uarm | 557.7 |
| thorax | 575.1 |