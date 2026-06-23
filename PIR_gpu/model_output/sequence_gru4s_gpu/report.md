# PIR Sequence Model

## Summary

- Model: gru
- Device: cuda
- Train trials: 002, 003, 004
- Validation trials/frames: 2915
- Test trials: 005
- Context: 4.0 sec (101 frames)
- Test RMSE: 289.0 mm
- Test MAE: 177.2 mm
- Test mean joint error: 409.7 mm
- Test R2: 0.946

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 464.3 |
| head | 480.4 |
| l_foot | 484.1 |
| l_hand | 555.5 |
| l_larm | 508.3 |
| l_shank | 505.5 |
| l_thigh | 480.1 |
| l_toes | 526.2 |
| l_uarm | 489.9 |
| neck | 476.5 |
| pelvis | 454.3 |
| r_foot | 490.3 |
| r_hand | 548.6 |
| r_larm | 510.1 |
| r_shank | 505.1 |
| r_thigh | 481.5 |
| r_toes | 556.5 |
| r_uarm | 497.5 |
| thorax | 480.6 |