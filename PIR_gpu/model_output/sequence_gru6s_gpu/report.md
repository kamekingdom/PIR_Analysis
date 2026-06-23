# PIR Sequence Model

## Summary

- Model: gru
- Device: cuda
- Train trials: 002, 003, 004
- Validation trials/frames: 2915
- Test trials: 005
- Context: 6.0 sec (151 frames)
- Test RMSE: 266.6 mm
- Test MAE: 158.6 mm
- Test mean joint error: 365.2 mm
- Test R2: 0.954

## Joint RMSE

| joint | RMSE [mm] |
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