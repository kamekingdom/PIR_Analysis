# PIR Sequence Model

## Summary

- Model: mlp
- Device: cuda
- Train trials: 002, 003, 004
- Validation trials/frames: 2915
- Test trials: 005
- Context: 2.0 sec (51 frames)
- Test RMSE: 293.8 mm
- Test MAE: 169.7 mm
- Test mean joint error: 389.0 mm
- Test R2: 0.945

## Joint RMSE

| joint | RMSE [mm] |
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