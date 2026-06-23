# PIR Sequence Model

## Summary

- Model: convtransformer
- Device: cuda
- Train trials: 002, 003, 004
- Validation trials/frames: 2915
- Test trials: 005
- Context: 4.0 sec (101 frames)
- Test RMSE: 355.0 mm
- Test MAE: 209.5 mm
- Test mean joint error: 482.9 mm
- Test R2: 0.919

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 599.2 |
| head | 605.3 |
| l_foot | 611.8 |
| l_hand | 646.7 |
| l_larm | 619.4 |
| l_shank | 623.2 |
| l_thigh | 597.6 |
| l_toes | 648.5 |
| l_uarm | 614.2 |
| neck | 602.8 |
| pelvis | 589.6 |
| r_foot | 614.4 |
| r_hand | 634.4 |
| r_larm | 606.4 |
| r_shank | 617.7 |
| r_thigh | 595.9 |
| r_toes | 656.8 |
| r_uarm | 597.9 |
| thorax | 596.6 |