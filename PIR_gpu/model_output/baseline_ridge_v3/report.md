# 20260623 PIR to Global Bone Model

## Summary

- Train trials: 002, 003, 004
- Test trials: 005
- Feature count: 375
- Target count: 57 global coordinate values
- Ridge alpha: 100.0
- Test RMSE: 1184.1 mm
- Test MAE: 829.7 mm
- Test mean joint error: 1889.9 mm
- Test R2: 0.103
- Sync mode: timestamps
- PIR time source: timestamp_unix

Targets are Theia global joint coordinates. `worldbody` is not predicted.

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 2043.5 |
| head | 2036.6 |
| l_foot | 2064.0 |
| l_hand | 2070.9 |
| l_larm | 2065.8 |
| l_shank | 2059.3 |
| l_thigh | 2051.2 |
| l_toes | 2066.9 |
| l_uarm | 2056.9 |
| neck | 2038.0 |
| pelvis | 2043.5 |
| r_foot | 2060.8 |
| r_hand | 2042.9 |
| r_larm | 2038.3 |
| r_shank | 2052.7 |
| r_thigh | 2038.8 |
| r_toes | 2063.0 |
| r_uarm | 2034.9 |
| thorax | 2038.0 |

## Files

- `pir_to_global_bones_ridge_model.npz`: model parameters and metadata
- `test_predictions.csv`: held-out trial predictions
- `test_frame_errors.csv`: held-out frame mean joint errors
- `aligned_dataset.npz`: 25 Hz aligned features and targets
- `metrics.json`: full metrics