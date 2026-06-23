# 20260623 PIR to Global Bone Model

## Summary

- Train trials: 002, 003, 004
- Test trials: 005
- Feature count: 375
- Target count: 57 global coordinate values
- Ridge alpha: 100000.0
- Test RMSE: 1251.5 mm
- Test MAE: 910.9 mm
- Test mean joint error: 2052.6 mm
- Test R2: -0.001
- Sync mode: correlation
- PIR time source: time_sec

Targets are Theia global joint coordinates. `worldbody` is not predicted.

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 2162.3 |
| head | 2154.3 |
| l_foot | 2176.7 |
| l_hand | 2164.3 |
| l_larm | 2163.5 |
| l_shank | 2165.0 |
| l_thigh | 2159.2 |
| l_toes | 2172.1 |
| l_uarm | 2158.9 |
| neck | 2155.1 |
| pelvis | 2162.3 |
| r_foot | 2185.8 |
| r_hand | 2177.2 |
| r_larm | 2177.2 |
| r_shank | 2176.2 |
| r_thigh | 2165.4 |
| r_toes | 2184.1 |
| r_uarm | 2170.1 |
| thorax | 2155.1 |

## Files

- `pir_to_global_bones_ridge_model.npz`: model parameters and metadata
- `test_predictions.csv`: held-out trial predictions
- `test_frame_errors.csv`: held-out frame mean joint errors
- `aligned_dataset.npz`: 25 Hz aligned features and targets
- `metrics.json`: full metrics