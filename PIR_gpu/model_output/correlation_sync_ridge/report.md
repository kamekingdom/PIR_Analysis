# 20260623 PIR to Global Bone Model

## Summary

- Train trials: 002, 003, 004
- Test trials: 005
- Feature count: 375
- Target count: 57 global coordinate values
- Ridge alpha: 10000.0
- Test RMSE: 1244.5 mm
- Test MAE: 905.5 mm
- Test mean joint error: 2037.8 mm
- Test R2: 0.010
- Sync mode: correlation
- PIR time source: timestamp_unix

Targets are Theia global joint coordinates. `worldbody` is not predicted.

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 2149.9 |
| head | 2141.2 |
| l_foot | 2169.6 |
| l_hand | 2167.4 |
| l_larm | 2166.3 |
| l_shank | 2159.6 |
| l_thigh | 2153.5 |
| l_toes | 2166.3 |
| l_uarm | 2158.1 |
| neck | 2142.4 |
| pelvis | 2149.9 |
| r_foot | 2169.1 |
| r_hand | 2151.1 |
| r_larm | 2150.3 |
| r_shank | 2158.0 |
| r_thigh | 2146.6 |
| r_toes | 2166.3 |
| r_uarm | 2146.0 |
| thorax | 2142.4 |

## Files

- `pir_to_global_bones_ridge_model.npz`: model parameters and metadata
- `test_predictions.csv`: held-out trial predictions
- `test_frame_errors.csv`: held-out frame mean joint errors
- `aligned_dataset.npz`: 25 Hz aligned features and targets
- `metrics.json`: full metrics