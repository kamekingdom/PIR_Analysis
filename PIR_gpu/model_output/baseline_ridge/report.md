# 20260623 PIR to Global Bone Model

## Summary

- Train trials: 002, 003, 004
- Test trials: 005
- Feature count: 375
- Target count: 57 global coordinate values
- Ridge alpha: 100.0
- Test RMSE: 1183.6 mm
- Test MAE: 829.4 mm
- Test R2: 0.104

Targets are Theia global joint coordinates. `worldbody` is not predicted.

## Joint RMSE

| joint | RMSE [mm] |
|---|---:|
| abdomen | 2042.7 |
| head | 2035.8 |
| l_foot | 2063.0 |
| l_hand | 2070.3 |
| l_larm | 2065.0 |
| l_shank | 2058.6 |
| l_thigh | 2050.4 |
| l_toes | 2066.0 |
| l_uarm | 2056.1 |
| neck | 2037.3 |
| pelvis | 2042.7 |
| r_foot | 2060.1 |
| r_hand | 2042.1 |
| r_larm | 2037.5 |
| r_shank | 2052.0 |
| r_thigh | 2038.0 |
| r_toes | 2062.3 |
| r_uarm | 2034.2 |
| thorax | 2037.3 |

## Files

- `pir_to_global_bones_ridge_model.npz`: model parameters and metadata
- `test_predictions.csv`: held-out trial predictions
- `metrics.json`: full metrics