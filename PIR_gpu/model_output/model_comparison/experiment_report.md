# PIR to Theia3D Model Comparison

## Experimental Setup

- Input: 25 Hz aligned PIR features generated from five PIR modules.
- Target: 19 Theia3D global joints, XYZ coordinates, 57 regression targets.
- Split: trials 002, 003, and 004 for training; trial 005 for held-out testing.
- Metrics: coordinate RMSE/MAE over all XYZ values, frame mean joint error, 95th percentile frame mean joint error, and global R2.

## Summary

- Best model: Ensemble_TCN2_TCN4_TCN6_GRU6
- Best RMSE: 240.1 mm
- Best mean joint error: 312.4 mm
- RMSE reduction from Ridge baseline: 79.7%

## Results

| Rank | Model | Type | Context [s] | Test rows | RMSE [mm] | MAE [mm] | Mean joint error [mm] | P95 mean joint error [mm] | R2 |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Ensemble_TCN2_TCN4_TCN6_GRU6 | prediction_average_ensemble |  | 6613 | 240.1 | 138.0 | 312.4 | 873.0 | 0.963 |
| 2 | Ensemble_TCN2_TCN4_TCN6_GRU6_MLP2 | prediction_average_ensemble |  | 6613 | 240.8 | 138.1 | 313.4 | 879.4 | 0.963 |
| 3 | Ensemble_TCN4_TCN6_GRU6 | prediction_average_ensemble |  | 6613 | 244.7 | 142.4 | 322.7 | 866.3 | 0.961 |
| 4 | Ensemble_TCN4_GRU6 | prediction_average_ensemble |  | 6613 | 247.3 | 145.0 | 330.2 | 886.1 | 0.960 |
| 5 | TCN_4s | tcn | 4.0 | 6663 | 260.5 | 153.4 | 347.9 | 901.2 | 0.957 |
| 6 | GRU_6s | gru | 6.0 | 6613 | 266.6 | 158.6 | 365.2 | 916.3 | 0.954 |
| 7 | TCN_2s | tcn | 2.0 | 6713 | 267.8 | 156.2 | 357.3 | 971.1 | 0.954 |
| 8 | TCN_6s | tcn | 6.0 | 6613 | 269.6 | 156.5 | 355.3 | 934.9 | 0.953 |
| 9 | GRU_4s | gru | 4.0 | 6663 | 289.0 | 177.2 | 409.7 | 932.1 | 0.946 |
| 10 | MLP_2s | mlp | 2.0 | 6713 | 293.8 | 169.7 | 389.0 | 1054.8 | 0.945 |
| 11 | Transformer_4s | transformer | 4.0 | 6663 | 338.7 | 199.7 | 459.0 | 1182.2 | 0.927 |
| 12 | ConvTransformer_4s | convtransformer | 4.0 | 6663 | 355.0 | 209.5 | 482.9 | 1294.0 | 0.919 |
| 13 | MLP_4s | mlp | 4.0 | 6663 | 409.0 | 233.0 | 532.0 | 1459.6 | 0.893 |
| 14 | Ridge | ridge_regression |  | 6763 | 1184.1 | 829.7 | 1889.9 | 3219.5 | 0.103 |

## Interpretation

- TCN variants were the strongest single models, suggesting that local temporal convolution is well matched to the PIR signal.
- GRU improved when context was extended to 6 seconds, but its frame mean joint error remained higher than the best TCN.
- Transformer-only variants overfit more easily on the current four-trial dataset.
- The best result came from averaging complementary TCN and GRU predictions.

## Caveat

This comparison uses a single held-out trial. A publication-ready claim should add trial-wise cross-validation and more recording sessions.