import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from train_pir_to_local_bones import DEFAULT_LAGS, build_features, read_pir_trial


def load_model(path):
    model = np.load(path, allow_pickle=False)
    metadata = json.loads(str(model["metadata_json"][0]))
    return {
        "coef": model["coef"],
        "x_mean": model["x_mean"],
        "x_std": model["x_std"],
        "feature_names": [str(item) for item in model["feature_names"]],
        "target_names": [str(item) for item in model["target_names"]],
        "metadata": metadata,
    }


def predict_ridge(x, coef):
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    return x_aug @ coef


def make_prediction_times(modules, frame_rate):
    start = max(module["times"][0] for module in modules) + max(DEFAULT_LAGS)
    end = min(module["times"][-1] for module in modules)
    step = 1.0 / frame_rate
    if end <= start:
        raise ValueError("PIR data is too short for the configured lag window.")
    return np.arange(start, end + step * 0.5, step, dtype=float)


def write_prediction_csv(path, times, target_names, pred):
    start = times[0]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Frame", "timestamp_local", "Time [sec]", *[f"pred_global_{name}" for name in target_names]])
        for frame, (time_value, row) in enumerate(zip(times, pred)):
            stamp = datetime.fromtimestamp(float(time_value)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            writer.writerow([frame, stamp, f"{time_value - start:.6f}", *[f"{value:.6f}" for value in row]])


def parse_args():
    parser = argparse.ArgumentParser(description="Predict Theia global bone coordinates from PIR data.")
    parser.add_argument("--trial-dir", required=True, help="Trimmed trial folder containing a pir directory.")
    parser.add_argument(
        "--model",
        default="analysis/20260623/model_output_global/pir_to_global_bones_ridge_model.npz",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--frame-rate", type=float, default=25.0)
    return parser.parse_args()


def main():
    args = parse_args()
    trial_dir = Path(args.trial_dir)
    model = load_model(args.model)
    preprocessing = model["metadata"].get("preprocessing", {})
    pir_time_source = preprocessing.get("pir_time_source", "auto")
    modules, _ = read_pir_trial(trial_dir, time_source=pir_time_source)
    times = make_prediction_times(modules, args.frame_rate)
    x, feature_names = build_features(modules, times)
    if feature_names != model["feature_names"]:
        raise ValueError("PIR feature layout differs from the trained model.")
    xz = (x - model["x_mean"]) / model["x_std"]
    pred = predict_ridge(xz, model["coef"])

    output = Path(args.output) if args.output else trial_dir / "predicted_global_bones.csv"
    write_prediction_csv(output, times, model["target_names"], pred)
    print(json.dumps({"output": str(output), "frames": int(pred.shape[0])}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
