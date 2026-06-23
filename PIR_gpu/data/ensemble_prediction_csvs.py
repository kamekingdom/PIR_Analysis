import argparse
import csv
import json
from pathlib import Path

import numpy as np


def joint_indices(target_names):
    groups = {}
    axes = {"X": 0, "Y": 1, "Z": 2}
    for i, name in enumerate(target_names):
        base = name.split(" [", 1)[0]
        if "_" not in base:
            continue
        joint, axis = base.rsplit("_", 1)
        if axis in axes:
            groups.setdefault(joint, [None, None, None])[axes[axis]] = i
    return {joint: idx for joint, idx in groups.items() if all(v is not None for v in idx)}


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true, y_pred):
    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - sse / sst) if sst > 1e-12 else None


def frame_mean_joint_error(target_names, y_true, y_pred):
    groups = [idx for _, idx in sorted(joint_indices(target_names).items())]
    errors = [np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1) for idx in groups]
    return np.mean(np.column_stack(errors), axis=1)


def joint_rmse(target_names, y_true, y_pred):
    rows = []
    for joint, idx in sorted(joint_indices(target_names).items()):
        distances = np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1)
        rows.append({"joint": joint, "position_rmse_mm": float(np.sqrt(np.mean(distances**2)))})
    return rows


def prefixes(header):
    true_cols = [name for name in header if name.startswith("true_global_")]
    pred_cols = [name for name in header if name.startswith("pred_global_")]
    if not true_cols or not pred_cols:
        raise ValueError("Only true_global_/pred_global_ prediction CSVs are supported.")
    target_names = [name.removeprefix("true_global_") for name in true_cols]
    return target_names, true_cols, [f"pred_global_{name}" for name in target_names]


def read_prediction_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        target_names, true_cols, pred_cols = prefixes(reader.fieldnames or [])
        rows = {}
        for row in reader:
            key = (row["trial"], int(float(row["Frame"])))
            rows[key] = {
                "true": np.array([float(row[col]) for col in true_cols], dtype=np.float64),
                "pred": np.array([float(row[col]) for col in pred_cols], dtype=np.float64),
            }
    return target_names, rows


def write_predictions(path, keys, target_names, y_true, y_pred):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["trial", "Frame"]
        header.extend([f"true_global_{name}" for name in target_names])
        header.extend([f"pred_global_{name}" for name in target_names])
        writer.writerow(header)
        for (trial, frame), true_row, pred_row in zip(keys, y_true, y_pred):
            writer.writerow([trial, frame, *[f"{v:.6f}" for v in true_row], *[f"{v:.6f}" for v in pred_row]])


def write_frame_errors(path, keys, errors):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "Frame", "mean_joint_error_mm"])
        for (trial, frame), error in zip(keys, errors):
            writer.writerow([trial, frame, f"{error:.6f}"])


def parse_args():
    parser = argparse.ArgumentParser(description="Average prediction CSVs and evaluate the ensemble.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--weights", nargs="+", type=float, default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = args.names or [Path(path).parent.name for path in args.inputs]
    if len(names) != len(args.inputs):
        raise ValueError("--names length must match --inputs length.")
    weights = np.asarray(args.weights if args.weights else [1.0] * len(args.inputs), dtype=np.float64)
    weights = weights / weights.sum()

    loaded = [read_prediction_csv(path) for path in args.inputs]
    target_names = loaded[0][0]
    for next_target_names, _ in loaded[1:]:
        if next_target_names != target_names:
            raise ValueError("Target columns differ between prediction CSVs.")
    common = sorted(set.intersection(*[set(rows.keys()) for _, rows in loaded]), key=lambda item: (item[0], item[1]))
    if not common:
        raise ValueError("No common trial/frame rows across inputs.")

    y_true = np.vstack([loaded[0][1][key]["true"] for key in common])
    preds = np.stack([np.vstack([rows[key]["pred"] for key in common]) for _, rows in loaded], axis=0)
    y_pred = np.tensordot(weights, preds, axes=(0, 0))
    errors = frame_mean_joint_error(target_names, y_true, y_pred)
    metrics = {
        "type": "prediction_average_ensemble",
        "members": [{"name": name, "input": path, "weight": float(weight)} for name, path, weight in zip(names, args.inputs, weights)],
        "samples": {"test_rows": int(len(common))},
        "target_count": int(y_true.shape[1]),
        "target_names": target_names,
        "test": {
            "rmse_mm": rmse(y_true, y_pred),
            "mae_mm": mae(y_true, y_pred),
            "mean_joint_error_mm": float(np.mean(errors)),
            "p95_mean_joint_error_mm": float(np.percentile(errors, 95)),
            "r2_global": r2_score(y_true, y_pred),
        },
        "joint_test_rmse": joint_rmse(target_names, y_true, y_pred),
    }
    write_predictions(out_dir / "test_predictions.csv", common, target_names, y_true, y_pred)
    write_frame_errors(out_dir / "test_frame_errors.csv", common, errors)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output_dir": str(out_dir), "test": metrics["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
