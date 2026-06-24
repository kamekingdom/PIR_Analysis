import argparse
import csv
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np


PIR_SENSOR_PREFIX = "PIR sensor"
DEFAULT_LAGS = [0.0, 0.04, 0.08, 0.16, 0.32, 0.64, 1.0, 1.5, 2.0, 3.0]
DELTA_LAGS = [0.16, 0.64, 1.5, 3.0]
DEFAULT_SYNC_GRID_SEC = 0.04


def parse_datetime(value):
    text = str(value or "").strip()
    for fmt in (
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d, %H:%M:%S.%f",
        "%Y-%m-%d, %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    raise ValueError(f"Could not parse datetime: {value}")


def read_skeleton_meta(tsv_path):
    meta = {"frequency": 25.0, "timestamp": None, "frames": None}
    with Path(tsv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in handle:
            parts = raw.rstrip("\r\n").split("\t")
            if not parts:
                continue
            if parts[0] == "Frame":
                break
            if parts[0] == "FREQUENCY" and len(parts) > 1:
                meta["frequency"] = float(parts[1])
            elif parts[0] == "TIME_STAMP" and len(parts) > 1:
                meta["timestamp"] = parse_datetime(parts[1])
            elif parts[0] == "NO_OF_FRAMES" and len(parts) > 1:
                meta["frames"] = int(float(parts[1]))
    if meta["timestamp"] is None:
        raise ValueError(f"TIME_STAMP not found in {tsv_path}")
    return meta


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


def read_skeleton_trial(trial_dir):
    name_parts = trial_dir.name.split("_")
    if name_parts and name_parts[0].isdigit():
        source_label = ""
        skeleton_stem = name_parts[0]
        trial_id = skeleton_stem
    elif len(name_parts) >= 2 and name_parts[1].isdigit():
        source_label = name_parts[0]
        skeleton_stem = name_parts[1]
        trial_id = f"{source_label}_{skeleton_stem}"
    else:
        source_label = ""
        skeleton_stem = trial_dir.name.split("_", 1)[0]
        trial_id = skeleton_stem
    skel_dir = trial_dir / "skelton"
    tsv_path = skel_dir / f"{skeleton_stem}.tsv"
    csv_path = skel_dir / "Theia_Sub0.csv"
    meta = read_skeleton_meta(tsv_path)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        rows = [[float(value) if value else 0.0 for value in row] for row in reader]

    data = np.asarray(rows, dtype=float)
    frames = data[:, 0].astype(int)
    y_abs = data[:, 1:]
    raw_target_names = header[1:]
    groups = joint_indices(raw_target_names)

    valid = np.isfinite(y_abs).all(axis=1) & (np.abs(y_abs).sum(axis=1) > 0)

    keep_cols = []
    target_names = []
    joint_names = []
    for joint, idx in sorted(groups.items()):
        if joint == "worldbody":
            continue
        keep_cols.extend(idx)
        target_names.extend([raw_target_names[i] for i in idx])
        joint_names.append(joint)

    frame_times = np.array(
        [(meta["timestamp"] + timedelta(seconds=float(frame) / meta["frequency"])).timestamp() for frame in frames],
        dtype=float,
    )
    return {
        "trial_id": trial_id,
        "source_label": source_label,
        "frames": frames,
        "times": frame_times,
        "y": y_abs[:, keep_cols],
        "valid": valid,
        "target_names": target_names,
        "joint_names": joint_names,
        "frame_rate": float(meta["frequency"]),
    }


def read_pir_times(rows, time_source):
    source = time_source
    if source == "auto":
        source = "timestamp_unix" if rows[0].get("timestamp_unix") else "timestamp_local"
    if source == "timestamp_unix":
        return np.array([float(row["timestamp_unix"]) for row in rows], dtype=float), source
    if source == "time_sec":
        return np.array([float(row["Time [sec]"]) for row in rows], dtype=float), source
    if source == "timestamp_local":
        return np.array([parse_datetime(row["timestamp_local"]).timestamp() for row in rows], dtype=float), source
    raise ValueError(f"Unknown PIR time source: {time_source}")


def read_pir_module(path, time_source="auto"):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    sensor_cols = [name for name in reader.fieldnames or [] if name.startswith(PIR_SENSOR_PREFIX)]
    times, resolved_time_source = read_pir_times(rows, time_source)
    values = np.array([[float(row[col]) for col in sensor_cols] for row in rows], dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    unique_times, inverse = np.unique(times, return_inverse=True)
    if unique_times.shape[0] != times.shape[0]:
        summed = np.zeros((unique_times.shape[0], values.shape[1]), dtype=float)
        counts = np.zeros(unique_times.shape[0], dtype=float)
        np.add.at(summed, inverse, values)
        np.add.at(counts, inverse, 1.0)
        times = unique_times
        values = summed / counts[:, None]
    module = rows[0].get("module_address") or path.stem.replace("pir_data_", "")
    return {
        "path": str(path),
        "module": module,
        "sensor_cols": sensor_cols,
        "time_source": resolved_time_source,
        "times": times,
        "values": values,
    }


def read_pir_trial(trial_dir, time_source="auto"):
    modules = []
    for path in sorted((trial_dir / "pir").glob("pir_data_*.csv")):
        module = read_pir_module(path, time_source=time_source)
        if module is not None:
            modules.append(module)
    modules.sort(key=lambda item: item["module"])
    if not modules:
        raise FileNotFoundError(f"No PIR data found in {trial_dir / 'pir'}")
    feature_names = []
    for module in modules:
        for sensor in module["sensor_cols"]:
            feature_names.append(f"{module['module']}:{sensor}")
    return modules, feature_names


def interpolate_modules(modules, query_times):
    pieces = []
    for module in modules:
        values = module["values"]
        times = module["times"]
        interp = np.column_stack([np.interp(query_times, times, values[:, i]) for i in range(values.shape[1])])
        pieces.append(interp)
    return np.concatenate(pieces, axis=1)


def build_features(modules, frame_times, pir_time_offset_sec=0.0):
    base_names = []
    for module in modules:
        for sensor in module["sensor_cols"]:
            base_names.append(f"{module['module']}:{sensor}")

    pieces = []
    feature_names = []
    current = None
    for lag in DEFAULT_LAGS:
        values = interpolate_modules(modules, frame_times + pir_time_offset_sec - lag)
        if lag == 0.0:
            current = values
        pieces.append(values)
        feature_names.extend([f"{name}:lag_{lag:.2f}s" for name in base_names])

    for lag in DELTA_LAGS:
        values = interpolate_modules(modules, frame_times + pir_time_offset_sec - lag)
        pieces.append(current - values)
        feature_names.extend([f"{name}:delta_{lag:.2f}s" for name in base_names])

    derivatives = []
    for module in modules:
        times = module["times"]
        values = module["values"]
        for i in range(values.shape[1]):
            grad = np.gradient(values[:, i], times)
            derivatives.append(np.interp(frame_times + pir_time_offset_sec, times, grad))
    pieces.append(np.column_stack(derivatives))
    feature_names.extend([f"{name}:d_dt" for name in base_names])
    return np.concatenate(pieces, axis=1), feature_names


def motion_energy_from_pir(modules, query_times):
    values = interpolate_modules(modules, query_times)
    centered = values - np.median(values, axis=0, keepdims=True)
    scale = np.median(np.abs(centered), axis=0, keepdims=True)
    scale[scale < 1e-6] = values.std(axis=0, keepdims=True)[scale < 1e-6]
    scale[scale < 1e-6] = 1.0
    z = centered / scale
    return np.mean(np.abs(z), axis=1)


def motion_energy_from_skeleton(y, frame_rate):
    joints = y.reshape(y.shape[0], -1, 3)
    velocity = np.gradient(joints, 1.0 / frame_rate, axis=0)
    return np.mean(np.linalg.norm(velocity, axis=2), axis=1)


def normalized_correlation(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 25:
        return None
    a = a[mask] - a[mask].mean()
    b = b[mask] - b[mask].mean()
    denom = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if denom < 1e-12:
        return None
    return float(np.sum(a * b) / denom)


def estimate_pir_time_offset(modules, frame_times, y, valid, frame_rate, max_offset_sec, grid_sec):
    pir_start = max(module["times"][0] for module in modules)
    pir_end = min(module["times"][-1] for module in modules)
    skeleton_energy = motion_energy_from_skeleton(y, frame_rate)
    offsets = np.arange(-max_offset_sec, max_offset_sec + grid_sec * 0.5, grid_sec, dtype=float)
    rows = []
    for offset in offsets:
        mask = valid & (frame_times + offset >= pir_start) & (frame_times + offset <= pir_end)
        corr = normalized_correlation(motion_energy_from_pir(modules, frame_times[mask] + offset), skeleton_energy[mask])
        if corr is not None:
            rows.append({"offset_sec": float(offset), "correlation": corr, "frames": int(mask.sum())})
    if not rows:
        return 0.0, []
    best = max(rows, key=lambda row: row["correlation"])
    return best["offset_sec"], rows


def load_trial(
    trial_dir,
    pir_time_source="auto",
    sync_mode="timestamps",
    max_sync_offset_sec=3.0,
    sync_grid_sec=DEFAULT_SYNC_GRID_SEC,
):
    skeleton = read_skeleton_trial(trial_dir)
    modules, _ = read_pir_trial(trial_dir, time_source=pir_time_source)
    if modules[0]["time_source"] == "time_sec":
        frame_times = skeleton["frames"].astype(float) / skeleton["frame_rate"]
    else:
        frame_times = skeleton["times"]
    pir_time_offset_sec = 0.0
    sync_candidates = []
    if sync_mode == "correlation":
        pir_time_offset_sec, sync_candidates = estimate_pir_time_offset(
            modules,
            frame_times,
            skeleton["y"],
            skeleton["valid"],
            skeleton["frame_rate"],
            max_sync_offset_sec,
            sync_grid_sec,
        )
    elif sync_mode != "timestamps":
        raise ValueError(f"Unknown sync mode: {sync_mode}")

    pir_start = max(module["times"][0] for module in modules)
    pir_end = min(module["times"][-1] for module in modules)
    usable = (
        skeleton["valid"]
        & (frame_times + pir_time_offset_sec >= pir_start + max(DEFAULT_LAGS))
        & (frame_times + pir_time_offset_sec <= pir_end)
    )
    if usable.sum() < 100:
        raise ValueError(f"Too few usable aligned frames in {trial_dir}")

    x, feature_names = build_features(modules, frame_times[usable], pir_time_offset_sec=pir_time_offset_sec)
    return {
        "trial_id": skeleton["trial_id"],
        "folder": str(trial_dir),
        "frames": skeleton["frames"][usable],
        "times": frame_times[usable],
        "x": x,
        "y": skeleton["y"][usable],
        "target_names": skeleton["target_names"],
        "joint_names": skeleton["joint_names"],
        "feature_names": feature_names,
        "frame_rate": skeleton["frame_rate"],
        "pir_time_source": modules[0]["time_source"],
        "pir_time_offset_sec": float(pir_time_offset_sec),
        "sync_candidates": sync_candidates,
    }


def standardize(x_train, x_other):
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-9] = 1.0
    return (x_train - mean) / std, (x_other - mean) / std, mean, std


def fit_ridge(x_train, y_train, alpha):
    x_aug = np.column_stack([np.ones(x_train.shape[0]), x_train])
    reg = np.eye(x_aug.shape[1], dtype=float) * alpha
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train)


def predict_ridge(x, coef):
    x_aug = np.column_stack([np.ones(x.shape[0]), x])
    return x_aug @ coef


def choose_alpha(x_train, y_train, candidates):
    n = x_train.shape[0]
    split = int(n * 0.8)
    x_fit, x_val = x_train[:split], x_train[split:]
    y_fit, y_val = y_train[:split], y_train[split:]
    x_fit_z, x_val_z, _, _ = standardize(x_fit, x_val)
    rows = []
    for alpha in candidates:
        coef = fit_ridge(x_fit_z, y_fit, alpha)
        pred = predict_ridge(x_val_z, coef)
        rows.append({"alpha": float(alpha), "validation_rmse_mm": rmse(y_val, pred)})
    best = min(rows, key=lambda row: row["validation_rmse_mm"])
    return best["alpha"], rows


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score(y_true, y_pred):
    sse = float(np.sum((y_pred - y_true) ** 2))
    sst = float(np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2))
    return float(1.0 - sse / sst) if sst > 1e-12 else None


def joint_rmse(target_names, y_true, y_pred):
    groups = joint_indices(target_names)
    rows = []
    for joint, idx in sorted(groups.items()):
        distances = np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1)
        rows.append({"joint": joint, "position_rmse_mm": float(np.sqrt(np.mean(distances**2)))})
    return rows


def frame_mean_joint_error(target_names, y_true, y_pred):
    idx_groups = joint_indices(target_names)
    ordered = [idx for _, idx in sorted(idx_groups.items())]
    errors = []
    for idx in ordered:
        errors.append(np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1))
    if not errors:
        return np.zeros(y_true.shape[0], dtype=float)
    return np.mean(np.column_stack(errors), axis=1)


def concatenate_trials(trials):
    return (
        np.vstack([trial["x"] for trial in trials]),
        np.vstack([trial["y"] for trial in trials]),
        np.concatenate([trial["frames"] for trial in trials]),
        np.concatenate([[trial["trial_id"]] * trial["x"].shape[0] for trial in trials]),
    )


def write_predictions(path, trial_ids, frames, target_names, y_true, y_pred):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["trial", "Frame"]
        header.extend([f"true_global_{name}" for name in target_names])
        header.extend([f"pred_global_{name}" for name in target_names])
        writer.writerow(header)
        for trial, frame, true_row, pred_row in zip(trial_ids, frames, y_true, y_pred):
            writer.writerow([trial, int(frame), *[f"{v:.6f}" for v in true_row], *[f"{v:.6f}" for v in pred_row]])


def write_frame_errors(path, trial_ids, frames, errors):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["trial", "Frame", "mean_joint_error_mm"])
        for trial, frame, error in zip(trial_ids, frames, errors):
            writer.writerow([trial, int(frame), f"{error:.6f}"])


def write_aligned_dataset(path, trials):
    x, y, frames, trial_ids = concatenate_trials(trials)
    np.savez_compressed(
        path,
        x=x,
        y=y,
        frames=frames,
        trial_ids=trial_ids,
        feature_names=np.array(trials[0]["feature_names"]),
        target_names=np.array(trials[0]["target_names"]),
        frame_rate=np.array([trials[0]["frame_rate"]], dtype=float),
    )


def write_report(path, metrics):
    mean_joint_error = metrics["test"].get("mean_joint_error_mm")
    preprocessing = metrics.get("preprocessing", {})
    lines = [
        "# 20260623 PIRからTheia3D Global Bone座標を推定するモデル",
        "",
        "## 概要",
        "",
        f"- 学習trial: {', '.join(metrics['train_trials'])}",
        f"- テストtrial: {', '.join(metrics['test_trials'])}",
        f"- 特徴量数: {metrics['feature_count']}",
        f"- 予測対象数: {metrics['target_count']}個のglobal座標値",
        f"- Ridge alpha: {metrics['model']['alpha']}",
        f"- テストRMSE: {metrics['test']['rmse_mm']:.1f} mm",
        f"- テストMAE: {metrics['test']['mae_mm']:.1f} mm",
        f"- テスト平均関節誤差: {mean_joint_error:.1f} mm" if mean_joint_error is not None else "- テスト平均関節誤差: 未算出",
        f"- テストR2: {metrics['test']['r2_global']:.3f}",
        f"- 同期モード: {preprocessing.get('sync_mode', '未記録')}",
        f"- PIR時刻ソース: {preprocessing.get('pir_time_source', '未記録')}",
        "",
        "予測対象はTheiaのglobal joint座標です。`worldbody` は予測対象から除外しています。",
        "",
        "## 関節別RMSE",
        "",
        "| 関節 | RMSE [mm] |",
        "|---|---:|",
    ]
    for row in metrics["joint_test_rmse"]:
        lines.append(f"| {row['joint']} | {row['position_rmse_mm']:.1f} |")
    lines.extend(
        [
            "",
            "## 出力ファイル",
            "",
            "- `pir_to_global_bones_ridge_model.npz`: モデルパラメータとメタデータ",
            "- `test_predictions.csv`: 未使用テストtrialの予測結果",
            "- `test_frame_errors.csv`: 未使用テストframeごとの平均関節誤差",
            "- `aligned_dataset.npz`: 25Hzに整列した特徴量と教師座標",
            "- `metrics.json`: 全評価指標",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train PIR sensor data to Theia global bone coordinates.")
    parser.add_argument("--trimmed-dir", default="analysis/20260623/trimmed")
    parser.add_argument("--output-dir", default="analysis/20260623/model_output_global")
    parser.add_argument("--test-trials", default=None, help="Comma separated trial IDs. Default: last trial.")
    parser.add_argument("--alpha", default="auto")
    parser.add_argument(
        "--pir-time-source",
        default="auto",
        choices=["auto", "timestamp_unix", "timestamp_local", "time_sec"],
        help="PIR timestamp column to use. auto prefers timestamp_unix.",
    )
    parser.add_argument(
        "--sync-mode",
        default="timestamps",
        choices=["timestamps", "correlation"],
        help="timestamps trusts file clocks; correlation estimates a per-trial PIR time offset.",
    )
    parser.add_argument("--max-sync-offset-sec", type=float, default=3.0)
    parser.add_argument("--sync-grid-sec", type=float, default=DEFAULT_SYNC_GRID_SEC)
    return parser.parse_args()


def main():
    args = parse_args()
    trimmed_dir = Path(args.trimmed_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trial_dirs = sorted(path for path in trimmed_dir.iterdir() if path.is_dir() and (path / "pir").exists())
    if len(trial_dirs) < 2:
        raise ValueError("Need at least two trimmed trials for train/test evaluation.")

    loaded = [
        load_trial(
            path,
            pir_time_source=args.pir_time_source,
            sync_mode=args.sync_mode,
            max_sync_offset_sec=args.max_sync_offset_sec,
            sync_grid_sec=args.sync_grid_sec,
        )
        for path in trial_dirs
    ]
    target_names = loaded[0]["target_names"]
    feature_names = loaded[0]["feature_names"]
    for trial in loaded[1:]:
        if trial["target_names"] != target_names:
            raise ValueError("Target columns differ between trials.")
        if trial["feature_names"] != feature_names:
            raise ValueError("PIR feature columns differ between trials.")

    if args.test_trials:
        test_ids = {item.strip() for item in args.test_trials.split(",") if item.strip()}
    else:
        test_ids = {loaded[-1]["trial_id"]}
    train_trials = [trial for trial in loaded if trial["trial_id"] not in test_ids]
    test_trials = [trial for trial in loaded if trial["trial_id"] in test_ids]
    if not train_trials or not test_trials:
        raise ValueError("Train/test split is empty. Check --test-trials.")

    x_train, y_train, train_frames, train_ids = concatenate_trials(train_trials)
    x_test, y_test, test_frames, test_ids_arr = concatenate_trials(test_trials)

    if str(args.alpha).lower() == "auto":
        alpha, alpha_validation = choose_alpha(
            x_train,
            y_train,
            [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0],
        )
    else:
        alpha = float(args.alpha)
        alpha_validation = []

    x_train_z, x_test_z, x_mean, x_std = standardize(x_train, x_test)
    coef = fit_ridge(x_train_z, y_train, alpha)
    pred_train = predict_ridge(x_train_z, coef)
    pred_test = predict_ridge(x_test_z, coef)
    train_frame_errors = frame_mean_joint_error(target_names, y_train, pred_train)
    test_frame_errors = frame_mean_joint_error(target_names, y_test, pred_test)

    train_mean = y_train.mean(axis=0, keepdims=True)
    baseline_test = np.repeat(train_mean, y_test.shape[0], axis=0)
    metrics = {
        "trimmed_dir": str(trimmed_dir),
        "train_trials": [trial["trial_id"] for trial in train_trials],
        "test_trials": [trial["trial_id"] for trial in test_trials],
        "samples": {
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "by_trial": {trial["trial_id"]: int(trial["x"].shape[0]) for trial in loaded},
        },
        "feature_count": int(x_train.shape[1]),
        "target_count": int(y_train.shape[1]),
        "target_names": target_names,
        "feature_names": feature_names,
        "preprocessing": {
            "pir_time_source": loaded[0]["pir_time_source"],
            "sync_mode": args.sync_mode,
            "max_sync_offset_sec": float(args.max_sync_offset_sec),
            "sync_grid_sec": float(args.sync_grid_sec),
            "trial_offsets_sec": {trial["trial_id"]: trial["pir_time_offset_sec"] for trial in loaded},
            "sync_candidates": {
                trial["trial_id"]: trial["sync_candidates"] for trial in loaded if trial["sync_candidates"]
            },
            "alignment_rate_hz": float(loaded[0]["frame_rate"]),
        },
        "model": {
            "type": "ridge_regression",
            "target_mode": "theia_global",
            "alpha": float(alpha),
            "alpha_validation": alpha_validation,
            "lags_sec": DEFAULT_LAGS,
            "delta_lags_sec": DELTA_LAGS,
        },
        "train": {
            "rmse_mm": rmse(y_train, pred_train),
            "mae_mm": mae(y_train, pred_train),
            "mean_joint_error_mm": float(np.mean(train_frame_errors)),
            "p95_mean_joint_error_mm": float(np.percentile(train_frame_errors, 95)),
            "r2_global": r2_score(y_train, pred_train),
        },
        "test": {
            "rmse_mm": rmse(y_test, pred_test),
            "mae_mm": mae(y_test, pred_test),
            "mean_joint_error_mm": float(np.mean(test_frame_errors)),
            "p95_mean_joint_error_mm": float(np.percentile(test_frame_errors, 95)),
            "r2_global": r2_score(y_test, pred_test),
            "baseline_rmse_mm": rmse(y_test, baseline_test),
            "baseline_mae_mm": mae(y_test, baseline_test),
        },
        "joint_test_rmse": joint_rmse(target_names, y_test, pred_test),
    }

    write_predictions(out_dir / "test_predictions.csv", test_ids_arr, test_frames, target_names, y_test, pred_test)
    write_frame_errors(out_dir / "test_frame_errors.csv", test_ids_arr, test_frames, test_frame_errors)
    write_aligned_dataset(out_dir / "aligned_dataset.npz", loaded)
    np.savez_compressed(
        out_dir / "pir_to_global_bones_ridge_model.npz",
        coef=coef,
        x_mean=x_mean,
        x_std=x_std,
        feature_names=np.array(feature_names),
        target_names=np.array(target_names),
        train_trials=np.array(metrics["train_trials"]),
        test_trials=np.array(metrics["test_trials"]),
        alpha=np.array([alpha], dtype=float),
        metadata_json=np.array([json.dumps(metrics, ensure_ascii=False)]),
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    write_report(out_dir / "report.md", metrics)

    print(json.dumps({"output_dir": str(out_dir), "test": metrics["test"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
