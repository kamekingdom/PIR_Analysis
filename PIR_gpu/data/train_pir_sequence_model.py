import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def joint_indices(target_names):
    groups = {}
    axes = {"X": 0, "Y": 1, "Z": 2}
    for i, name in enumerate(target_names):
        base = str(name).split(" [", 1)[0]
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


def joint_rmse(target_names, y_true, y_pred):
    rows = []
    for joint, idx in sorted(joint_indices(target_names).items()):
        distances = np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1)
        rows.append({"joint": joint, "position_rmse_mm": float(np.sqrt(np.mean(distances**2)))})
    return rows


def frame_mean_joint_error(target_names, y_true, y_pred):
    groups = [idx for _, idx in sorted(joint_indices(target_names).items())]
    errors = [np.linalg.norm(y_pred[:, idx] - y_true[:, idx], axis=1) for idx in groups]
    return np.mean(np.column_stack(errors), axis=1)


def eligible_indices(trial_ids, mask, seq_len):
    ids = np.asarray(trial_ids)
    out = []
    for i in np.flatnonzero(mask):
        start = i - seq_len + 1
        if start < 0:
            continue
        if np.all(ids[start : i + 1] == ids[i]):
            out.append(i)
    return np.asarray(out, dtype=np.int64)


class SequenceDataset(Dataset):
    def __init__(self, x, y, indices, seq_len):
        self.x = x.astype(np.float32, copy=False)
        self.y = y.astype(np.float32, copy=False)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, item):
        end = int(self.indices[item])
        start = end - self.seq_len + 1
        return self.x[start : end + 1], self.y[end]


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding, dilation=dilation),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.padding = padding
        self.norm = nn.BatchNorm1d(channels)

    def forward(self, x):
        y = self.net(x)
        if self.padding:
            y = y[..., : -self.padding]
        if y.shape[-1] != x.shape[-1]:
            y = y[..., -x.shape[-1] :]
        return self.norm(x + y)


class TCNRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=192, layers=5, kernel_size=3, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        blocks = []
        for i in range(layers):
            blocks.append(ResidualBlock(hidden_dim, kernel_size, 2**i, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        z = self.blocks(self.in_proj(x))
        return self.head(z[:, :, -1])


class MLPWindowRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dim=512, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim * seq_len, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class GRURegressor(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=192, layers=2, dropout=0.1, bidirectional=True):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.in_proj(x)
        z, _ = self.gru(z)
        return self.head(z[:, -1])


class TransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dim=192, layers=4, heads=6, dropout=0.1):
        super().__init__()
        if hidden_dim % heads != 0:
            heads = 4
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.in_proj(x) + self.pos[:, : x.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1])


class ConvTransformerRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, seq_len, hidden_dim=192, layers=4, heads=6, dropout=0.1):
        super().__init__()
        if hidden_dim % heads != 0:
            heads = 4
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.pos = nn.Parameter(torch.zeros(1, seq_len, hidden_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        z = self.in_proj(x)
        z = z + self.conv(z.transpose(1, 2)).transpose(1, 2)
        z = z + self.pos[:, : z.shape[1]]
        z = self.encoder(z)
        return self.head(z[:, -1])


def make_model(kind, input_dim, output_dim, seq_len, hidden_dim, layers, dropout):
    if kind == "tcn":
        return TCNRegressor(input_dim, output_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if kind == "mlp":
        return MLPWindowRegressor(input_dim, output_dim, seq_len, hidden_dim=hidden_dim, dropout=dropout)
    if kind == "gru":
        return GRURegressor(input_dim, output_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if kind == "transformer":
        return TransformerRegressor(input_dim, output_dim, seq_len, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if kind == "convtransformer":
        return ConvTransformerRegressor(input_dim, output_dim, seq_len, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    raise ValueError(f"Unknown model: {kind}")


def evaluate(model, loader, device):
    model.eval()
    preds = []
    losses = []
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            losses.append(float(loss_fn(pred, yb).item()))
            preds.append(pred.detach().cpu().numpy())
    return np.vstack(preds), float(np.mean(losses))


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


def write_report(path, metrics):
    lines = [
        "# PIR時系列モデル",
        "",
        "## 概要",
        "",
        f"- モデル: {metrics['model']['type']}",
        f"- 実行デバイス: {metrics['device']}",
        f"- 学習trial: {', '.join(metrics['train_trials'])}",
        f"- 検証フレーム数: {metrics['validation']['rows']}",
        f"- テストtrial: {', '.join(metrics['test_trials'])}",
        f"- 文脈長: {metrics['model']['context_sec']}秒 ({metrics['model']['seq_len']}フレーム)",
        f"- テストRMSE: {metrics['test']['rmse_mm']:.1f} mm",
        f"- テストMAE: {metrics['test']['mae_mm']:.1f} mm",
        f"- テスト平均関節誤差: {metrics['test']['mean_joint_error_mm']:.1f} mm",
        f"- テストR2: {metrics['test']['r2_global']:.3f}",
        "",
        "## 関節別RMSE",
        "",
        "| 関節 | RMSE [mm] |",
        "|---|---:|",
    ]
    for row in metrics["joint_test_rmse"]:
        lines.append(f"| {row['joint']} | {row['position_rmse_mm']:.1f} |")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a sequence model from aligned PIR features to Theia coordinates.")
    parser.add_argument("--dataset", default="PIR_gpu/model_output/baseline_ridge_v3/aligned_dataset.npz")
    parser.add_argument("--output-dir", default="PIR_gpu/model_output/sequence_tcn")
    parser.add_argument("--model", choices=["tcn", "mlp", "gru", "transformer", "convtransformer"], default="tcn")
    parser.add_argument("--test-trials", default="005")
    parser.add_argument("--val-trials", default=None, help="Optional comma-separated validation trial IDs.")
    parser.add_argument("--context-sec", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset)
    x_raw = data["x"].astype(np.float32)
    y_raw = data["y"].astype(np.float32)
    frames = data["frames"]
    trial_ids = data["trial_ids"].astype(str)
    feature_names = [str(v) for v in data["feature_names"]]
    target_names = [str(v) for v in data["target_names"]]
    frame_rate = float(data["frame_rate"][0])
    seq_len = int(round(args.context_sec * frame_rate)) + 1

    test_trials = {item.strip() for item in args.test_trials.split(",") if item.strip()}
    train_mask = np.array([trial not in test_trials for trial in trial_ids])
    test_mask = np.array([trial in test_trials for trial in trial_ids])
    if args.val_trials:
        val_trials = {item.strip() for item in args.val_trials.split(",") if item.strip()}
        val_mask = np.array([trial in val_trials for trial in trial_ids])
        train_mask = train_mask & ~val_mask
    else:
        val_mask = np.zeros_like(train_mask, dtype=bool)
        for trial in sorted(set(trial_ids[train_mask])):
            idx = np.flatnonzero((trial_ids == trial) & train_mask)
            cut = int(math.floor(idx.shape[0] * 0.85))
            val_mask[idx[cut:]] = True
            train_mask[idx[cut:]] = False

    train_idx = eligible_indices(trial_ids, train_mask, seq_len)
    val_idx = eligible_indices(trial_ids, val_mask, seq_len)
    test_idx = eligible_indices(trial_ids, test_mask, seq_len)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Empty train/validation/test split after sequence windowing.")

    x_mean = x_raw[train_idx].mean(axis=0)
    x_std = x_raw[train_idx].std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = y_raw[train_idx].mean(axis=0)
    y_std = y_raw[train_idx].std(axis=0)
    y_std[y_std < 1e-6] = 1.0
    x = (x_raw - x_mean) / x_std
    y = (y_raw - y_mean) / y_std

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    model = make_model(args.model, x.shape[1], y.shape[1], seq_len, args.hidden_dim, args.layers, args.dropout).to(device)
    train_loader = DataLoader(
        SequenceDataset(x, y, train_idx, seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(SequenceDataset(x, y, val_idx, seq_len), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x, y, test_idx, seq_len), batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
    loss_fn = nn.SmoothL1Loss(beta=0.5)
    best = {"loss": float("inf"), "epoch": 0, "state": None}
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))
        _, val_loss = evaluate(model, val_loader, device)
        scheduler.step(val_loss)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "val_loss": val_loss,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best["loss"]:
            best = {"loss": val_loss, "epoch": epoch, "state": {k: v.detach().cpu() for k, v in model.state_dict().items()}}
        elif epoch - best["epoch"] >= args.patience:
            break

    model.load_state_dict(best["state"])
    pred_test_z, _ = evaluate(model, test_loader, device)
    pred_train_z, _ = evaluate(model, DataLoader(SequenceDataset(x, y, train_idx, seq_len), batch_size=args.batch_size), device)
    y_test = y_raw[test_idx]
    y_train = y_raw[train_idx]
    pred_test = pred_test_z * y_std + y_mean
    pred_train = pred_train_z * y_std + y_mean
    test_errors = frame_mean_joint_error(target_names, y_test, pred_test)
    train_errors = frame_mean_joint_error(target_names, y_train, pred_train)
    baseline_test = np.repeat(y_train.mean(axis=0, keepdims=True), y_test.shape[0], axis=0)

    metrics = {
        "dataset": str(args.dataset),
        "device": str(device),
        "train_trials": sorted(set(trial_ids[train_idx].tolist())),
        "test_trials": sorted(test_trials),
        "samples": {
            "train_rows": int(len(train_idx)),
            "validation_rows": int(len(val_idx)),
            "test_rows": int(len(test_idx)),
        },
        "validation": {"best_epoch": int(best["epoch"]), "best_loss": float(best["loss"]), "rows": int(len(val_idx))},
        "model": {
            "type": args.model,
            "context_sec": float(args.context_sec),
            "seq_len": int(seq_len),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "dropout": float(args.dropout),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
        },
        "feature_count": int(x.shape[1]),
        "target_count": int(y.shape[1]),
        "target_names": target_names,
        "feature_names": feature_names,
        "train": {
            "rmse_mm": rmse(y_train, pred_train),
            "mae_mm": mae(y_train, pred_train),
            "mean_joint_error_mm": float(np.mean(train_errors)),
            "p95_mean_joint_error_mm": float(np.percentile(train_errors, 95)),
            "r2_global": r2_score(y_train, pred_train),
        },
        "test": {
            "rmse_mm": rmse(y_test, pred_test),
            "mae_mm": mae(y_test, pred_test),
            "mean_joint_error_mm": float(np.mean(test_errors)),
            "p95_mean_joint_error_mm": float(np.percentile(test_errors, 95)),
            "r2_global": r2_score(y_test, pred_test),
            "baseline_rmse_mm": rmse(y_test, baseline_test),
            "baseline_mae_mm": mae(y_test, baseline_test),
        },
        "joint_test_rmse": joint_rmse(target_names, y_test, pred_test),
        "history": history,
    }

    write_predictions(out_dir / "test_predictions.csv", trial_ids[test_idx], frames[test_idx], target_names, y_test, pred_test)
    write_frame_errors(out_dir / "test_frame_errors.csv", trial_ids[test_idx], frames[test_idx], test_errors)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    write_report(out_dir / "report.md", metrics)
    torch.save(
        {
            "model_state": model.state_dict(),
            "args": vars(args),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "metrics": metrics,
        },
        out_dir / "model.pt",
    )
    print(json.dumps({"output_dir": str(out_dir), "device": str(device), "test": metrics["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
