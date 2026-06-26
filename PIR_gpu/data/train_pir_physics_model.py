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

from train_pir_sequence_model import (
    ResidualBlock,
    TCNRegressor,
    eligible_indices,
    frame_mean_joint_error,
    joint_indices,
    joint_rmse,
    mae,
    rmse,
    r2_score,
    write_frame_errors,
    write_predictions,
    write_report,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_feature_name(name):
    module, sensor, feature = str(name).split(":", 2)
    return module, sensor, feature


def build_sensor_layout(feature_names):
    parsed = [parse_feature_name(name) for name in feature_names]
    modules = sorted({module for module, _, _ in parsed})
    sensors = sorted({sensor for _, sensor, _ in parsed})
    feature_order = []
    for prefix in ("lag_", "delta_"):
        values = sorted(
            {
                float(feature.split("_", 1)[1].rstrip("s"))
                for _, _, feature in parsed
                if feature.startswith(prefix)
            }
        )
        feature_order.extend([f"{prefix}{value:.2f}s" for value in values])
    if any(feature == "d_dt" for _, _, feature in parsed):
        feature_order.append("d_dt")

    index_by_key = {(module, sensor, feature): i for i, (module, sensor, feature) in enumerate(parsed)}
    order = []
    missing = []
    for module in modules:
        for sensor in sensors:
            for feature in feature_order:
                key = (module, sensor, feature)
                if key not in index_by_key:
                    missing.append(key)
                else:
                    order.append(index_by_key[key])
    if missing:
        raise ValueError(f"Feature grid is incomplete. First missing key: {missing[0]}")
    return {
        "modules": modules,
        "sensors": sensors,
        "feature_order": feature_order,
        "order": np.asarray(order, dtype=np.int64),
    }


def skeleton_edges(target_names):
    groups = joint_indices(target_names)
    edge_names = [
        ("pelvis", "abdomen"),
        ("abdomen", "thorax"),
        ("thorax", "neck"),
        ("neck", "head"),
        ("thorax", "l_uarm"),
        ("l_uarm", "l_larm"),
        ("l_larm", "l_hand"),
        ("thorax", "r_uarm"),
        ("r_uarm", "r_larm"),
        ("r_larm", "r_hand"),
        ("pelvis", "l_thigh"),
        ("l_thigh", "l_shank"),
        ("l_shank", "l_foot"),
        ("l_foot", "l_toes"),
        ("pelvis", "r_thigh"),
        ("r_thigh", "r_shank"),
        ("r_shank", "r_foot"),
        ("r_foot", "r_toes"),
    ]
    out = []
    for a, b in edge_names:
        if a in groups and b in groups:
            out.append((groups[a], groups[b], a, b))
    return out


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


class StructuredPIRFrameEncoder(nn.Module):
    def __init__(self, feature_order, module_count, sensor_count, hidden_dim, dropout, use_token_transformer=False):
        super().__init__()
        self.module_count = int(module_count)
        self.sensor_count = int(sensor_count)
        self.feature_count = int(feature_order.shape[0] // (module_count * sensor_count))
        self.register_buffer("feature_order", torch.as_tensor(feature_order, dtype=torch.long), persistent=False)
        self.token_proj = nn.Sequential(
            nn.Linear(self.feature_count, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.module_embed = nn.Parameter(torch.zeros(1, 1, self.module_count, 1, hidden_dim))
        self.sensor_embed = nn.Parameter(torch.zeros(1, 1, 1, self.sensor_count, hidden_dim))
        nn.init.normal_(self.module_embed, std=0.02)
        nn.init.normal_(self.sensor_embed, std=0.02)
        self.use_token_transformer = use_token_transformer
        if use_token_transformer:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.token_score = nn.Linear(hidden_dim, 1)
        self.out_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        bsz, steps, _ = x.shape
        x = x.index_select(-1, self.feature_order)
        x = x.view(bsz, steps, self.module_count, self.sensor_count, self.feature_count)
        z = self.token_proj(x)
        z = z + self.module_embed + self.sensor_embed
        z = z.view(bsz * steps, self.module_count * self.sensor_count, -1)
        if self.use_token_transformer:
            z = self.token_encoder(z)
        weights = torch.softmax(self.token_score(z), dim=1)
        z = torch.sum(weights * z, dim=1)
        return self.out_norm(z.view(bsz, steps, -1))


class StructuredTCNRegressor(nn.Module):
    def __init__(
        self,
        feature_order,
        module_count,
        sensor_count,
        output_dim,
        hidden_dim=192,
        layers=5,
        dropout=0.1,
        use_token_transformer=False,
        experts=1,
    ):
        super().__init__()
        self.frame_encoder = StructuredPIRFrameEncoder(
            feature_order,
            module_count,
            sensor_count,
            hidden_dim,
            dropout,
            use_token_transformer=use_token_transformer,
        )
        blocks = [ResidualBlock(hidden_dim, 3, 2**i, dropout) for i in range(layers)]
        self.temporal = nn.Sequential(*blocks)
        self.experts = int(experts)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.experts > 1:
            self.router = nn.Linear(hidden_dim, self.experts)
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(self.experts)])
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.frame_encoder(x).transpose(1, 2)
        z = self.temporal(z)[:, :, -1]
        h = self.shared_head(z)
        if self.experts > 1:
            gates = torch.softmax(self.router(h), dim=-1)
            preds = torch.stack([head(h) for head in self.heads], dim=1)
            return torch.sum(gates.unsqueeze(-1) * preds, dim=1)
        return self.head(h)


class HybridStructuredTCNRegressor(nn.Module):
    def __init__(
        self,
        input_dim,
        feature_order,
        module_count,
        sensor_count,
        output_dim,
        hidden_dim=192,
        layers=5,
        dropout=0.1,
        use_token_transformer=False,
        experts=1,
    ):
        super().__init__()
        self.structured = StructuredPIRFrameEncoder(
            feature_order,
            module_count,
            sensor_count,
            hidden_dim,
            dropout,
            use_token_transformer=use_token_transformer,
        )
        self.raw = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.temporal = nn.Sequential(*[ResidualBlock(hidden_dim, 3, 2**i, dropout) for i in range(layers)])
        self.experts = int(experts)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.experts > 1:
            self.router = nn.Linear(hidden_dim, self.experts)
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, output_dim) for _ in range(self.experts)])
        else:
            self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z_struct = self.structured(x)
        z_raw = self.raw(x)
        z = self.fuse(torch.cat([z_raw, z_struct], dim=-1)).transpose(1, 2)
        z = self.temporal(z)[:, :, -1]
        h = self.shared_head(z)
        if self.experts > 1:
            gates = torch.softmax(self.router(h), dim=-1)
            preds = torch.stack([head(h) for head in self.heads], dim=1)
            return torch.sum(gates.unsqueeze(-1) * preds, dim=1)
        return self.head(h)


class RootLocalTCNRegressor(nn.Module):
    def __init__(self, input_dim, output_dim, target_names, y_mean, y_std, hidden_dim=192, layers=5, dropout=0.1):
        super().__init__()
        groups = joint_indices(target_names)
        if "pelvis" not in groups:
            raise ValueError("RootLocalTCNRegressor requires a pelvis joint in target_names.")
        self.output_dim = int(output_dim)
        self.joint_count = output_dim // 3
        self.pelvis_joint = int(groups["pelvis"][0] // 3)
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32), persistent=False)
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32), persistent=False)
        self.in_proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.temporal = nn.Sequential(*[ResidualBlock(hidden_dim, 3, 2**i, dropout) for i in range(layers)])
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pelvis_head = nn.Linear(hidden_dim, 3)
        self.local_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        z = self.temporal(self.in_proj(x.transpose(1, 2)))[:, :, -1]
        h = self.shared(z)
        pelvis_z = self.pelvis_head(h)
        pelvis_cols = slice(self.pelvis_joint * 3, self.pelvis_joint * 3 + 3)
        pelvis_mm = pelvis_z * self.y_std[pelvis_cols] + self.y_mean[pelvis_cols]
        local_mm = self.local_head(h).view(h.shape[0], self.joint_count, 3) * 1000.0
        local_mm[:, self.pelvis_joint, :] = 0.0
        global_mm = pelvis_mm[:, None, :] + local_mm
        return (global_mm.reshape(h.shape[0], self.output_dim) - self.y_mean) / self.y_std


class BoneAwareLoss(nn.Module):
    def __init__(self, target_names, y_mean, y_std, bone_weight=0.0, local_weight=0.0):
        super().__init__()
        self.base = nn.SmoothL1Loss(beta=0.5)
        self.bone_weight = float(bone_weight)
        self.local_weight = float(local_weight)
        self.register_buffer("y_mean", torch.as_tensor(y_mean, dtype=torch.float32))
        self.register_buffer("y_std", torch.as_tensor(y_std, dtype=torch.float32))
        groups = joint_indices(target_names)
        pelvis = groups.get("pelvis")
        self.pelvis_idx = None if pelvis is None else torch.as_tensor(pelvis, dtype=torch.long)
        edges = skeleton_edges(target_names)
        self.edge_a = torch.as_tensor([a for a, _, _, _ in edges], dtype=torch.long)
        self.edge_b = torch.as_tensor([b for _, b, _, _ in edges], dtype=torch.long)

    def forward(self, pred_z, true_z):
        loss = self.base(pred_z, true_z)
        if self.bone_weight <= 0 and self.local_weight <= 0:
            return loss
        pred = pred_z * self.y_std + self.y_mean
        true = true_z * self.y_std + self.y_mean
        pred_j = pred.view(pred.shape[0], -1, 3)
        true_j = true.view(true.shape[0], -1, 3)
        if self.local_weight > 0 and self.pelvis_idx is not None:
            pelvis_joint = int(self.pelvis_idx[0].item() // 3)
            pred_local = (pred_j - pred_j[:, pelvis_joint : pelvis_joint + 1]) / 1000.0
            true_local = (true_j - true_j[:, pelvis_joint : pelvis_joint + 1]) / 1000.0
            loss = loss + self.local_weight * nn.functional.smooth_l1_loss(pred_local, true_local, beta=0.05)
        if self.bone_weight > 0 and len(self.edge_a) > 0:
            edge_a = self.edge_a.to(pred.device)
            edge_b = self.edge_b.to(pred.device)
            pred_len = torch.linalg.norm(pred[:, edge_a] - pred[:, edge_b], dim=-1) / 1000.0
            true_len = torch.linalg.norm(true[:, edge_a] - true[:, edge_b], dim=-1) / 1000.0
            loss = loss + self.bone_weight * nn.functional.smooth_l1_loss(pred_len, true_len, beta=0.02)
        return loss


def make_model(kind, input_dim, layout, output_dim, hidden_dim, layers, dropout, target_names=None, y_mean=None, y_std=None):
    if kind in {"raw_tcn", "raw_bone_tcn"}:
        return TCNRegressor(input_dim, output_dim, hidden_dim=hidden_dim, layers=layers, dropout=dropout)
    if kind in {"rootlocal_tcn", "rootlocal_bone_tcn"}:
        return RootLocalTCNRegressor(
            input_dim,
            output_dim,
            target_names,
            y_mean,
            y_std,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
        )
    use_tokens = kind in {"sensorformer", "sensorformer_bone", "moe_struct_bone_tcn"}
    if kind in {"hybrid_struct_tcn", "hybrid_struct_bone_tcn", "hybrid_moe_struct_bone_tcn"}:
        return HybridStructuredTCNRegressor(
            input_dim,
            layout["order"],
            len(layout["modules"]),
            len(layout["sensors"]),
            output_dim,
            hidden_dim=hidden_dim,
            layers=layers,
            dropout=dropout,
            use_token_transformer=False,
            experts=3 if kind == "hybrid_moe_struct_bone_tcn" else 1,
        )
    experts = 3 if kind == "moe_struct_bone_tcn" else 1
    return StructuredTCNRegressor(
        layout["order"],
        len(layout["modules"]),
        len(layout["sensors"]),
        output_dim,
        hidden_dim=hidden_dim,
        layers=layers,
        dropout=dropout,
        use_token_transformer=use_tokens,
        experts=experts,
    )


def evaluate(model, loader, device, loss_fn=None):
    model.eval()
    preds = []
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            if loss_fn is not None:
                losses.append(float(loss_fn(pred, yb).item()))
            preds.append(pred.detach().cpu().numpy())
    return np.vstack(preds), float(np.mean(losses)) if losses else 0.0


def parse_args():
    parser = argparse.ArgumentParser(description="Train physics/structure-aware PIR sequence models.")
    parser.add_argument("--dataset", default="PIR_gpu_mixed/model_output/baseline_ridge/aligned_dataset.npz")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=[
            "struct_tcn",
            "sensorformer",
            "struct_bone_tcn",
            "sensorformer_bone",
            "moe_struct_bone_tcn",
            "hybrid_struct_tcn",
            "hybrid_struct_bone_tcn",
            "hybrid_moe_struct_bone_tcn",
            "raw_tcn",
            "raw_bone_tcn",
            "rootlocal_tcn",
            "rootlocal_bone_tcn",
        ],
        default="struct_tcn",
    )
    parser.add_argument("--test-trials", default="rw_005,sit_008")
    parser.add_argument("--val-trials", default=None)
    parser.add_argument("--context-sec", type=float, default=4.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--layers", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bone-weight", type=float, default=0.0)
    parser.add_argument("--local-weight", type=float, default=0.0)
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
    layout = build_sensor_layout(feature_names)

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

    if "bone" in args.model and args.bone_weight == 0.0 and args.local_weight == 0.0:
        args.bone_weight = 0.2
        args.local_weight = 0.4

    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    model = make_model(
        args.model,
        x.shape[1],
        layout,
        y.shape[1],
        args.hidden_dim,
        args.layers,
        args.dropout,
        target_names=target_names,
        y_mean=y_mean,
        y_std=y_std,
    ).to(device)
    train_loader = DataLoader(
        SequenceDataset(x, y, train_idx, seq_len),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(SequenceDataset(x, y, val_idx, seq_len), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(SequenceDataset(x, y, test_idx, seq_len), batch_size=args.batch_size, shuffle=False)

    loss_fn = BoneAwareLoss(
        target_names,
        y_mean,
        y_std,
        bone_weight=args.bone_weight,
        local_weight=args.local_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)
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
        _, val_loss = evaluate(model, val_loader, device, loss_fn)
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
            "bone_weight": float(args.bone_weight),
            "local_weight": float(args.local_weight),
            "modules": layout["modules"],
            "sensors": layout["sensors"],
            "structured_feature_count": len(layout["feature_order"]),
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
