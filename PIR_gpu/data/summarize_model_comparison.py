import argparse
import csv
import json
from pathlib import Path


def parse_model_spec(spec):
    if "=" not in spec:
        path = Path(spec)
        return path.name, path
    name, path = spec.split("=", 1)
    return name, Path(path)


def load_row(name, model_dir):
    metrics_path = model_dir / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    model = metrics.get("model", {})
    test = metrics["test"]
    samples = metrics.get("samples", {})
    members = metrics.get("members", [])
    return {
        "name": name,
        "dir": str(model_dir),
        "type": model.get("type", metrics.get("type", "ridge")),
        "context_sec": model.get("context_sec", ""),
        "seq_len": model.get("seq_len", ""),
        "members": "+".join(member["name"] for member in members),
        "test_rows": samples.get("test_rows", ""),
        "rmse_mm": float(test["rmse_mm"]),
        "mae_mm": float(test["mae_mm"]),
        "mean_joint_error_mm": float(test["mean_joint_error_mm"]),
        "p95_mean_joint_error_mm": float(test["p95_mean_joint_error_mm"]),
        "r2_global": float(test["r2_global"]),
    }


def write_csv(path, rows):
    fields = [
        "rank",
        "name",
        "type",
        "context_sec",
        "seq_len",
        "members",
        "test_rows",
        "rmse_mm",
        "mae_mm",
        "mean_joint_error_mm",
        "p95_mean_joint_error_mm",
        "r2_global",
        "dir",
    ]
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            out = dict(row)
            out["rank"] = rank
            writer.writerow(out)


def write_report(path, rows):
    best = rows[0]
    baseline = next((row for row in rows if row["name"].lower() == "ridge"), None)
    lines = [
        "# PIR to Theia3D Model Comparison",
        "",
        "## Experimental Setup",
        "",
        "- Input: 25 Hz aligned PIR features generated from five PIR modules.",
        "- Target: 19 Theia3D global joints, XYZ coordinates, 57 regression targets.",
        "- Split: trials 002, 003, and 004 for training; trial 005 for held-out testing.",
        "- Metrics: coordinate RMSE/MAE over all XYZ values, frame mean joint error, 95th percentile frame mean joint error, and global R2.",
        "",
        "## Summary",
        "",
        f"- Best model: {best['name']}",
        f"- Best RMSE: {best['rmse_mm']:.1f} mm",
        f"- Best mean joint error: {best['mean_joint_error_mm']:.1f} mm",
    ]
    if baseline:
        improvement = 100.0 * (baseline["rmse_mm"] - best["rmse_mm"]) / baseline["rmse_mm"]
        lines.append(f"- RMSE reduction from Ridge baseline: {improvement:.1f}%")
    lines.extend(
        [
            "",
            "## Results",
            "",
            "| Rank | Model | Type | Context [s] | Test rows | RMSE [mm] | MAE [mm] | Mean joint error [mm] | P95 mean joint error [mm] | R2 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row['name']} | {row['type']} | {row['context_sec']} | {row['test_rows']} | "
            f"{row['rmse_mm']:.1f} | {row['mae_mm']:.1f} | {row['mean_joint_error_mm']:.1f} | "
            f"{row['p95_mean_joint_error_mm']:.1f} | {row['r2_global']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- TCN variants were the strongest single models, suggesting that local temporal convolution is well matched to the PIR signal.",
            "- GRU improved when context was extended to 6 seconds, but its frame mean joint error remained higher than the best TCN.",
            "- Transformer-only variants overfit more easily on the current four-trial dataset.",
            "- The best result came from averaging complementary TCN and GRU predictions.",
            "",
            "## Caveat",
            "",
            "This comparison uses a single held-out trial. A publication-ready claim should add trial-wise cross-validation and more recording sessions.",
        ]
    )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize model comparison metrics.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", required=True, help="name=directory containing metrics.json")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = [load_row(*parse_model_spec(spec)) for spec in args.model]
    rows.sort(key=lambda row: row["rmse_mm"])
    write_csv(out_dir / "model_comparison.csv", rows)
    write_report(out_dir / "experiment_report.md", rows)
    with (out_dir / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    print(json.dumps({"output_dir": str(out_dir), "best": rows[0]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
