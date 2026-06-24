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
        "train_trials": metrics.get("train_trials", []),
        "test_trials": metrics.get("test_trials", []),
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
        "train_trials",
        "test_trials",
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
            out["train_trials"] = "+".join(out.get("train_trials", []))
            out["test_trials"] = "+".join(out.get("test_trials", []))
            writer.writerow(out)


def write_report(path, rows):
    best = rows[0]
    baseline = next((row for row in rows if row["name"].lower() == "ridge"), None)
    reference = baseline or best
    train_trials = ", ".join(f"trial {trial}" for trial in reference.get("train_trials", []))
    test_trials = ", ".join(f"trial {trial}" for trial in reference.get("test_trials", []))
    if not train_trials:
        train_trials = "metrics.jsonに記録された学習trial"
    if not test_trials:
        test_trials = "metrics.jsonに記録された未使用テストtrial"
    test_trial_count = len(reference.get("test_trials", []))
    if test_trial_count > 1:
        limitation_text = (
            f"この比較は{test_trial_count}本の未使用テストtrialに基づく。"
            "論文として強い主張を行うには、trial単位の交差検証と、別日・別条件での追加収録が必要である。"
        )
    else:
        limitation_text = (
            "この比較は単一の未使用テストtrialに基づく。"
            "論文として強い主張を行うには、trial単位の交差検証と、別日・別条件での追加収録が必要である。"
        )
    type_labels = {
        "prediction_average_ensemble": "予測平均アンサンブル",
        "ridge_regression": "Ridge回帰",
        "tcn": "TCN",
        "gru": "GRU",
        "mlp": "MLP",
        "transformer": "Transformer",
        "convtransformer": "ConvTransformer",
    }
    lines = [
        "# PIRからTheia3D骨格座標を推定するモデル比較",
        "",
        "## 実験設定",
        "",
        "- 入力: 5台のPIRモジュールから作成した25Hz整列済みPIR特徴量。",
        "- 教師: Theia3Dのglobal joint座標。19関節のXYZ、合計57次元を予測対象とした。",
        f"- 分割: {train_trials}を学習に使用し、{test_trials}を未使用テストに使用した。",
        "- 評価指標: 全XYZ座標のRMSE/MAE、フレームごとの平均関節誤差、その95パーセンタイル、global R2。",
        "",
        "## 概要",
        "",
        f"- 最良モデル: {best['name']}",
        f"- 最良RMSE: {best['rmse_mm']:.1f} mm",
        f"- 最良平均関節誤差: {best['mean_joint_error_mm']:.1f} mm",
    ]
    if baseline:
        improvement = 100.0 * (baseline["rmse_mm"] - best["rmse_mm"]) / baseline["rmse_mm"]
        lines.append(f"- RidgeベースラインからのRMSE削減率: {improvement:.1f}%")
    lines.extend(
        [
            "",
            "## 結果",
            "",
            "| 順位 | モデル | 種類 | 文脈長 [s] | テスト行数 | RMSE [mm] | MAE [mm] | 平均関節誤差 [mm] | P95平均関節誤差 [mm] | R2 |",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for rank, row in enumerate(rows, start=1):
        type_label = type_labels.get(row["type"], row["type"])
        lines.append(
            f"| {rank} | {row['name']} | {type_label} | {row['context_sec']} | {row['test_rows']} | "
            f"{row['rmse_mm']:.1f} | {row['mae_mm']:.1f} | {row['mean_joint_error_mm']:.1f} | "
            f"{row['p95_mean_joint_error_mm']:.1f} | {row['r2_global']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 解釈",
            "",
            "- 単体モデルではTCN系が最も強く、PIR信号には局所的な時間畳み込みがよく合っていると考えられる。",
            "- RNN系、MLP系、Transformer系も一定の性能を示したが、最良単体モデルには届かなかった。",
            "- 最良結果は、上位モデルの予測を平均することで得られた。誤差傾向の違いが相補的に働いた可能性がある。",
            "",
            "## 注意点",
            "",
            limitation_text,
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
