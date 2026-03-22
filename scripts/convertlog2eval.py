#!/usr/bin/env python3
import argparse
import csv
import json
import os
from typing import Dict, List, Optional, Tuple


PREFER_MAX_KEYWORDS = (
    "psnr",
    "ssim",
    "acc",
    "accuracy",
    "iou",
    "f1",
    "precision",
    "recall",
)

NON_METRIC_COLUMNS = {"epoch", "split", "checkpoint"}


def parse_float(value: str) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_int(value: str) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def prefers_max(metric_name: str) -> bool:
    name = metric_name.lower()
    return any(keyword in name for keyword in PREFER_MAX_KEYWORDS)


def load_eval_rows(csv_path: str) -> Tuple[List[Dict], List[str]]:
    with open(csv_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = reader.fieldnames or []
        rows: List[Dict] = []
        for raw in reader:
            row = {
                "epoch": parse_int(raw.get("epoch", "")),
                "split": raw.get("split", "eval"),
                "checkpoint": raw.get("checkpoint", ""),
            }
            for field in fieldnames:
                if field in NON_METRIC_COLUMNS:
                    continue
                row[field] = parse_float(raw.get(field, ""))
            rows.append(row)
    return rows, fieldnames


def select_best_for_metric(rows: List[Dict], metric_name: str) -> Optional[Dict]:
    candidates = [row for row in rows if row.get(metric_name) is not None and row.get("checkpoint")]
    if not candidates:
        return None

    if prefers_max(metric_name):
        best = max(candidates, key=lambda row: row[metric_name])
        mode = "max"
    else:
        best = min(candidates, key=lambda row: row[metric_name])
        mode = "min"

    return {
        "metric": metric_name,
        "mode": mode,
        "value": best[metric_name],
        "epoch": best.get("epoch"),
        "checkpoint": best.get("checkpoint"),
    }


def select_balanced(rows: List[Dict], metric_names: List[str]) -> Optional[Dict]:
    metric_stats = {}
    for metric_name in metric_names:
        values = [row[metric_name] for row in rows if row.get(metric_name) is not None]
        if not values:
            continue
        metric_stats[metric_name] = {
            "min": min(values),
            "max": max(values),
            "prefers_max": prefers_max(metric_name),
        }

    if not metric_stats:
        return None

    best_row = None
    best_score = -1.0

    for row in rows:
        score_sum = 0.0
        score_count = 0
        for metric_name, stats in metric_stats.items():
            value = row.get(metric_name)
            if value is None:
                continue

            min_v = stats["min"]
            max_v = stats["max"]
            if max_v == min_v:
                normalized = 1.0
            elif stats["prefers_max"]:
                normalized = (value - min_v) / (max_v - min_v)
            else:
                normalized = (max_v - value) / (max_v - min_v)

            score_sum += normalized
            score_count += 1

        if score_count == 0:
            continue

        score = score_sum / score_count
        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        return None

    return {
        "score": best_score,
        "epoch": best_row.get("epoch"),
        "checkpoint": best_row.get("checkpoint"),
        "metric_count": len(metric_stats),
    }


def run_conversion(log_path: str, csv_path: Optional[str] = None, output_json: Optional[str] = None) -> Dict:
    if not log_path.endswith(".txt"):
        raise ValueError(f"log path must end with .txt, got: {log_path}")
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"log file not found: {log_path}")

    workspace = os.path.dirname(os.path.abspath(log_path))
    resolved_csv = csv_path if csv_path else os.path.join(workspace, "eval_metrics.csv")
    if not os.path.exists(resolved_csv):
        raise FileNotFoundError(f"eval csv not found: {resolved_csv}")

    rows, fieldnames = load_eval_rows(resolved_csv)
    if not rows:
        raise ValueError(f"eval csv has no rows: {resolved_csv}")

    # Prefer validation rows for model selection if available.
    val_rows = [row for row in rows if str(row.get("split", "")).lower() == "val"]
    selected_rows = val_rows if val_rows else rows
    selected_split = "val" if val_rows else "all"

    metric_names = [field for field in fieldnames if field not in NON_METRIC_COLUMNS]
    metric_names = [name for name in metric_names if any(row.get(name) is not None for row in selected_rows)]

    best_per_metric = {}
    for metric_name in metric_names:
        best_info = select_best_for_metric(selected_rows, metric_name)
        if best_info is not None:
            best_per_metric[metric_name] = best_info

    balanced = select_balanced(selected_rows, metric_names)

    summary = {
        "log_path": os.path.abspath(log_path),
        "csv_path": os.path.abspath(resolved_csv),
        "selected_split": selected_split,
        "best_per_metric": best_per_metric,
        "balanced": balanced,
    }

    if output_json is None:
        output_json = os.path.join(workspace, "eval_selection.json")
    with open(output_json, "w") as output_file:
        json.dump(summary, output_file, indent=2)

    output_txt = os.path.join(workspace, "eval_selection.txt")
    with open(output_txt, "w") as output_file:
        output_file.write(f"log: {summary['log_path']}\n")
        output_file.write(f"csv: {summary['csv_path']}\n")
        output_file.write(f"split: {summary['selected_split']}\n")
        output_file.write("\n")
        output_file.write("Best checkpoint per metric:\n")
        for metric_name, item in best_per_metric.items():
            output_file.write(
                f"- {metric_name} ({item['mode']}): epoch={item['epoch']} value={item['value']:.6f} ckpt={item['checkpoint']}\n"
            )
        if balanced is not None:
            output_file.write("\n")
            output_file.write(
                f"Balanced checkpoint: epoch={balanced['epoch']} score={balanced['score']:.6f} ckpt={balanced['checkpoint']}\n"
            )

    print(f"[convertlog2eval] split={selected_split}")
    for metric_name, item in best_per_metric.items():
        print(
            f"[convertlog2eval] best {metric_name} ({item['mode']}): "
            f"epoch={item['epoch']}, value={item['value']:.6f}, ckpt={item['checkpoint']}"
        )
    if balanced is not None:
        print(
            f"[convertlog2eval] balanced: epoch={balanced['epoch']}, "
            f"score={balanced['score']:.6f}, ckpt={balanced['checkpoint']}"
        )
    print(f"[convertlog2eval] wrote: {output_json}")
    print(f"[convertlog2eval] wrote: {output_txt}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert training log/eval csv to best checkpoint selection."
    )
    parser.add_argument("log_path", type=str, help="Path to training log file (*.txt).")
    parser.add_argument(
        "--csv",
        dest="csv_path",
        type=str,
        default=None,
        help="Path to eval csv. Default: <log_dir>/eval_metrics.csv",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional output json path. Default: <log_dir>/eval_selection.json",
    )
    args = parser.parse_args()
    run_conversion(args.log_path, csv_path=args.csv_path, output_json=args.output_json)


if __name__ == "__main__":
    main()
