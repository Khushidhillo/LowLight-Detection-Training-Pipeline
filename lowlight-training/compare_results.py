"""Compare multiple evaluation metrics.json files with baseline deltas and per-class breakdowns."""

import argparse
import csv
import json
from pathlib import Path


def compare(result_paths, out_csv, per_class_csv=None):
    rows = []
    per_class_rows = []
    baseline_map50 = None
    baseline_map = None

    for idx, path in enumerate(result_paths):
        p = Path(path)
        d = json.loads(p.read_text(encoding="utf-8"))
        map50 = d.get("metrics/mAP50(B)")
        map50_95 = d.get("metrics/mAP50-95(B)")
        if idx == 0:
            baseline_map50 = map50
            baseline_map = map50_95

        row = {"run": str(p.parent)}
        for k, v in d.items():
            if isinstance(v, (float, int)):
                row[k] = v

        if map50 is not None and baseline_map50 is not None:
            row["delta_mAP50"] = round(float(map50) - float(baseline_map50), 6)
        if map50_95 is not None and baseline_map is not None:
            row["delta_mAP50_95"] = round(float(map50_95) - float(baseline_map), 6)

        speed = d.get("detector_speed_ms")
        if isinstance(speed, dict) and "inference" in speed:
            row["detector_inference_ms"] = round(float(speed["inference"]), 3)

        rows.append(row)

        per_cls = d.get("per_class_mAP50_95")
        if isinstance(per_cls, dict):
            per_class_rows.append({"run": str(p.parent), **per_cls})

    keys = list(dict.fromkeys(k for row in rows for k in row))
    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    if per_class_csv and per_class_rows:
        pc_keys = list(dict.fromkeys(k for row in per_class_rows for k in row))
        pc_out = Path(per_class_csv)
        pc_out.parent.mkdir(parents=True, exist_ok=True)
        with pc_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pc_keys)
            writer.writeheader()
            writer.writerows(per_class_rows)

    # Print formatted summary table
    print("\n| Run | Precision | Recall | mAP50 | Δ mAP50 | mAP50-95 | Δ mAP50-95 | Inf (ms) |")
    print("|-----|-----------|--------|-------|---------|----------|------------|----------|")
    for r in rows:
        prec = f"{r['metrics/precision(B)']:.4f}" if "metrics/precision(B)" in r else "-"
        rec = f"{r['metrics/recall(B)']:.4f}" if "metrics/recall(B)" in r else "-"
        m50 = f"{r['metrics/mAP50(B)']:.4f}" if "metrics/mAP50(B)" in r else "-"
        dm50 = f"{r['delta_mAP50']:+.4f}" if "delta_mAP50" in r else "-"
        m95 = f"{r['metrics/mAP50-95(B)']:.4f}" if "metrics/mAP50-95(B)" in r else "-"
        dm95 = f"{r['delta_mAP50_95']:+.4f}" if "delta_mAP50_95" in r else "-"
        inf = f"{r['detector_inference_ms']:.2f}" if "detector_inference_ms" in r else "-"
        print(f"| {r['run']} | {prec} | {rec} | {m50} | {dm50} | {m95} | {dm95} | {inf} |")

    print(f"\nSaved: {out.resolve()}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Compare evaluation metrics.json files")
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--per-class-out", default=None, help="Optional CSV path for per-class mAP50-95")
    a = p.parse_args()
    compare(a.results, a.out, a.per_class_out)


if __name__ == "__main__":
    main()
