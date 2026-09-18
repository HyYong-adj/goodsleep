"""Phase 1 결과 집계: variant x seed 표와 A0 대비 seed-paired 차이.

단일 seed 절대값이 아니라 **같은 seed 끼리의 차이**만 판단 근거로 쓴다.
참조 트랙 실측상 best-epoch 선택 부풀림이 +0.049 이므로 절대값은 낙관적이다.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

CLASSES = ("Wake", "REM", "Light", "Deep")


def load(run_dir: Path) -> list[dict]:
    payload = json.loads((run_dir / "phase1.json").read_text())
    if payload["status"] != "COMPLETE":
        print(f"# WARNING: status={payload['status']} (부분 결과)")
    return payload["records"]


def by_variant(records: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for record in records:
        out.setdefault(record["variant"], {})[record["seed"]] = record
    return out


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--baseline", default="A0")
    args = parser.parse_args()

    records = load(args.run_dir)
    grouped = by_variant(records)
    base = grouped.get(args.baseline, {})

    print("## Macro-F1 (variant x seed)\n")
    seeds = sorted({r["seed"] for r in records})
    header = "| Variant | 설명 | " + " | ".join(f"s{s}" for s in seeds) + " | 평균 | 표준편차 |"
    print(header)
    print("| --- | --- | " + " | ".join("---:" for _ in seeds) + " | ---: | ---: |")
    for variant in sorted(grouped):
        runs = grouped[variant]
        values = [runs[s]["macro_f1"] for s in seeds if s in runs]
        label = next(iter(runs.values()))["variant_label"]
        sd = f"{statistics.stdev(values):.4f}" if len(values) > 1 else "—"
        cells = " | ".join(fmt(runs[s]["macro_f1"]) if s in runs else "—" for s in seeds)
        print(f"| {variant} | {label} | {cells} | {statistics.fmean(values):.4f} | {sd} |")

    print(f"\n## {args.baseline} 대비 seed-paired 차이 (Macro-F1)\n")
    print("| Variant | " + " | ".join(f"s{s}" for s in seeds) + " | 평균 Δ | 전 seed 동일 부호 |")
    print("| --- | " + " | ".join("---:" for _ in seeds) + " | ---: | :---: |")
    for variant in sorted(grouped):
        if variant == args.baseline:
            continue
        deltas = [grouped[variant][s]["macro_f1"] - base[s]["macro_f1"] for s in seeds if s in grouped[variant] and s in base]
        if not deltas:
            continue
        same = "예" if all(d > 0 for d in deltas) or all(d < 0 for d in deltas) else "아니오"
        cells = " | ".join(f"{d:+.4f}" for d in deltas)
        print(f"| {variant} | {cells} | {statistics.fmean(deltas):+.4f} | {same} |")

    print("\n## 클래스별 F1 (seed 평균)\n")
    print("| Variant | " + " | ".join(CLASSES) + " | κ | Acc |")
    print("| --- | " + " | ".join("---:" for _ in CLASSES) + " | ---: | ---: |")
    for variant in sorted(grouped):
        runs = list(grouped[variant].values())
        cells = " | ".join(fmt(statistics.fmean([r["per_class_f1"][c] for c in [c] for r in runs])) for c in CLASSES)
        kappa = statistics.fmean([r["cohen_kappa"] for r in runs])
        acc = statistics.fmean([r["accuracy"] for r in runs])
        print(f"| {variant} | {cells} | {kappa:.4f} | {acc:.4f} |")

    print("\n## 선택 부풀림 (best epoch − 인접 epoch 평균)\n")
    print("| Variant | 평균 부풀림 | 부풀림 보정 Macro-F1 |")
    print("| --- | ---: | ---: |")
    for variant in sorted(grouped):
        runs = list(grouped[variant].values())
        infl = [r["selection_inflation"]["inflation"] for r in runs if r.get("selection_inflation", {}).get("inflation") is not None]
        if not infl:
            continue
        mean_infl = statistics.fmean(infl)
        mean_f1 = statistics.fmean([r["macro_f1"] for r in runs])
        print(f"| {variant} | {mean_infl:+.4f} | {mean_f1 - mean_infl:.4f} |")

    print("\n## 과적합 지표 (best epoch 위치)\n")
    print("| Variant | best epoch (seed별) |")
    print("| --- | --- |")
    for variant in sorted(grouped):
        runs = grouped[variant]
        cells = ", ".join(str(runs[s]["selection_inflation"].get("best_epoch")) for s in seeds if s in runs)
        print(f"| {variant} | {cells} |")


if __name__ == "__main__":
    main()
