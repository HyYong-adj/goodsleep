"""Phase 2 결과를 markdown 으로 집계한다. 판정은 C0 대조군 대비로 한다."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASSES = ("Wake", "REM", "Light", "Deep")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    payload = json.loads((args.run_dir / "phase2.json").read_text())
    records = {r["variant"]: r for r in payload["records"]}
    control = records.get("C0")

    print(f"# Phase 2 결과 — encoder 단계 combo (night_norm, SpecAugment)\n")
    print(f"실행: `{args.run_dir.name}`, 상태 **{payload['status']}**, "
          f"device {payload.get('device')}, seed {payload.get('seeds')}\n")
    print("Validation-only, 단일 seed screening. `test_locked` 미접근.\n")

    print("## 전체\n")
    print("| Variant | 변경 | C1 (단일 epoch) | B1 (downstream) | B1 Δ vs C0 | κ | Acc |")
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for name in ("C0", "C2", "C3", "C4"):
        r = records.get(name)
        if not r:
            continue
        delta = "기준" if name == "C0" else (f"{r['b1_macro_f1'] - control['b1_macro_f1']:+.4f}" if control else "—")
        print(f"| {name} | {r['variant_label']} | {r['c1_macro_f1']:.4f} | **{r['b1_macro_f1']:.4f}** | "
              f"{delta} | {r['b1_cohen_kappa']:.4f} | {r['b1_accuracy']:.4f} |")

    print("\n## 클래스별 B1 F1\n")
    print("| Variant | " + " | ".join(CLASSES) + " | REM recall | Deep recall |")
    print("| --- | " + " | ".join("---:" for _ in CLASSES) + " | ---: | ---: |")
    for name in ("C0", "C2", "C3", "C4"):
        r = records.get(name)
        if not r:
            continue
        cells = " | ".join(f"{r['b1_per_class_f1'][c]:.4f}" for c in CLASSES)
        print(f"| {name} | {cells} | {r['b1_per_class_recall']['REM']:.4f} | {r['b1_per_class_recall']['Deep']:.4f} |")

    print("\n## 판정\n")
    print("기준: C4 의 C0 대비 Δ ≥ +0.01 **이면서** REM·Deep F1 미하락. 단일 seed 이므로 채택 시 3 seed 확장이 선행 조건.\n")
    if control:
        for name in ("C2", "C3", "C4"):
            r = records.get(name)
            if not r:
                continue
            delta = r["b1_macro_f1"] - control["b1_macro_f1"]
            rem = r["b1_per_class_f1"]["REM"] - control["b1_per_class_f1"]["REM"]
            deep = r["b1_per_class_f1"]["Deep"] - control["b1_per_class_f1"]["Deep"]
            verdict = "임계 충족" if (delta >= 0.01 and rem >= 0 and deep >= 0) else "임계 미달"
            print(f"- **{name}** ({r['variant_label']}): ΔMacro-F1 {delta:+.4f}, ΔREM {rem:+.4f}, ΔDeep {deep:+.4f} → {verdict}")
    print("\n## 비용\n")
    print("| Variant | C1 학습 | 전체 |")
    print("| --- | ---: | ---: |")
    for name in ("C0", "C2", "C3", "C4"):
        r = records.get(name)
        if r:
            print(f"| {name} | {r['c1_train_seconds']/60:.0f} min | {r['wall_seconds']/60:.0f} min |")
    print("\n> C0 는 같은 GPU 대조군이다. parent(GPU 3) B1 은 0.483261 이었으나 C0 는 다른 값이 나온다 — "
          "GPU 간 비트 재현이 성립하지 않기 때문이며, 따라서 **판정은 parent 가 아니라 C0 대비**로 한다.")


if __name__ == "__main__":
    main()
