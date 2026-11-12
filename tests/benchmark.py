"""Standalone benchmark for RadGuard over the synthetic radiology cases.

Treats ``expected_safe`` as ground truth for ``is_safe``. Precision, recall, and
F1 are reported on the unsafe class (``expected_safe=false``). Latency is the
mean / p99 of repeated engine evaluations.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.engine import evaluate  # noqa: E402

CASES_PATH = ROOT / "data" / "synthetic_cases.json"
LATENCY_REPEATS = 100


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (q in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def f1_score(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def main() -> None:
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    true_positive = 0
    false_positive = 0
    false_negative = 0
    latencies: list[float] = []
    fp_ids: list[str] = []
    fn_ids: list[str] = []

    for case in cases:
        result = evaluate(case["findings"], case["impression"])
        predicted_unsafe = not result.is_safe
        actual_unsafe = not case["expected_safe"]
        if predicted_unsafe and actual_unsafe:
            true_positive += 1
        elif predicted_unsafe and not actual_unsafe:
            false_positive += 1
            fp_ids.append(str(case["id"]))
        elif not predicted_unsafe and actual_unsafe:
            false_negative += 1
            fn_ids.append(str(case["id"]))

        for _ in range(LATENCY_REPEATS):
            timed = evaluate(case["findings"], case["impression"])
            latencies.append(timed.latency_ms)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = f1_score(precision, recall)
    mean_latency = sum(latencies) / len(latencies) if latencies else 0.0
    p99_latency = percentile(latencies, 99)

    rows = [
        ("Total Cases", str(len(cases))),
        ("Precision", f"{precision * 100:.1f}%"),
        ("Recall", f"{recall * 100:.1f}%"),
        ("F1-Score", f"{f1:.2f}"),
        ("Mean Latency (ms)", f"{mean_latency:.3f}"),
        ("P99 Latency (ms)", f"{p99_latency:.3f}"),
    ]
    name_width = max(len(name) for name, _ in rows)
    value_width = max(len(value) for _, value in rows)
    divider = f"|{'-' * (name_width + 2)}|{'-' * (value_width + 2)}|"
    print(f"| {'Metric'.ljust(name_width)} | {'Result'.ljust(value_width)} |")
    print(divider)
    for name, value in rows:
        print(f"| {name.ljust(name_width)} | {value.ljust(value_width)} |")

    if fp_ids or fn_ids:
        print()
        if fp_ids:
            print("False positives (safe labeled unsafe): " + ", ".join(fp_ids))
        if fn_ids:
            print("False negatives (unsafe labeled safe): " + ", ".join(fn_ids))
    else:
        print()
        print("No false positives or false negatives on this labeled set.")


if __name__ == "__main__":
    main()
