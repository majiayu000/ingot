#!/usr/bin/env python3
"""Summarize frozen teacher-forced logits and independently graded JSON tasks."""
import json
import math
from pathlib import Path
import sys

import numpy as np


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith('{"id":')]


def summarize(root):
    plan = json.loads((root / "evaluation-freeze.json").read_text())
    limits = plan["thresholds"]
    metadata = {arm: {x["id"]: x for x in records(root / (arm + "-teacher.txt"))}
                for arm in ("A", "B")}
    values = []
    documents = []
    for item in plan["inputs"]:
        ident = item["id"]
        a, b = metadata["A"][ident], metadata["B"][ident]
        if a != b or a["return"] != 0:
            raise ValueError("Incomplete or mismatched teacher-forced record: " + ident)
        loss = {arm: np.fromfile(root / arm / (ident + ".loss.f64"), dtype="<f8")
                for arm in ("A", "B")}
        samples = {arm: np.fromfile(root / arm / (ident + ".logits.f32"), dtype="<f4")
                   for arm in ("A", "B")}
        if any(x.size != a["eligible"] or not np.isfinite(x).all() for x in loss.values()):
            raise ValueError("Invalid complete next-token losses: " + ident)
        if any(x.size != a["sampled"] * a["vocab"] or not np.isfinite(x).all()
               for x in samples.values()):
            raise ValueError("Invalid full-vocabulary samples: " + ident)
        x, y = (samples[arm].reshape(a["sampled"], a["vocab"])
                for arm in ("A", "B"))
        documents.append({"id": ident, "category": item["category"],
                          "positions": a["eligible"], "samples": a["sampled"],
                          "A_loss_sum": float(loss["A"].sum()),
                          "B_loss_sum": float(loss["B"].sum())})
        for left, right in zip(x, y):
            left, right = left.astype(np.float64), right.astype(np.float64)
            lp = left - left.max()
            lq = right - right.max()
            lp -= np.log(np.exp(lp).sum())
            lq -= np.log(np.exp(lq).sum())
            ka, kb = (np.argpartition(v, -32)[-32:] for v in (left, right))
            ka = ka[np.lexsort((ka, -left[ka]))]
            kb = kb[np.lexsort((kb, -right[kb]))]
            values.append({"category": item["category"],
                           "kl": float(np.dot(np.exp(lp), lp - lq)),
                           "top1": bool(left.argmax() == right.argmax()),
                           "top32_overlap": len(set(ka) & set(kb)) / 32,
                           "top32_order_identical": bool(np.array_equal(ka, kb))})
    groups = {}
    for category in ["chinese", "code", "math", "overall"]:
        d = [v for v in documents if category == "overall" or v["category"] == category]
        v = [v for v in values if category == "overall" or v["category"] == category]
        count = sum(x["positions"] for x in d)
        nll_a = sum(x["A_loss_sum"] for x in d) / count
        nll_b = sum(x["B_loss_sum"] for x in d) / count
        result = {"positions": count, "sampled_positions": len(v),
                  "A_perplexity": math.exp(nll_a), "B_perplexity": math.exp(nll_b),
                  "ppl_ratio": math.exp(nll_b - nll_a),
                  "kl_mean": float(np.mean([x["kl"] for x in v])),
                  "kl_p95": float(np.percentile([x["kl"] for x in v], 95)),
                  "top1_agreement": float(np.mean([x["top1"] for x in v])),
                  "top32_overlap": float(np.mean([x["top32_overlap"] for x in v])),
                  "top32_order_agreement": float(np.mean([x["top32_order_identical"] for x in v]))}
        result["numerical_gate_passed"] = (
            result["ppl_ratio"] <= limits["ppl_ratio_max"]
            and result["kl_mean"] <= limits["kl_mean_max"]
            and result["kl_p95"] <= limits["kl_p95_max"]
            and result["top1_agreement"] >= limits["top1_min"]
            and result["top32_overlap"] >= limits["top32_overlap_min"])
        groups[category] = result
    task_results = {}
    for arm in ("A", "B"):
        path = root / (arm + "-tasks.json")
        if path.exists():
            data = json.loads(path.read_text())
            task_results[arm] = {category: sum(x["correct"] for x in data["cases"]
                                             if x["category"] == category)
                                 for category in ("chinese", "code", "math")}
    tasks_complete = all((root / (arm + "-tasks.json")).exists()
                         and json.loads((root / (arm + "-tasks.json")).read_text())["completed"]
                         for arm in ("A", "B"))
    tasks_passed = (tasks_complete and all(task_results["B"][c] >= task_results["A"][c]
                                         for c in ("chinese", "code", "math")))
    return {"reference": plan["reference"], "groups": groups, "documents": documents,
            "task_correct_counts": task_results, "tasks_complete": tasks_complete,
            "task_gate_passed": tasks_passed,
            "numerical_gate_passed": all(x["numerical_gate_passed"] for x in groups.values()),
            "adoption_ready": False,
            "remaining_adoption_requirements": ["independent_operator_reference", "error_and_API_checks",
                                                "existing_functional_checks", "paired_complete_request_speed"],
            "scope": "Six fixed teacher-forced documents plus fixed tasks; no broad quality guarantee."}


if __name__ == "__main__":
    destination = Path(sys.argv[1])
    report = summarize(destination)
    (destination / "quality-summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"groups": report["groups"], "tasks": report["task_correct_counts"],
                      "numerical_gate_passed": report["numerical_gate_passed"],
                      "task_gate_passed": report["task_gate_passed"]}, indent=2))
