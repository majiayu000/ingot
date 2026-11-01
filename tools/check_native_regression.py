#!/usr/bin/env python3
"""Compare complete native logits on a saved token corpus, including sparse boundaries.

Use examples/validate binaries built before and after a change. This is an exact
native regression check, not an independent model-quality oracle. Requires NumPy.
"""
import argparse
import hashlib
import json
import pathlib
import subprocess
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ["baseline", "candidate", "model", "corpus", "output"]:
        p.add_argument("--" + key, type=pathlib.Path, required=True)
    p.add_argument("--prefix", choices=["qwen", "glm"], required=True)
    p.add_argument("--budget-gib", type=int, required=True)
    p.add_argument("--long", action="store_true", help="include 2112-token sparse attention boundary case")
    a = p.parse_args()
    corpus = json.loads(a.corpus.read_text())
    a.output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "Full-logit native before/after regression; not an independent quality oracle",
              "model": str(a.model), "budget_gib": a.budget_gib, "runs": [], "binaries": {}}
    for arm, binary in [("A", a.baseline), ("B", a.candidate)]:
        report["binaries"][arm] = {"path": str(binary.resolve()),
                                  "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    for name, case in corpus.items():
        if not name.startswith(a.prefix + "-") or (name.endswith("-long") and not a.long):
            continue
        tokens = a.corpus.parent / case["file"]
        count = tokens.stat().st_size // 4
        if count != case["tokens"] or tokens.stat().st_size % 4:
            raise ValueError("token corpus size changed")
        checkpoints = [1, 8, 64, 256, 512, 1024, 2047, 2048, 2052, 2055, count] if name.endswith("-long") else list(range(1, count + 1))
        context = 2112 if name.endswith("-long") else 256
        arrays = []
        row = {"case": name, "checkpoints": checkpoints, "context": context,
               "input_sha256": hashlib.sha256(tokens.read_bytes()).hexdigest(), "arms": []}
        report["runs"].append(row)
        for arm in ["A", "B"]:
            prefix = a.output / f"{name}-{arm}"
            logits = prefix.with_suffix(".f32")
            command = ["/usr/bin/time", "-l", report["binaries"][arm]["path"], str(a.model),
                       str(a.budget_gib), str(context), str(tokens), ",".join(map(str, checkpoints)), str(logits)]
            print(f"starting {name}-{arm}", flush=True)
            start = time.time()
            with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
                result = subprocess.run(command, stdout=out, stderr=err)
            row["arms"].append({"arm": arm, "command": command, "returncode": result.returncode,
                                "elapsed_seconds": time.time() - start})
            (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            result.check_returncode()
            values = np.fromfile(logits, dtype="<f4").reshape(len(checkpoints), -1)
            if values.shape[1] == 0 or not np.isfinite(values).all():
                raise ValueError(f"invalid logits at {name}-{arm}")
            arrays.append(values)
        if arrays[0].shape != arrays[1].shape:
            raise ValueError("logit shapes changed")
        row["vocabulary"] = arrays[0].shape[1]
        row["bitwise_identical"] = arrays[0].tobytes() == arrays[1].tobytes()
        row["max_abs_error"] = float(np.max(np.abs(arrays[0] - arrays[1])))
        row["top_token_matches"] = int(np.sum(arrays[0].argmax(1) == arrays[1].argmax(1)))
        (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        if not row["bitwise_identical"]:
            raise ValueError(f"native logits changed at {name}: max_abs={row['max_abs_error']}")
        if not name.endswith("-long"):
            grouped = [1, 2, 4, 8, count]
            prefix = a.output / f"{name}-B-grouped"
            logits = prefix.with_suffix(".f32")
            command = [report["binaries"]["B"]["path"], str(a.model), str(a.budget_gib),
                       str(context), str(tokens), ",".join(map(str, grouped)), str(logits)]
            with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
                result = subprocess.run(command, stdout=out, stderr=err)
            row["grouped"] = {"command": command, "returncode": result.returncode, "checkpoints": grouped}
            (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            result.check_returncode()
            actual = np.fromfile(logits, dtype="<f4").reshape(len(grouped), -1)
            expected = arrays[0][np.asarray(grouped) - 1]
            row["grouped"]["bitwise_identical"] = actual.tobytes() == expected.tobytes()
            (a.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            if not row["grouped"]["bitwise_identical"]:
                raise ValueError(f"grouped logits changed at {name}")
        print(f"passed {name}: {len(checkpoints)} complete logits identical", flush=True)


if __name__ == "__main__":
    main()
