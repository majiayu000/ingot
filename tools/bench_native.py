#!/usr/bin/env python3
"""Sequential paired benchmarks for the native examples/bench executable (macOS)."""
import argparse
import ast
import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import time


def seconds(value):
    match = re.fullmatch(r"([\d.]+)(ns|µs|ms|s)", value.strip())
    if not match:
        raise ValueError(f"invalid Rust duration: {value!r}")
    return float(match[1]) * {"ns": 1e-9, "µs": 1e-6, "ms": 1e-3, "s": 1}[match[2]]


def samples(stdout):
    lines = stdout.splitlines()
    if len(lines) != 4:
        raise ValueError("expected one retained warmup and one measured request")
    result = []
    for index in range(0, len(lines), 2):
        head = re.fullmatch(r"round=(\d+) stop=(\w+) token_ids=(\[.*\])", lines[index])
        if not head:
            raise ValueError("invalid benchmark output")
        stats = lines[index + 1]
        durations = re.search(r"decode: \[(.*?)\]", stats)[1]
        decode = [seconds(s) for s in durations.split(", ") if s]
        row = {"round": int(head[1]), "warmup": index == 0, "stop": head[2],
               "tokens": ast.literal_eval(head[3]), "decode_seconds": decode}
        for key in ["load", "prefill", "first_token", "total", "read_time"]:
            row[key + "_seconds"] = seconds(re.search(rf"\b{key}: ([^,}}]+)", stats)[1])
        for key in ["reserved_bytes", "metal_bytes", "file_bytes", "cache_hits", "cache_misses"]:
            row[key] = int(re.search(rf"\b{key}: (\d+)", stats)[1])
        result.append(row)
    return result


def summarize(rows):
    decode_count = sum(len(r["decode_seconds"]) for r in rows)
    decode_time = sum(sum(r["decode_seconds"]) for r in rows)
    result = {"requests": len(rows), "decode_steps": decode_count,
              "weighted_decode_tokens_per_second": decode_count / decode_time}
    for key in ["total_seconds", "prefill_seconds", "first_token_seconds"]:
        values = [r[key] for r in rows]
        result[key] = {"mean": statistics.mean(values), "median": statistics.median(values),
                       "min": min(values), "max": max(values), "sample_variance": statistics.variance(values)}
    rates = [len(r["decode_seconds"]) / sum(r["decode_seconds"]) for r in rows]
    result["decode_tokens_per_second"] = {
        "median": statistics.median(rates), "min": min(rates), "max": max(rates),
        "sample_variance": statistics.variance(rates)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["baseline", "candidate", "model", "output"]:
        parser.add_argument("--" + key, type=pathlib.Path, required=True)
    parser.add_argument("--budget-gib", type=int, required=True)
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--prompt", required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = {"order": ["A", "B", "B", "A", "A", "B"], "runs": [], "binaries": {},
              "conditions": ["Fresh Engine per process; one full warmup then one measured request",
                             "Sequential AB/BA/AB; all warmups and measured samples retained",
                             "OS file cache and other applications not controlled; no cold SSD claim"],
              "model": str(args.model), "budget_gib": args.budget_gib, "context": args.context,
              "max_tokens": args.max_tokens, "prompt": args.prompt}
    for arm, binary in [("A", args.baseline), ("B", args.candidate)]:
        report["binaries"][arm] = {"path": str(binary.resolve()),
                                  "sha256": hashlib.sha256(binary.read_bytes()).hexdigest()}
    reference = None
    for index, arm in enumerate(report["order"]):
        command = ["/usr/bin/time", "-l", report["binaries"][arm]["path"], str(args.model),
                   str(args.budget_gib), str(args.context), str(args.max_tokens), "1", args.prompt]
        prefix = args.output / f"{index}-{arm}"
        started = time.time()
        print(f"starting {prefix.name}", flush=True)
        with prefix.with_suffix(".stdout").open("w") as out, prefix.with_suffix(".stderr").open("w") as err:
            process = subprocess.run(command, stdout=out, stderr=err)
        run = {"arm": arm, "command": command, "started_unix": started,
               "elapsed_seconds": time.time() - started, "returncode": process.returncode}
        report["runs"].append(run)
        if process.returncode == 0:
            run["samples"] = samples(prefix.with_suffix(".stdout").read_text())
            run["peak_footprint_bytes"] = int(re.search(r"(\d+)\s+peak memory footprint", prefix.with_suffix(".stderr").read_text())[1])
            (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            for row in run["samples"]:
                output = (row["tokens"], row["stop"])
                if reference is None:
                    reference = output
                if output != reference:
                    raise ValueError(f"output changed at {prefix.name} round {row['round']}")
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        process.check_returncode()
        print(f"finished {prefix.name}: {run['elapsed_seconds']:.2f}s", flush=True)
    report["summary"] = {arm: summarize([r for run in report["runs"] if run["arm"] == arm
                                         for r in run["samples"] if not r["warmup"]]) for arm in ["A", "B"]}
    report["outputs_identical"] = True
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
