#!/usr/bin/env python3
"""Independent real-weight oracle: ggml C dequantization + NumPy, not Rust math.

The reference library is only loaded into this validation process. The Rust binary
receives GGUF and raw F32 files and never loads ggml or llama.
"""
import argparse
import ctypes
import hashlib
import json
import pathlib
import subprocess
import tempfile

import numpy as np

from gguf_reader import GGUFFile


class Reference:
    def __init__(self, path, library, layer):
        self.model = GGUFFile(path)
        self.tensors = {t.name: t for t in self.model.tensors}
        self.file = path.open("rb")
        self.library = ctypes.CDLL(str(library.resolve()))
        self.layer = layer
        self.formats = set()

    def matrix(self, role, expert=None):
        t = self.tensors[f"blk.{self.layer}.ffn_{role}.weight"]
        self.formats.add(t.type_name)
        cols = t.shape[0]
        rows = t.shape[1] if len(t.shape) > 1 else 1
        n = cols * rows
        size = t.expected_bytes()
        if size is None:
            raise ValueError(f"unknown tensor geometry: {t}")
        if expert is not None:
            size //= t.shape[2]
        self.file.seek(self.model.data_start + t.offset + (expert or 0) * size)
        raw = self.file.read(size)
        if len(raw) != size:
            raise ValueError("truncated weight tensor")
        if t.type_name == "BF16":
            values = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            fn = getattr(self.library, f"dequantize_row_{t.type_name.lower()}")
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
            fn.restype = None
            values = np.empty(n, np.float32)
            source = ctypes.create_string_buffer(raw)
            fn(source, values.ctypes.data, n)
        return values.reshape(rows, cols)

    def ffn(self, x, shared=False, expert=None):
        suffix = "shexp" if shared else "exps"
        gate = self.matrix(f"gate_{suffix}", expert) @ x
        up = self.matrix(f"up_{suffix}", expert) @ x
        # Double exp avoids overflowing the oracle on large finite gate values.
        act = (gate / (1.0 + np.exp(-gate.astype(np.float64)))).astype(np.float32) * up
        return self.matrix(f"down_{suffix}", expert) @ act

    def run(self, x):
        logits = self.matrix("gate_inp") @ x
        top = self.model.metadata["qwen4exp.expert_used_count"]
        ids = np.argsort(-logits, kind="stable")[:top]
        # Match the reference graph's full softmax, gather, then renormalize.
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        weights = probs[ids] / probs[ids].sum()
        routed = sum((self.ffn(x, expert=int(e)) * w for e, w in zip(ids, weights)),
                     np.zeros_like(x))
        shared_gate = (self.matrix("gate_inp_shexp") @ x).item()
        shared_weight = 1.0 / (1.0 + np.exp(-shared_gate))
        return routed + self.ffn(x, shared=True) * shared_weight, ids, weights


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=pathlib.Path, required=True)
    ap.add_argument("--ggml", type=pathlib.Path, required=True)
    ap.add_argument("--binary", type=pathlib.Path, default=pathlib.Path("target/release/ingot"))
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--report", type=pathlib.Path, required=True)
    args = ap.parse_args()
    reference = Reference(args.model, args.ggml, args.layer)
    width = reference.model.metadata["qwen4exp.embedding_length"]
    rng = np.random.default_rng(20261011)
    cases = [("normal", rng.normal(size=width).astype(np.float32)),
             ("small", (rng.normal(size=width) * 0.01).astype(np.float32)),
             ("zero_ties", np.zeros(width, np.float32))]
    results = []
    for name, x in cases:
        expected, ids, weights = reference.run(x)
        with tempfile.TemporaryDirectory(prefix="rust-moe-check-") as tmp:
            inp, out = pathlib.Path(tmp) / "input.f32", pathlib.Path(tmp) / "output.f32"
            x.astype("<f4").tofile(inp)
            command = [str(args.binary.resolve()), "dev", "moe-layer", str(args.model), str(args.layer), str(inp), str(out)]
            run = subprocess.run(command, text=True, capture_output=True)
            if run.returncode:
                raise RuntimeError(f"Rust engine exited {run.returncode}: {run.stderr}")
            lines = dict(line.split("=", 1) for line in run.stdout.splitlines())
            actual_ids = json.loads(lines["selected_experts"])
            actual_weights = json.loads(lines["routing_weights"])
            np.testing.assert_array_equal(actual_ids, ids)
            np.testing.assert_allclose(actual_weights, weights, rtol=2e-5, atol=2e-7)
            actual = np.fromfile(out, dtype="<f4")
            np.testing.assert_equal(actual.shape, expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=3e-5)
            error = actual.astype(np.float64) - expected.astype(np.float64)
            relative_l2 = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-30))
            if relative_l2 > 2e-5:
                raise AssertionError(f"relative L2 error {relative_l2} exceeds 2e-5")
            # A bad input and an existing output must be errors, preserving artifacts.
            before = out.read_bytes()
            collision = subprocess.run(command, text=True, capture_output=True)
            if collision.returncode == 0 or out.read_bytes() != before:
                raise AssertionError("existing-output protection failed")
            bad = pathlib.Path(tmp) / "bad.f32"
            np.full(width, np.nan, np.float32).tofile(bad)
            bad_out = pathlib.Path(tmp) / "bad-output.f32"
            failure = subprocess.run(command[:-2] + [str(bad), str(bad_out)], text=True, capture_output=True)
            if failure.returncode == 0 or bad_out.exists():
                raise AssertionError("non-finite-input error contract failed")
            results.append({"case": name, "device": lines["device"], "selected_experts": actual_ids,
                            "max_abs_error": float(np.abs(error).max()), "relative_l2_error": relative_l2,
                            "output_protection": "passed", "invalid_input": "passed"})
            print(f"{name}: PASS, max abs={results[-1]['max_abs_error']:.3g}, relative L2={relative_l2:.3g}", flush=True)
    report = {"stage": "single qwen4exp FFN; no attention, residual, tokenizer or text generation",
              "model": args.model.name, "layer": args.layer,
              "reference": "pinned ggml C dequantization plus NumPy; independent of Metal kernels",
              "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
              "ggml_sha256": hashlib.sha256(args.ggml.read_bytes()).hexdigest(),
              "numpy_version": np.__version__, "formats": sorted(reference.formats), "cases": results,
              "speed_comparison": "not performed"}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
