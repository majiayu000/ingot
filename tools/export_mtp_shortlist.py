"""Export an exact Q8_0/Q4_0 Qwen4Exp or GLM5-Next draft head from local int32 vocabulary IDs.

Use tools/draft_vocab.py for Qwen, or the native draft_vocab example for GLM.
The Python corpus tokenizer currently implements only the Qwen pre-tokenizer.
Requires NumPy and the pinned llama.cpp gguf-py package on PYTHONPATH.
The full tokenizer, embeddings and all other tensors are retained.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="original full-vocabulary Q8_0/Q4_0 MTP GGUF")
    parser.add_argument("ids", type=Path, help="unique little-endian int32 global token IDs")
    parser.add_argument("output", type=Path, help="new GGUF file")
    args = parser.parse_args()

    import numpy as np
    import gguf

    if args.output.exists():
        parser.error("output already exists")
    raw = args.ids.read_bytes()
    if len(raw) % 4:
        parser.error("IDs must be a little-endian int32 array")
    ids = np.frombuffer(raw, dtype="<i4")
    reader = gguf.GGUFReader(args.model)
    architecture = reader.get_field("general.architecture").contents()
    if architecture not in ("qwen4exp", "glm5-next"):
        parser.error("requires a supported Qwen4Exp or GLM5-Next MTP artifact")
    width = reader.get_field(architecture + ".embedding_length").contents()
    if reader.get_field("strata.draft_vocab_ids") is not None:
        parser.error("source already has a compact draft head")
    if any(t.name.startswith("blk.0.") for t in reader.tensors):
        parser.error("source must be an MTP-only model")
    vocab = len(reader.get_field("tokenizer.ggml.tokens").contents())
    if not 0 < len(ids) < vocab or np.any(ids < 0) or np.any(ids >= vocab) or len(np.unique(ids)) != len(ids):
        parser.error("IDs must be unique, in range, and smaller than the full vocabulary")
    head = next(t for t in reader.tensors if t.name == "output.weight")
    if head.tensor_type not in (gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0) or head.shape.tolist() != [width, vocab]:
        parser.error("expected a full Q8_0/Q4_0 [embedding length, vocabulary] draft head")

    writer = gguf.GGUFWriter(args.output, architecture)
    for field in reader.fields.values():
        if field.name == "general.architecture" or field.name.startswith("GGUF."):
            continue
        value_type = field.types[0]
        subtype = field.types[-1] if value_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), value_type, sub_type=subtype)
    writer.add_key_value("strata.draft_vocab_ids", ids.tolist(), gguf.GGUFValueType.ARRAY,
                         sub_type=gguf.GGUFValueType.INT32)
    arrays = []
    for tensor in reader.tensors:
        data = np.array(tensor.data[ids], copy=True) if tensor.name == "output.weight" else tensor.data
        writer.add_tensor_info(tensor.name, data.shape, data.dtype, data.nbytes, tensor.tensor_type)
        arrays.append(data)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for data in arrays:
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()

    check = gguf.GGUFReader(args.output)
    assert check.get_field("strata.draft_vocab_ids").contents() == ids.tolist()
    assert len(check.tensors) == len(reader.tensors)
    for old, new in zip(reader.tensors, check.tensors):
        expected = old.data[ids] if old.name == "output.weight" else old.data
        assert old.name == new.name and old.tensor_type == new.tensor_type
        assert np.array_equal(expected.view(np.uint8), new.data.view(np.uint8)), old.name
    assert all(reader.fields[k].contents() == check.fields[k].contents()
               for k in reader.fields if k.startswith("tokenizer."))
    print(f"Exported {len(ids)}/{vocab} exact quantized rows to {args.output}")


if __name__ == "__main__":
    main()
