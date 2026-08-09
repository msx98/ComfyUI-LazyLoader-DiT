"""Rewrite a safetensors checkpoint in a smaller float dtype.

    python convert_dtype.py in.safetensors out.safetensors --dtype fp16

One tensor is in memory at a time, so an fp32 Flux checkpoint converts without
needing 23 GB of RAM.  Integer and already-narrower tensors are copied through
unchanged.  Metadata is preserved.

This is a one-time offline step, not part of the nodes.  It exists because the
pager converts during copy_ when the file dtype and the model dtype differ,
which costs both the extra bytes read and a per-element conversion on the
critical path.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import torch
from safetensors import safe_open

TARGETS = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
NAME_OF = {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32", torch.float64: "F64"}
WIDTH_OF = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4, torch.float64: 8}


def read_header(path):
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    return header


def raw_bytes(tensor):
    tensor = tensor.contiguous()
    if tensor.dtype is torch.bfloat16:
        # numpy has no bfloat16; reinterpret the same 2-byte elements.
        return tensor.view(torch.int16).numpy().tobytes()
    return tensor.numpy().tobytes()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--dtype", choices=sorted(TARGETS), default="fp16")
    args = parser.parse_args()
    target = TARGETS[args.dtype]

    if os.path.exists(args.destination):
        sys.exit(f"refusing to overwrite {args.destination}")

    original = read_header(args.source)
    metadata = original.get("__metadata__")
    keys = [key for key in original if key != "__metadata__"]

    # Plan the new layout first: only float tensors wider than the target are
    # narrowed, everything else keeps its dtype and size.
    plan, header, offset = [], {}, 0
    for key in keys:
        info = original[key]
        shape = info["shape"]
        source_name = info["dtype"]
        elements = 1
        for dimension in shape:
            elements *= dimension
        source_width = (info["data_offsets"][1] - info["data_offsets"][0]) // max(elements, 1)
        convert = source_name in ("F64", "F32", "F16", "BF16") and WIDTH_OF[target] < source_width
        new_name = NAME_OF[target] if convert else source_name
        new_width = WIDTH_OF[target] if convert else source_width
        length = elements * new_width
        header[key] = {"dtype": new_name, "shape": shape, "data_offsets": [offset, offset + length]}
        plan.append((key, convert, length))
        offset += length
    if metadata is not None:
        header["__metadata__"] = metadata

    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    padding = (-len(blob)) % 8          # data section starts 8-byte aligned
    blob += b" " * padding

    saved = 0
    with safe_open(args.source, framework="pt", device="cpu") as source, open(args.destination, "wb") as out:
        out.write(struct.pack("<Q", len(blob)))
        out.write(blob)
        for index, (key, convert, length) in enumerate(plan, 1):
            tensor = source.get_tensor(key)
            if convert:
                saved += tensor.numel() * tensor.element_size() - length
                tensor = tensor.to(target)
            written = out.write(raw_bytes(tensor))
            if written != length:
                sys.exit(f"{key}: wrote {written} bytes, planned {length}")
            del tensor
            if index % 50 == 0 or index == len(plan):
                print(f"  {index}/{len(plan)} tensors", end="\r", flush=True)
    print()
    print(f"wrote {args.destination}: {offset / 2**30:.2f} GiB ({saved / 2**30:.2f} GiB saved)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
