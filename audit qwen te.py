"""Standalone. Prints the facts needed to add a Qwen TE profile.

    python audit_qwen_te.py /path/to/qwen3vl_4b_bf16.safetensors --comfy /path/to/ComfyUI

Deliberately stdlib-only: it parses the safetensors header directly and reads
the ComfyUI checkout as text, so it can run outside the ComfyUI environment and
never loads a tensor.  It is not imported by the node package and is not run
automatically.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys
from collections import defaultdict


def read_header(path):
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length))
    header.pop("__metadata__", None)
    return header


def collapse(key):
    return re.sub(r"\.\d+\.", ".{}.", key)


def report_checkpoint(path):
    header = read_header(path)
    print(f"== checkpoint == {path}")
    print(f"   tensors: {len(header)}   file: {os.path.getsize(path) / 2**30:.2f} GiB")

    # 1. Block lists: any "<stem>.<int>." segment, with its index range.
    lists = defaultdict(set)
    for key in header:
        for match in re.finditer(r"(?P<stem>[A-Za-z_][\w.]*?)\.(?P<index>\d+)\.", key):
            lists[match.group("stem")].add(int(match.group("index")))
    print("\n-- block lists --")
    for stem in sorted(lists):
        indices = lists[stem]
        print(f"   {stem}.N.  N in [{min(indices)}..{max(indices)}]  count={len(indices)}"
              f"  contiguous={sorted(indices) == list(range(min(indices), max(indices) + 1))}")

    # 2. Embedding / head / norm candidates, with dtype and shape.
    print("\n-- resident candidates (non-indexed keys) --")
    flat = [k for k in header if not re.search(r"\.\d+\.", k)]
    for key in sorted(flat):
        info = header[key]
        print(f"   {key}: {info['dtype']} {tuple(info['shape'])}")
    if not flat:
        print("   (none)")

    # 3. Per-block key signature and byte cost, per list.  This is what the
    #    pager streams, and what determines the slot size in the budget math.
    print("\n-- per-block signature --")
    for stem in sorted(lists):
        first = min(lists[stem])
        prefix = f"{stem}.{first}."
        members = sorted(k for k in header if k.startswith(prefix))
        if not members:
            continue
        block_bytes = sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0] for k in members)
        print(f"   {stem}.{first}.*  ({len(members)} tensors, {block_bytes / 2**20:.1f} MiB/block,"
              f" x{len(lists[stem])} = {block_bytes * len(lists[stem]) / 2**30:.2f} GiB)")
        for key in members:
            info = header[key]
            print(f"      {key[len(prefix):]}: {info['dtype']} {tuple(info['shape'])}")
        # Homogeneity across the list: _BlockSlots rejects mixed geometry.
        signature = [(k[len(prefix):], header[k]["dtype"], tuple(header[k]["shape"])) for k in members]
        odd = []
        for index in sorted(lists[stem]):
            other_prefix = f"{stem}.{index}."
            other = sorted(k for k in header if k.startswith(other_prefix))
            if [(k[len(other_prefix):], header[k]["dtype"], tuple(header[k]["shape"])) for k in other] != signature:
                odd.append(index)
        print(f"      homogeneous across list: {not odd}" + (f"  differing indices: {odd[:8]}" if odd else ""))

    # 4. Norm naming: decides whether the profile needs the Flux *_norm.weight
    #    -> *_norm.scale rewrite or the identity transform.
    print("\n-- norm key naming --")
    weights = sorted({collapse(k) for k in header if k.endswith("_norm.weight")})
    scales = sorted({collapse(k) for k in header if k.endswith("_norm.scale")})
    print(f"   *_norm.weight: {len(weights)} distinct -> {weights[:6]}")
    print(f"   *_norm.scale : {len(scales)} distinct -> {scales[:6]}")
    print("   => key_transform should be "
          + ("_identity_key" if weights and not scales else "_checkpoint_key" if scales and not weights else "AMBIGUOUS -- inspect"))

    # 5. Byte totals per top-level prefix, for the resident/budget estimate.
    print("\n-- bytes by top-level prefix --")
    totals = defaultdict(int)
    for key, info in header.items():
        totals[key.split(".")[0]] += info["data_offsets"][1] - info["data_offsets"][0]
    for prefix in sorted(totals, key=lambda p: -totals[p]):
        print(f"   {prefix}: {totals[prefix] / 2**30:.2f} GiB")


def report_comfy(root):
    print(f"\n== ComfyUI checkout == {root}")
    targets = [
        os.path.join(root, "comfy", "sd.py"),
        os.path.join(root, "comfy", "text_encoders", "flux.py"),
        os.path.join(root, "comfy", "text_encoders", "llama.py"),
        os.path.join(root, "comfy", "text_encoders", "qwen_image.py"),
    ]
    pattern = re.compile(r"qwen3.?vl|qwen3_vl|vision|visual|klein_te|Tokenizer|clip_qwen", re.IGNORECASE)
    for path in targets:
        if not os.path.exists(path):
            print(f"\n-- {path}: MISSING --")
            continue
        print(f"\n-- {os.path.relpath(path, root)} --")
        with open(path, encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                text = line.rstrip()
                if pattern.search(text) or re.match(r"\s*(def |class )", text) and "qwen" in text.lower():
                    print(f"   {number:5d}: {text[:160]}")

    # Anything else in the tree that mentions the VL encoder at all.
    print("\n-- other files mentioning qwen3vl / qwen3_vl --")
    hits = 0
    for base, dirs, files in os.walk(os.path.join(root, "comfy")):
        dirs[:] = [d for d in dirs if d not in {"__pycache__", ".git"}]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            with open(path, encoding="utf-8", errors="replace") as handle:
                for number, line in enumerate(handle, 1):
                    if re.search(r"qwen3.?vl", line, re.IGNORECASE):
                        print(f"   {os.path.relpath(path, root)}:{number}: {line.strip()[:160]}")
                        hits += 1
    if not hits:
        print("   (none -- this ComfyUI checkout has no Qwen3-VL text encoder support)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--comfy", default=None, help="path to the ComfyUI checkout")
    args = parser.parse_args()
    report_checkpoint(args.checkpoint)
    if args.comfy:
        report_comfy(args.comfy)
    else:
        print("\n(pass --comfy /path/to/ComfyUI for the module-side half of the profile)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
