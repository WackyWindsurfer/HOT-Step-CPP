#!/usr/bin/env python3
"""Synthesize the placeholder ``mm3-dit-trt.onnx.data`` a weightless ONNX expects.

The graph ships without its external-data file.  Anything that parses the ONNX
- TensorRT's builder, onnx.checker - needs *some* bytes at the offsets the
graph names, so they are generated locally instead of downloaded.  Every one
of the 368 values is overwritten by the GGUF refit before a single sample is
rendered, so the content only has to satisfy two constraints:

* nothing may be zero, NaN or Inf, or a constant-folding pass will rewrite the
  graph (an all-zero Gemm bias disappears entirely);
* no two tensors may hold the same bytes, or TensorRT's parser - which keys
  weights by data pointer and value - rejects the network with
  "Weights of same values but of different counts are used in the network".

Both fall out of making the value a function of its own position in the file,
through a hash that avalanches - every output bit depends on every input bit:

    k     = byte_offset / 2                    (BF16 element index, uint32)
    h     = k                                  (all arithmetic mod 2**32)
    h    ^= h >> 16
    h    *= 0x7FEB352D
    h    ^= h >> 15
    h    *= 0x846CA68B
    h    ^= h >> 16
    bits  = 0x3C00 + ((h >> 7) & 0x1FF)
    bits |= ((h >> 31) & 1) << 15              (sign)
    little-endian uint16 at byte_offset

``bits`` spans 0x3C00..0x3DFF, i.e. BF16 magnitudes in [0.0078125, 0.1245].
No exponent in that window is zero, denormal, or the all-ones NaN/Inf pattern.

The avalanche matters.  A plain multiplicative hash (``k * 2654435761``) leaves
the low 16 output bits a function of the low 16 input bits only, so the pattern
repeats every 256 KB; with 4096-aligned tensor offsets that gave only 61
distinct byte sequences across the 368 tensors, and TensorRT deduplicates by
content.  With the mix below all 368 differ.

Because the value depends only on the absolute byte offset, the generator
needs to know nothing about tensor boundaries: it writes N bytes of pattern.

That is the whole contract, and it is what makes a C++ reimplementation inside
``mm3-dit-trt.h``'s ``build()`` a twenty-line loop.  See README.md for the
offsets themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CHUNK_ELEMENTS = 32 << 20  # 64 MiB of output per pass


def pattern_chunk(start: int, count: int) -> np.ndarray:
    """The BF16 bit patterns for elements ``[start, start + count)``."""
    h = np.arange(start, start + count, dtype=np.uint32)
    with np.errstate(over="ignore"):  # uint32 multiplies wrap, as intended
        h ^= h >> np.uint32(16)
        h *= np.uint32(0x7FEB352D)
        h ^= h >> np.uint32(15)
        h *= np.uint32(0x846CA68B)
        h ^= h >> np.uint32(16)
    bits = (np.uint32(0x3C00) + ((h >> np.uint32(7)) & np.uint32(0x1FF))).astype(np.uint16)
    bits |= (((h >> np.uint32(31)) & np.uint32(1)) << np.uint32(15)).astype(np.uint16)
    return bits


def write_placeholder(path: Path, total_bytes: int) -> int:
    """Write ``total_bytes`` of the pattern to ``path``.  Returns bytes written."""
    if total_bytes % 2:
        raise RuntimeError("placeholder size must be an even number of bytes")
    written, elements = 0, total_bytes // 2
    with path.open("wb") as handle:
        start = 0
        while start < elements:
            count = min(CHUNK_ELEMENTS, elements - start)
            chunk = pattern_chunk(start, count)
            handle.write(chunk.tobytes())
            written += chunk.nbytes
            start += count
    if written != total_bytes:
        raise RuntimeError(f"wrote {written} bytes, expected {total_bytes}")
    return written


def required_bytes(onnx_path: Path) -> tuple[int, str]:
    """Bytes the graph's external initializers need, and the file they name."""
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    total, location = 0, ""
    for init in graph.initializer:
        if init.data_location != onnx.TensorProto.EXTERNAL:
            continue
        entry = {kv.key: kv.value for kv in init.external_data}
        total = max(total, int(entry["offset"]) + int(entry["length"]))
        if location and entry["location"] != location:
            raise RuntimeError("graph points at more than one external data file")
        location = entry["location"]
    if not location:
        raise RuntimeError(f"{onnx_path.name} has no external initializers")
    return total, location


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True,
                        help="the weightless graph; its offsets decide the file size")
    parser.add_argument("--output", type=Path,
                        help="defaults to the location the graph names, beside the .onnx")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve(strict=True)
    total, location = required_bytes(onnx_path)
    output = args.output.resolve() if args.output else onnx_path.parent / location
    if output.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite {output} (pass --force)")
    written = write_placeholder(output, total)
    print(f"{output}: {written} bytes of placeholder BF16 for {onnx_path.name}")


if __name__ == "__main__":
    main()
