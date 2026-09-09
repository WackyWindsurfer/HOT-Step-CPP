#!/usr/bin/env python3
"""Convert an ace-train PiSSA SVD cache (.pissa, train/lm-pissa.h) into the
shippable residual file (pissa-residual.h):

    <models>/mm3/<base stem>.pissa-r<rank>.safetensors

One-off tool (2026-09-09). The trainer writes this file itself on the first
PiSSA run beside a base that has none; this exists so the file for the shipped
mm3-lm-q8_0 base could be produced from the cache Rob's machine already held,
uploaded, and registered before any new run.

The cache stores the factors AFTER the 1/sqrt(s) division of the run that
wrote it. The residual stores CANONICAL factors (scale 1). They coincide when
that run had alpha == rank, which is what every HOT-PiZZA recipe uses; pass
--scale otherwise and the values are multiplied back by sqrt(scale).

Usage:
    python pissa_cache_to_residual.py <cache.pissa> <base.gguf> [--out <file>] [--scale 1.0]
"""
import argparse
import os
import struct
import sys

import numpy as np
from safetensors.numpy import save_file, load_file

SITES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
         "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
VERSION = 1


def read_cache(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == b"HPIS", f"not a PiSSA cache: {magic!r}"
        (ver,) = struct.unpack("<I", f.read(4))
        assert ver == 1, ver
        rank, q, iters, lo, hi, n = struct.unpack("<6i", f.read(24))
        sites = []
        for _ in range(n):
            l, s, inn, out = struct.unpack("<4i", f.read(16))
            (frac,) = struct.unpack("<d", f.read(8))
            a0 = np.frombuffer(f.read(inn * rank * 4), dtype=np.float32).reshape(rank, inn)   # ggml [in,r] == torch [r,in]
            b0 = np.frombuffer(f.read(rank * out * 4), dtype=np.float32).reshape(out, rank)   # ggml [r,out] == torch [out,r]
            sites.append((l, s, inn, out, frac, a0, b0))
    return dict(rank=rank, q=q, iters=iters, layer_lo=lo, layer_hi=hi), sites


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("base")
    ap.add_argument("--out")
    ap.add_argument("--scale", type=float, default=1.0)
    a = ap.parse_args()

    meta, sites = read_cache(a.cache)
    base_size = os.path.getsize(a.base)
    stem = os.path.splitext(os.path.basename(a.base))[0]
    out = a.out or os.path.join(os.path.dirname(a.base), f"{stem}.pissa-r{meta['rank']}.safetensors")

    fracs = [s[4] for s in sites]
    e_mean, e_min = float(np.mean(fracs)), float(np.min(fracs))
    hi, lo = float(base_size // 65536), float(base_size % 65536)
    tensors = {
        "pissa.meta": np.array([VERSION, meta["rank"], meta["q"], meta["iters"], meta["layer_lo"], meta["layer_hi"],
                                hi, lo, e_mean, e_min], dtype=np.float32),
    }
    sq = np.float32(np.sqrt(a.scale))
    for (l, s, inn, outd, frac, a0, b0) in sites:
        name = f"pissa.L{l}.{SITES[s]}"
        tensors[name + ".A0"] = (a0 * sq).astype(np.float16)   # RNE, same as the engine's rounding
        tensors[name + ".B0"] = (b0 * sq).astype(np.float16)
    md = {"format": "pt", "hot_step_pissa_residual": "v1", "producer": "pissa_cache_to_residual.py",
          "base_file": os.path.basename(a.base), "base_size": str(base_size)}
    save_file(tensors, out, metadata=md)

    # Read back and check one site round-trips exactly.
    back = load_file(out)
    l, s, inn, outd, frac, a0, b0 = sites[0]
    assert np.array_equal(back[f"pissa.L{l}.{SITES[s]}.A0"], (a0 * sq).astype(np.float16))
    print(f"wrote {out}: {len(sites)} sites, rank {meta['rank']}, q {meta['q']}, iters {meta['iters']}, "
          f"layers {meta['layer_lo']}-{meta['layer_hi']}, base {base_size} B, energy mean {e_mean:.4f} min {e_min:.4f}, "
          f"{os.path.getsize(out) / 1e9:.2f} GB")


if __name__ == "__main__":
    sys.exit(main())
