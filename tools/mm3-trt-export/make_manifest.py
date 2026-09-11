#!/usr/bin/env python3
"""Build ``mm3-dit-trt.engine.json``: the GGUF-name -> TRT-weight-name map.

``engine/src/minimax/mm3-dit-trt.h`` reads this next to the serialized engine
and refits all 368 DiT parameters out of the selected GGUF, so every entry has
to name a real GGUF tensor and a real ONNX initializer, and the 368 have to
cover every ``dit.*`` tensor in the GGUF except the two the renderer computes
itself (``dit.rope_inv_freq`` and ``dit.time_fourier.weight``).

The ONNX initializer names are not stable across exports: the dynamo exporter
folds ``Linear`` weights into a pre-transposed constant and names those
``val_<n>`` by graph position.  So this manifest belongs to one specific
``mm3-dit-trt.onnx`` and must be regenerated whenever that file is.

Entry order is this script's own (module parameter order).  The order in the
manifest committed at ``models/mm3/mm3-dit-trt.engine.json`` came from a
TensorRT refitter enumeration of a built engine, which this tool has no access
to; the refit loop reads the array by name, so order carries no meaning.

Run with the same interpreter as export_mm3_dit_onnx.py - see README.md.
"""

from __future__ import annotations

import argparse
import json
import struct
from ast import literal_eval
from pathlib import Path

# ---------------------------------------------------------------- GGUF reader

GGUF_MAGIC = b"GGUF"
_SCALARS = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


class _Reader:
    def __init__(self, data: memoryview) -> None:
        self.data, self.pos = data, 0

    def take(self, count: int) -> memoryview:
        chunk = self.data[self.pos:self.pos + count]
        self.pos += count
        return chunk

    def scalar(self, fmt: str):
        return struct.unpack_from("<" + fmt, self.data, self._advance(struct.calcsize(fmt)))[0]

    def _advance(self, count: int) -> int:
        start, self.pos = self.pos, self.pos + count
        return start

    def string(self) -> str:
        return bytes(self.take(self.scalar("Q"))).decode("utf-8")

    def value(self, kind: int):
        if kind in _SCALARS:
            return self.scalar(_SCALARS[kind])
        if kind == 8:
            return self.string()
        if kind == 9:
            item = self.scalar("I")
            return [self.value(item) for _ in range(self.scalar("Q"))]
        raise RuntimeError(f"unknown GGUF value type {kind}")


def read_gguf_tensors(path: Path) -> dict[str, dict]:
    """Name -> {ne, ggml_type} from a GGUF header.  No tensor data is read."""
    with path.open("rb") as handle:
        head = memoryview(handle.read(64 * 1024 * 1024))
    reader = _Reader(head)
    if bytes(reader.take(4)) != GGUF_MAGIC:
        raise RuntimeError(f"not a GGUF file: {path}")
    version = reader.scalar("I")
    if version != 3:
        raise RuntimeError(f"unsupported GGUF version {version}")
    tensor_count, kv_count = reader.scalar("Q"), reader.scalar("Q")
    for _ in range(kv_count):
        reader.string()
        reader.value(reader.scalar("I"))
    tensors = {}
    for _ in range(tensor_count):
        name = reader.string()
        ne = [reader.scalar("Q") for _ in range(reader.scalar("I"))]
        tensors[name] = {"ne": ne, "ggml_type": reader.scalar("I"), "offset": reader.scalar("Q")}
    return tensors


# -------------------------------------------------------------- name mapping

def gguf_name_for(parameter: str) -> str:
    """Module parameter name -> GGUF tensor name (engine/tools/convert-mm3.py)."""
    fixed = {
        "preprocess_conv.weight": "dit.preprocess_conv.weight",
        "postprocess_conv.weight": "dit.postprocess_conv.weight",
        "project_in.weight": "dit.proj_in.weight",
        "project_out.weight": "dit.proj_out.weight",
        "time_embed.0.weight": "dit.time_embd.0.weight",
        "time_embed.0.bias": "dit.time_embd.0.bias",
        "time_embed.2.weight": "dit.time_embd.1.weight",
        "time_embed.2.bias": "dit.time_embd.1.bias",
    }
    if parameter in fixed:
        return fixed[parameter]
    head, index, tail = parameter.split(".", 2)
    if head != "blocks":
        raise RuntimeError(f"unmapped parameter: {parameter}")
    block = {
        "pre_gamma": "attn_norm.weight",
        "pre_beta": "attn_norm.bias",
        "ff_gamma": "ffn_norm.weight",
        "ff_beta": "ffn_norm.bias",
        "qkv.weight": "attn_qkv.weight",
        "attn_out.weight": "attn_output.weight",
        "ff_in.weight": "ffn_in.weight",
        "ff_in.bias": "ffn_in.bias",
        "ff_out.weight": "ffn_out.weight",
        "ff_out.bias": "ffn_out.bias",
    }[tail]
    return f"dit.blk.{index}.{block}"


def kind_for(parameter: str, transpose: bool, gemm: bool) -> str:
    if parameter.endswith(("pre_gamma", "ff_gamma")):
        return "norm_weight"
    if parameter.endswith(("pre_beta", "ff_beta")):
        return "norm_bias"
    if parameter.endswith(".bias"):
        return "bias"
    if parameter in ("preprocess_conv.weight", "postprocess_conv.weight"):
        return "conv1x1"
    if transpose:
        return "linear"
    if gemm:
        return "linear_gemm"
    raise RuntimeError(f"cannot classify {parameter}")


def resolve_engine_names(onnx_path: Path, parameters: dict[str, list[int]]) -> dict[str, dict]:
    """Parameter -> {engine_name, dims, transpose, gemm} from the ONNX graph.

    Two cases.  A parameter the exporter kept by name is its own initializer.
    A ``Linear`` weight was folded into a pre-transposed ``val_<n>`` constant
    feeding a MatMul; that node's ``pkg.torch.onnx.name_scopes`` metadata is
    what ties it back to the owning module.
    """
    import onnx

    graph = onnx.load(str(onnx_path), load_external_data=False).graph
    initializers = {i.name: i for i in graph.initializer}
    refittable = {
        name for name, i in initializers.items()
        if i.data_type == onnx.TensorProto.BFLOAT16 and len(i.dims) >= 1
    }
    modules = {p.rsplit(".", 1)[0] for p in parameters}
    resolved: dict[str, dict] = {}
    for name in list(refittable):
        if name in parameters:
            resolved[name] = {"engine_name": name, "dims": list(initializers[name].dims),
                              "transpose": False, "gemm": False}
    for node in graph.node:
        if node.op_type != "MatMul" or len(node.input) != 2 or node.input[1] not in refittable:
            continue
        props = {e.key: e.value for e in node.metadata_props}
        scopes = literal_eval(props.get("pkg.torch.onnx.name_scopes", "[]"))
        owners = [s for s in scopes if s and s in modules]
        if not owners:
            raise RuntimeError(f"MatMul {node.name} has no resolvable owning module")
        parameter = owners[-1] + ".weight"
        if parameter in resolved:
            raise RuntimeError(f"{parameter} resolved twice")
        dims = list(initializers[node.input[1]].dims)
        if dims != list(reversed(parameters[parameter])):
            raise RuntimeError(f"transposed shape mismatch for {parameter}: {dims}")
        resolved[parameter] = {"engine_name": node.input[1], "dims": dims,
                               "transpose": True, "gemm": False}
    gemm_weights = {
        node.input[1] for node in graph.node
        if node.op_type == "Gemm" and len(node.input) >= 2
        and any(a.name == "transB" and a.i == 1 for a in node.attribute)
    }
    for parameter, entry in resolved.items():
        entry["gemm"] = entry["engine_name"] in gemm_weights
    missing = sorted(set(parameters) - set(resolved))
    if missing:
        raise RuntimeError(f"{len(missing)} parameters unresolved, first: {missing[0]}")
    unmapped = sorted(refittable - {e["engine_name"] for e in resolved.values()})
    if unmapped:
        raise RuntimeError(f"refittable initializers not claimed by any parameter: {unmapped}")
    return resolved


# -------------------------------------------------------------------- driver

def build_manifest(onnx_path: Path, gguf_path: Path) -> dict:
    import torch

    from export_mm3_dit_onnx import MM3FullDiT

    with torch.device("meta"):
        model = MM3FullDiT()
    parameters = {name: list(p.shape) for name, p in model.named_parameters()}
    resolved = resolve_engine_names(onnx_path, parameters)
    gguf = read_gguf_tensors(gguf_path)

    weights, claimed = [], set()
    for parameter, shape in parameters.items():
        entry = resolved[parameter]
        gguf_name = gguf_name_for(parameter)
        tensor = gguf.get(gguf_name)
        if tensor is None:
            raise RuntimeError(f"{gguf_path.name} has no tensor {gguf_name}")
        # ggml stores ne fastest-axis first, i.e. the reverse of the torch shape.
        if tensor["ne"] != list(reversed(shape)):
            raise RuntimeError(f"{gguf_name} is {tensor['ne']}, expected {list(reversed(shape))}")
        claimed.add(gguf_name)
        count = 1
        for value in shape:
            count *= value
        weights.append({
            "gguf_name": gguf_name,
            "engine_name": entry["engine_name"],
            "transpose": entry["transpose"],
            "count": count,
            "dims": entry["dims"],
            "gguf_dims": list(tensor["ne"]),
            "pytorch_dims": list(shape),
            "kind": kind_for(parameter, entry["transpose"], entry["gemm"]),
        })

    # The engine asserts the same thing at load time; fail here instead.
    computed = {"dit.rope_inv_freq", "dit.time_fourier.weight"}
    stray = sorted(n for n in gguf if n.startswith("dit.") and n not in claimed and n not in computed)
    if stray:
        raise RuntimeError(f"GGUF DiT tensors no manifest entry covers: {stray}")
    if len(weights) != 368:
        raise RuntimeError(f"expected 368 weights, built {len(weights)}")
    # The engine synthesizes the (unshipped) external data file before building
    # (mm3-dit-trt.h build()); it learns the file name and size from here.
    total, location = external_data_contract(onnx_path)
    return {"version": 1, "contract": "mm3-dit-bf16-fp32-rope-v1",
            "onnx_data": {"location": location, "bytes": total}, "weights": weights}


def external_data_contract(onnx_path: Path) -> tuple[int, str]:
    """Bytes the graph's external initializers need, and the file they name
    (same rule as make_placeholder_data.required_bytes)."""
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


def compare_manifests(built: dict, reference: Path) -> dict:
    existing = json.loads(reference.read_text(encoding="utf-8"))
    index = {e["gguf_name"]: e for e in existing["weights"]}
    mine = {e["gguf_name"]: e for e in built["weights"]}
    differing = [
        {"gguf_name": name, "reference": index[name], "built": mine[name]}
        for name in sorted(set(index) & set(mine)) if index[name] != mine[name]
    ]
    same_order = [e["gguf_name"] for e in existing["weights"]] == [e["gguf_name"] for e in built["weights"]]
    return {
        "reference": str(reference),
        "header_equal": (existing["version"], existing["contract"]) == (built["version"], built["contract"]),
        "entry_count": [len(index), len(mine)],
        "gguf_names_equal": set(index) == set(mine),
        "only_in_reference": sorted(set(index) - set(mine))[:10],
        "only_in_built": sorted(set(mine) - set(index))[:10],
        "differing_entries": differing[:10],
        "differing_count": len(differing),
        "entry_order_equal": same_order,
        "semantically_identical": (
            set(index) == set(mine) and not differing
            and (existing["version"], existing["contract"]) == (built["version"], built["contract"])
        ),
        "byte_identical": False,
        "byte_identical_after_reorder": _reordered_bytes(built, existing) == reference.read_bytes(),
    }


def _reordered_bytes(built: dict, existing: dict) -> bytes:
    """This manifest written out in the reference's entry order, CRLF as on Windows."""
    mine = {e["gguf_name"]: e for e in built["weights"]}
    try:
        ordered = [mine[e["gguf_name"]] for e in existing["weights"]]
    except KeyError:
        return b""
    text = json.dumps({"version": built["version"], "contract": built["contract"],
                       "weights": ordered}, indent=2)
    return text.replace("\n", "\r\n").encode("utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--gguf", type=Path, required=True, help="an F16/F32 MM3 DiT GGUF")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path, help="an existing engine.json to check against")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = build_manifest(args.onnx.resolve(strict=True), args.gguf.resolve(strict=True))
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise RuntimeError(f"refusing to overwrite {output} (pass --force)")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(manifest, indent=2)
    output.write_text(text, encoding="utf-8")

    result = {"output": str(output), "bytes": output.stat().st_size,
              "weights": len(manifest["weights"]), "contract": manifest["contract"]}
    if args.compare:
        comparison = compare_manifests(manifest, args.compare.resolve(strict=True))
        comparison["byte_identical"] = (
            args.compare.read_bytes() == output.read_bytes()
        )
        result["comparison"] = comparison
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
