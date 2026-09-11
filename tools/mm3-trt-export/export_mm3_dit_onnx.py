#!/usr/bin/env python3
"""Export the MM3 flow-DiT forward graph to ONNX for the TensorRT renderer.

The engine this feeds (``engine/src/minimax/mm3-dit-trt.h``) refits every one
of the 368 DiT parameters from the selected GGUF at load time, so the ONNX is
only ever used for its *structure*: node graph, initializer names, shapes and
dtypes, and the graph I/O contract.  The initializer *values* are dead weight.

``--weights none`` therefore builds the module with throwaway BF16 parameters
and drops every one of them: the graph keeps the standard external-data layout
(each initializer at its own offset in ``mm3-dit-trt.onnx.data``) but that file
is never published.  Whatever needs to parse the graph - TensorRT's builder
here, ``mm3-dit-trt.h``'s ``build()`` on a user's machine - synthesizes the
placeholder locally with ``make_placeholder_data`` and deletes it afterwards.
So a ~2 MB file ships instead of 4.86 GB.

``--weights <checkpoint>`` reproduces the original full export from
``flowmatching_vae.pth`` (4.86 GB of external data) for A/B work.

The module definition is intentionally self-contained: it is the contract the
GGUF converter (``engine/tools/convert-mm3.py``) and the native renderer both
implement, and it must not drift with a reference-source checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

# This exporter is CPU-only by contract: it must never take a share of a GPU
# that a training or benchmark run is using.  Set before torch is imported.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
from torch import Tensor, nn  # noqa: E402

from make_placeholder_data import write_placeholder  # noqa: E402

EMBED = 2048
HEADS = 32
HEAD_DIM = 64
ROPE_DIM = 32
FF_INNER = 8192
EPS = 1e-5

IN_CHANNELS = 128
CONDITION_DIM = 2048
CONCAT_CHANNELS = IN_CHANNELS * 2 + CONDITION_DIM
OUT_CHANNELS = 128
BLOCKS = 36
FOURIER_DIM = 256

ONNX_NAME = "mm3-dit-trt.onnx"
DATA_NAME = "mm3-dit-trt.onnx.data"
ALIGNMENT = 4096


class MM3TransformerBlock(nn.Module):
    """One MM3 transformer block with explicit FP32 RoPE table inputs."""

    def __init__(self) -> None:
        super().__init__()
        self.pre_gamma = nn.Parameter(torch.empty(EMBED))
        self.pre_beta = nn.Parameter(torch.empty(EMBED))
        self.qkv = nn.Linear(EMBED, EMBED * 3, bias=False)
        self.attn_out = nn.Linear(EMBED, EMBED, bias=False)
        self.ff_gamma = nn.Parameter(torch.empty(EMBED))
        self.ff_beta = nn.Parameter(torch.empty(EMBED))
        self.ff_in = nn.Linear(EMBED, FF_INNER * 2, bias=True)
        self.ff_out = nn.Linear(FF_INNER, EMBED, bias=True)

    @staticmethod
    def _rope(x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        # x is [B, H, S, D]; only the first 32 of D=64 rotate, rotate-half.
        rotated = x[..., :ROPE_DIM]
        cos = rope_cos[: x.shape[-2]].to(dtype=rotated.dtype, device=x.device)
        sin = rope_sin[: x.shape[-2]].to(dtype=rotated.dtype, device=x.device)
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        left, right = rotated.chunk(2, dim=-1)
        half = torch.cat((-right, left), dim=-1)
        rotated = rotated * cos + half * sin
        return torch.cat((rotated, x[..., ROPE_DIM:]), dim=-1)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        x = x.to(dtype=self.pre_gamma.dtype)
        pre = torch.nn.functional.layer_norm(x, (EMBED,), self.pre_gamma, self.pre_beta, EPS)
        q, k, v = self.qkv(pre).chunk(3, dim=-1)
        batch, seq, _ = q.shape
        q = self._rope(q.reshape(batch, seq, HEADS, HEAD_DIM).transpose(1, 2), rope_cos, rope_sin)
        k = self._rope(k.reshape(batch, seq, HEADS, HEAD_DIM).transpose(1, 2), rope_cos, rope_sin)
        v = v.reshape(batch, seq, HEADS, HEAD_DIM).transpose(1, 2)
        attended = (
            torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                scale=1.0 / math.sqrt(HEAD_DIM),
            )
            .transpose(1, 2)
            .contiguous()
            .reshape(batch, seq, EMBED)
        )
        x = x + self.attn_out(attended)
        ff_norm = torch.nn.functional.layer_norm(x, (EMBED,), self.ff_gamma, self.ff_beta, EPS)
        value, gate = self.ff_in(ff_norm).chunk(2, dim=-1)
        return (x + self.ff_out(value * torch.nn.functional.silu(gate))).float()


class MM3FullDiT(nn.Module):
    """Full one-step MM3 DiT, with precomputed Fourier timestep features."""

    def __init__(self) -> None:
        super().__init__()
        self.preprocess_conv = nn.Conv1d(CONCAT_CHANNELS, CONCAT_CHANNELS, 1, bias=False)
        self.project_in = nn.Linear(CONCAT_CHANNELS, EMBED, bias=False)
        self.time_embed = nn.Sequential(
            nn.Linear(FOURIER_DIM, EMBED),
            nn.SiLU(),
            nn.Linear(EMBED, EMBED),
        )
        self.blocks = nn.ModuleList([MM3TransformerBlock() for _ in range(BLOCKS)])
        self.project_out = nn.Linear(EMBED, OUT_CHANNELS, bias=False)
        self.postprocess_conv = nn.Conv1d(OUT_CHANNELS, OUT_CHANNELS, 1, bias=False)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        timestep_fourier: Tensor,
        rope_cos: Tensor,
        rope_sin: Tensor,
    ) -> Tensor:
        dtype = self.blocks[0].pre_gamma.dtype
        x = x.to(dtype=dtype)
        cond = cond.to(dtype=dtype)
        fourier = timestep_fourier.to(dtype=dtype)
        zeros = torch.zeros_like(x)
        full = torch.cat((x, zeros, cond), dim=1)
        full = self.preprocess_conv(full) + full
        h = self.project_in(full.transpose(1, 2))
        temb = self.time_embed(fourier)
        h = torch.cat((temb.unsqueeze(1), h), dim=1)
        for block in self.blocks:
            h = block(h, rope_cos, rope_sin).to(dtype=dtype)
        h = h[:, 1:]
        out = self.project_out(h).transpose(1, 2)
        return (self.postprocess_conv(out) + out).float()


def checkpoint_mapping() -> dict[str, str]:
    """Module parameter -> ``flowmatching_vae.pth`` tensor name."""
    result = {
        "preprocess_conv.weight": "diffusion_transformer.preprocess_conv.weight",
        "project_in.weight": "diffusion_transformer.transformer.project_in.weight",
        "project_out.weight": "diffusion_transformer.transformer.project_out.weight",
        "time_embed.0.weight": "diffusion_transformer.to_timestep_embed.0.weight",
        "time_embed.0.bias": "diffusion_transformer.to_timestep_embed.0.bias",
        "time_embed.2.weight": "diffusion_transformer.to_timestep_embed.2.weight",
        "time_embed.2.bias": "diffusion_transformer.to_timestep_embed.2.bias",
        "postprocess_conv.weight": "diffusion_transformer.postprocess_conv.weight",
    }
    for i in range(BLOCKS):
        target, source = f"blocks.{i}.", f"diffusion_transformer.transformer.layers.{i}."
        result.update({
            target + "pre_gamma": source + "pre_norm.gamma",
            target + "pre_beta": source + "pre_norm.beta",
            target + "qkv.weight": source + "self_attn.to_qkv.weight",
            target + "attn_out.weight": source + "self_attn.to_out.weight",
            target + "ff_gamma": source + "ff_norm.gamma",
            target + "ff_beta": source + "ff_norm.beta",
            target + "ff_in.weight": source + "ff.ff.0.proj.weight",
            target + "ff_in.bias": source + "ff.ff.0.proj.bias",
            target + "ff_out.weight": source + "ff.ff.2.weight",
            target + "ff_out.bias": source + "ff.ff.2.bias",
        })
    return result


def build_placeholder_model(dtype: torch.dtype, init: str, seed: int) -> MM3FullDiT:
    """Construct the module without any checkpoint.

    ``zeros`` looks like the obvious choice and is wrong: the ONNX optimizer
    drops an all-zero Gemm bias, so a zero-initialised export loses
    ``time_embed.0.bias`` and ``time_embed.2.bias`` as initializers and the
    refit then has nothing to write them into (379 initializers instead of
    381).  ``randn`` is the default for that reason.  The values themselves
    are thrown away by ``externalise`` - only the structure survives.
    """
    with torch.device("meta"):
        model = MM3FullDiT()
    generator = torch.Generator().manual_seed(seed)
    state = {}
    for name, parameter in model.named_parameters():
        if init == "zeros":
            state[name] = torch.zeros(parameter.shape, dtype=dtype)
        else:
            state[name] = (
                torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.02
            ).to(dtype=dtype)
    model.load_state_dict(state, strict=True, assign=True)
    return model.eval()


def load_checkpoint_model(checkpoint: Path, dtype: torch.dtype) -> tuple[MM3FullDiT, Tensor]:
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    mapping = checkpoint_mapping()
    missing = [s for s in mapping.values() if s not in state]
    if missing:
        raise RuntimeError(f"checkpoint is missing {len(missing)} tensors, first: {missing[0]}")
    fourier = state["diffusion_transformer.timestep_features.weight"].detach().float().cpu()
    with torch.device("meta"):
        model = MM3FullDiT()
    model.load_state_dict(
        {t: state[s].to(dtype=dtype) for t, s in mapping.items()}, strict=True, assign=True
    )
    del state
    return model.eval(), fourier


def rope_tables(seq: int) -> tuple[Tensor, Tensor]:
    positions = torch.arange(seq, dtype=torch.float32)
    inv = 1.0 / (10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM))
    freq = torch.einsum("i,j->ij", positions, inv)
    freq = torch.cat((freq, freq), dim=-1)
    return freq.cos(), freq.sin()


def export_program(model: MM3FullDiT, opset: int):
    batch, latent = 2, 689
    x = torch.randn(batch, IN_CHANNELS, latent, dtype=torch.float32)
    cond = torch.randn(batch, CONDITION_DIM, latent, dtype=torch.float32)
    fourier = torch.randn(batch, FOURIER_DIM, dtype=torch.float32)
    cos, sin = rope_tables(latent + 1)
    batch_dim = torch.export.Dim("batch", min=1, max=2)
    latent_dim = torch.export.Dim("latent", min=3, max=689)
    program = torch.onnx.export(
        model,
        (x, cond, fourier, cos, sin),
        input_names=["x", "cond", "timestep_fourier", "rope_cos", "rope_sin"],
        output_names=["output"],
        dynamic_shapes={
            "x": {0: batch_dim, 2: latent_dim},
            "cond": {0: batch_dim, 2: latent_dim},
            "timestep_fourier": {0: batch_dim},
            "rope_cos": {0: latent_dim + 1},
            "rope_sin": {0: latent_dim + 1},
        },
        opset_version=opset,
        dynamo=True,
    )
    if program is None:
        raise RuntimeError("torch.onnx.export returned no ONNXProgram; torch >= 2.6 is required")
    return program


def externalise(proto, blob=None) -> tuple[int, int]:
    """Move every rank>=1 BF16 initializer out of the protobuf.

    Each tensor gets its own disjoint range: initializers in graph order, each
    starting at the next 64-byte aligned offset, ``length = count * 2``.

    An earlier design pointed every tensor at offset 0 of one small shared
    blob.  onnx accepted it; TensorRT did not.  Its ONNX parser keys network
    weights by data pointer and value, so identical bytes at identical
    addresses collide:

        INetworkDefinition::setWeightsName: Error Code 3: API Usage Error
        (condition: wtsPtr->count() == w.count. Weights of same values but of
        different counts are used in the network.)

    Hence disjoint ranges, and a placeholder pattern that differs per offset.

    With ``blob`` open for writing the real values are written at the offsets
    assigned here (checkpoint mode).  Without it they are dropped, and the
    file is synthesized later by ``make_placeholder_data``.
    """
    import onnx
    from onnx.external_data_helper import set_external_data

    offset, count = 0, 0
    for init in proto.graph.initializer:
        if init.data_type != onnx.TensorProto.BFLOAT16 or not init.dims:
            continue
        length = len(init.raw_data)
        if length == 0:
            raise RuntimeError(f"initializer {init.name} has no raw_data to externalise")
        set_external_data(init, location=DATA_NAME, offset=offset, length=length)
        if blob is not None:
            blob.seek(offset)
            blob.write(init.raw_data)
        init.ClearField("raw_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        offset += (length + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
        count += 1
    return count, offset


def compare_graphs(left: Path, right: Path) -> dict:
    """Structural diff of two ONNX files, ignoring initializer values."""
    import onnx

    def summary(path: Path) -> dict:
        model = onnx.load(str(path), load_external_data=False)
        graph = model.graph
        return {
            "nodes": len(graph.node),
            "op_types": sorted({n.op_type for n in graph.node}),
            "op_counts": {t: sum(1 for n in graph.node if n.op_type == t)
                          for t in sorted({n.op_type for n in graph.node})},
            "inputs": [i.name for i in graph.input],
            "outputs": [o.name for o in graph.output],
            "opset": [(o.domain, o.version) for o in model.opset_import],
            "ir_version": model.ir_version,
            "initializers": {i.name: (i.data_type, list(i.dims)) for i in graph.initializer},
        }

    a, b = summary(left), summary(right)
    names_a, names_b = set(a["initializers"]), set(b["initializers"])
    shared = names_a & names_b
    return {
        "left": str(left),
        "right": str(right),
        "node_count": [a["nodes"], b["nodes"]],
        "node_count_equal": a["nodes"] == b["nodes"],
        "op_counts_equal": a["op_counts"] == b["op_counts"],
        "inputs_equal": a["inputs"] == b["inputs"],
        "outputs_equal": a["outputs"] == b["outputs"],
        "opset_equal": a["opset"] == b["opset"] and a["ir_version"] == b["ir_version"],
        "initializer_count": [len(names_a), len(names_b)],
        "initializer_names_equal": names_a == names_b,
        "only_in_left": sorted(names_a - names_b)[:10],
        "only_in_right": sorted(names_b - names_a)[:10],
        "initializer_shape_dtype_mismatches": [
            {"name": n, "left": a["initializers"][n], "right": b["initializers"][n]}
            for n in sorted(shared) if a["initializers"][n] != b["initializers"][n]
        ][:10],
        "identical": (
            a["nodes"] == b["nodes"]
            and a["op_counts"] == b["op_counts"]
            and a["inputs"] == b["inputs"]
            and a["outputs"] == b["outputs"]
            and a["opset"] == b["opset"]
            and a["ir_version"] == b["ir_version"]
            and names_a == names_b
            and all(a["initializers"][n] == b["initializers"][n] for n in shared)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True,
                        help="'none' for the shippable weightless graph, or a flowmatching_vae.pth path")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--init", choices=("zeros", "randn"), default="randn",
                        help="placeholder initializer values in weightless mode; "
                             "'zeros' is kept only to reproduce the folded-bias failure")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--compare", type=Path,
                        help="an existing .onnx to structurally diff the result against")
    parser.add_argument("--no-data", dest="write_data", action="store_false",
                        help="do not synthesize the placeholder .onnx.data (it is never shipped)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, (os.cpu_count() or 8) // 2))
    torch.manual_seed(args.seed)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path, data_path = out_dir / ONNX_NAME, out_dir / DATA_NAME
    for path in (onnx_path, data_path):
        if path.exists() and not args.force:
            raise RuntimeError(f"refusing to overwrite {path} (pass --force)")
        if path.exists():
            path.unlink()

    import onnx

    weightless = args.weights.lower() == "none"
    if weightless:
        model = build_placeholder_model(torch.bfloat16, args.init, args.seed)
    else:
        model, _ = load_checkpoint_model(Path(args.weights).resolve(strict=True), torch.bfloat16)

    program = export_program(model, args.opset)
    del model
    proto = program.model_proto
    del program
    if weightless:
        tensors, required = externalise(proto)
    else:
        # Checkpoint mode keeps the real BF16 values, written at the same
        # offsets the weightless layout would have assigned.
        with data_path.open("wb") as blob:
            tensors, required = externalise(proto, blob)
            blob.truncate(required)
    onnx.save(proto, str(onnx_path))
    del proto

    # The .onnx is the only file that ships.  Its external data is synthesized
    # here so the checks below (and TensorRT's builder) have bytes to read;
    # mm3-dit-trt.h does the same thing in C++ before it builds, and deletes
    # the file afterwards.
    if weightless and args.write_data:
        write_placeholder(data_path, required)

    # full_check=True runs ONNX shape inference, which rejects BF16 Conv in
    # onnx 1.22 (opset 18 Conv does not declare bfloat16).  The original full
    # weight export fails the same way; TensorRT's parser accepts it.  So the
    # structural check is the one that means anything here.
    onnx.checker.check_model(str(onnx_path))
    loaded = None
    if data_path.exists():
        reloaded = onnx.load(str(onnx_path), load_external_data=True)
        loaded = sum(1 for i in reloaded.graph.initializer if i.raw_data)
        del reloaded

    result = {
        "mode": "weightless" if weightless else "checkpoint",
        "init": args.init if weightless else str(Path(args.weights).resolve()),
        "onnx": str(onnx_path),
        "onnx_bytes": onnx_path.stat().st_size,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "external_data": str(data_path),
        "external_data_bytes": data_path.stat().st_size if data_path.exists() else 0,
        "external_data_required_bytes": required,
        "external_data_shipped": False,
        "external_tensors": tensors,
        "alignment": ALIGNMENT,
        "checker": "ok (structural; full_check rejects BF16 Conv in onnx 1.22, as it does for the full export)",
        "load_external_data": f"ok ({loaded} initializers materialised)" if loaded else "skipped",
        "contract": "mm3-dit-bf16-fp32-rope-v1",
    }
    if args.compare:
        result["graph_diff"] = compare_graphs(args.compare.resolve(strict=True), onnx_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
