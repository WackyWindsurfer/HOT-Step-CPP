"""Build a reversible SGLang MiniMax-Music3 checkpoint bridge from explicit paths.

This writes only the requested experiment directory. It never opens an output
file for replacement and never mutates the official source or app models.
Existing Qwen shards, tokenizer files, and dav.pth are hardlinked; only the
renamed audio safetensors shard, config/index JSON, manifest, and strict DIT
flowmatching_vae.pth are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True, help="Official MM3 checkpoint root")
    p.add_argument("--output", type=Path, required=True, help="New output directory; must not contain generated files")
    p.add_argument("--allow-existing-output", action="store_true", help="reuse identical existing hardlinks only; generated files are always refused")
    return p.parse_args()


def ensure_new(path: Path, allow_existing: bool) -> None:
    if path.exists() and not allow_existing:
        raise RuntimeError(f"refusing existing output path (use a new directory): {path}")


def link_new(src: Path, dst: Path, allow_existing: bool) -> None:
    src = src.resolve(strict=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not allow_existing:
            raise RuntimeError(f"refusing existing output file: {dst}")
        if os.stat(src).st_ino == os.stat(dst).st_ino:
            return
        raise RuntimeError(f"output exists but is not the source hardlink: {dst}")
    os.link(src, dst)


def generated_path(dst: Path) -> None:
    if dst.exists():
        raise RuntimeError(f"refusing to overwrite generated output: {dst}")


def rename_rvq(src: Path) -> dict[str, torch.Tensor]:
    old = load_file(str(src), device="cpu")
    out: dict[str, torch.Tensor] = {}
    fixed = {
        "audio_embeddings.weight": "model.audio_extra_embedding.weight",
        "projection.weight": "model.audio_decoder.projection.weight",
        "pos_embedding.weight": "model.audio_decoder.pos_embedding.weight",
        "norm.weight": "model.audio_decoder.norm.weight",
    }
    for key, tensor in old.items():
        if key in fixed:
            target = fixed[key]
        elif key.startswith("audio_heads."):
            target = "model.audio_decoder." + key
        elif key.startswith("layers."):
            parts = key.split(".")
            if len(parts) == 4 and parts[2] in {"gate_proj", "up_proj", "down_proj"} and parts[3] == "weight":
                target = f"model.audio_decoder.layers.{parts[1]}.mlp.{parts[2]}.weight"
            elif len(parts) == 5 and parts[2] == "attn" and parts[4] == "weight":
                attn_name = {"to_q": "q_proj", "to_k": "k_proj", "to_v": "v_proj", "to_out": "o_proj"}.get(parts[3])
                if attn_name is None:
                    raise RuntimeError(f"unmapped RVQ attention key: {key}")
                target = f"model.audio_decoder.layers.{parts[1]}.self_attn.{attn_name}.weight"
            elif len(parts) == 4 and parts[2] in {"input_layernorm", "post_attention_layernorm"} and parts[3] == "weight":
                target = f"model.audio_decoder.layers.{parts[1]}.{parts[2]}.weight"
            else:
                # The known source has only the exact names validated by the metadata pass.
                raise RuntimeError(f"unmapped RVQ key: {key}")
        else:
            # MLP keys are layers.N.gate/up/down_proj.weight.
            parts = key.split(".")
            if len(parts) == 3 and parts[0] == "layers" and parts[2] == "weight":
                target = f"model.audio_decoder.layers.{parts[1]}.mlp.{parts[2]}"
            else:
                raise RuntimeError(f"unmapped RVQ key: {key}")
        if target in out:
            raise RuntimeError(f"duplicate generated RVQ key: {target}")
        out[target] = tensor
    if len(out) != len(old) or len(out) != 47:
        raise RuntimeError(f"RVQ mapping coverage mismatch: {len(old)} -> {len(out)} (expected 47)")
    return out


def load_sharded(directory: Path, index_name: str) -> dict[str, torch.Tensor]:
    index = json.loads((directory / index_name).read_text(encoding="utf-8"))
    out: dict[str, torch.Tensor] = {}
    for filename in sorted(set(index["weight_map"].values())):
        part = load_file(str(directory / filename), device="cpu")
        overlap = set(out).intersection(part)
        if overlap:
            raise RuntimeError(f"duplicate sharded tensor keys: {sorted(overlap)[:3]}")
        out.update(part)
    if set(out) != set(index["weight_map"]):
        raise RuntimeError("sharded safetensors keys do not match index")
    return out


def build_dit(source: Path) -> dict[str, torch.Tensor]:
    tr = load_sharded(source / "transformer", "diffusion_pytorch_model.safetensors.index.json")
    cond = load_file(str(source / "condition_encoder" / "diffusion_pytorch_model.safetensors"), device="cpu")
    out: dict[str, torch.Tensor] = {}
    direct = {
        "preprocess_conv.weight": "diffusion_transformer.preprocess_conv.weight",
        "proj_in.weight": "diffusion_transformer.transformer.project_in.weight",
        "proj_out.weight": "diffusion_transformer.transformer.project_out.weight",
        "time_embed.linear_1.weight": "diffusion_transformer.to_timestep_embed.0.weight",
        "time_embed.linear_1.bias": "diffusion_transformer.to_timestep_embed.0.bias",
        "time_embed.linear_2.weight": "diffusion_transformer.to_timestep_embed.2.weight",
        "time_embed.linear_2.bias": "diffusion_transformer.to_timestep_embed.2.bias",
        "time_proj.weight": "diffusion_transformer.timestep_features.weight",
        "postprocess_conv.weight": "diffusion_transformer.postprocess_conv.weight",
    }
    for old, target in direct.items():
        out[target] = tr.pop(old)
    for i in range(36):
        prefix = f"transformer_blocks.{i}"
        target = f"diffusion_transformer.transformer.layers.{i}"
        out[f"{target}.self_attn.to_qkv.weight"] = torch.cat(
            [tr.pop(f"{prefix}.attn.to_{n}.weight") for n in ("q", "k", "v")], dim=0
        )
        out[f"{target}.self_attn.to_out.weight"] = tr.pop(f"{prefix}.attn.to_out.0.weight")
        out[f"{target}.pre_norm.gamma"] = tr.pop(f"{prefix}.norm1.weight")
        out[f"{target}.pre_norm.beta"] = tr.pop(f"{prefix}.norm1.bias")
        out[f"{target}.ff_norm.gamma"] = tr.pop(f"{prefix}.norm2.weight")
        out[f"{target}.ff_norm.beta"] = tr.pop(f"{prefix}.norm2.bias")
        out[f"{target}.ff.ff.0.proj.weight"] = tr.pop(f"{prefix}.ff_in.weight")
        out[f"{target}.ff.ff.0.proj.bias"] = tr.pop(f"{prefix}.ff_in.bias")
        out[f"{target}.ff.ff.2.weight"] = tr.pop(f"{prefix}.ff_out.weight")
        out[f"{target}.ff.ff.2.bias"] = tr.pop(f"{prefix}.ff_out.bias")
    if tr:
        raise RuntimeError(f"unmapped transformer keys remain: {sorted(tr)[:5]}")
    out["cond_layer_logits"] = cond.pop("layer_weight_logits")
    out["cond_layer_scale"] = cond.pop("layer_scale")
    out["latent_conditioners.0.weight"] = cond.pop("proj.weight")
    out["latent_conditioners.0.bias"] = cond.pop("proj.bias")
    if cond:
        raise RuntimeError(f"unmapped condition keys remain: {sorted(cond)}")
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32) / 32.0))
    out["diffusion_transformer.transformer.rotary_pos_emb.inv_freq"] = inv_freq
    if len(out) != 374:
        raise RuntimeError(f"unexpected strict DIT state count: {len(out)} (expected 374)")
    return out


def main() -> None:
    a = parse_args()
    source = a.source.resolve(strict=True)
    output = a.output.resolve()
    allow = a.allow_existing_output
    if output.exists() and any(output.iterdir()) and not allow:
        raise RuntimeError(f"refusing non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    qwen = output / "qwen_7B" / "qwen_7B"
    tokenizer = output / "qwen_7B" / "qwen3-8B-tokenizer-music"
    ensure_new(qwen, allow)
    ensure_new(tokenizer, allow)
    qwen.mkdir(parents=True, exist_ok=True)
    tokenizer.mkdir(parents=True, exist_ok=True)
    for name in [
        "model-00001-of-00004.safetensors", "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors", "model-00004-of-00004.safetensors",
    ]:
        link_new(source / "language_model" / name, qwen / name, allow)
    for src_path in (source / "tokenizer").rglob("*"):
        if src_path.is_file(): link_new(src_path, tokenizer / src_path.relative_to(source / "tokenizer"), allow)
    link_new(source / "dav.pth", output / "dav.pth", allow)

    config = json.loads((source / "language_model" / "config.json").read_text(encoding="utf-8"))
    config.update({
        "model_type": "qwen3", "audio_vocab_size": 1024, "audio_num_codebooks": 8,
        "decoder_num_layers": 4, "decoder_num_heads": 16, "decoder_intermediate_size": 6144,
    })
    generated_path(qwen / "config.json")
    (qwen / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    # Keep core map filenames and metadata, then add generated audio keys.
    index = json.loads((source / "language_model" / "model.safetensors.index.json").read_text(encoding="utf-8"))
    audio = rename_rvq(source / "rvq_depth_decoder" / "diffusion_pytorch_model.safetensors")
    audio_path = qwen / "model-00005-of-00005.safetensors"
    generated_path(audio_path)
    save_file(audio, str(audio_path), metadata={"format": "pt"})
    for key in audio: index["weight_map"][key] = audio_path.name
    index.setdefault("metadata", {})["total_size"] = sum(t.numel() * t.element_size() for t in audio.values()) + int(index.get("metadata", {}).get("total_size", 0))
    index["metadata"]["total_parameters"] = int(index.get("metadata", {}).get("total_parameters", 0)) + sum(t.numel() for t in audio.values())
    generated_path(qwen / "model.safetensors.index.json")
    (qwen / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")

    dit = build_dit(source)
    flow = output / "flowmatching_vae.pth"
    generated_path(flow)
    torch.save(dit, str(flow))
    manifest = {
        "source": str(source), "target": str(output), "writes": {"audio_tensors": len(audio), "dit_tensors": len(dit)},
        "audio_dtype_counts": {str(dtype): sum(1 for t in audio.values() if t.dtype == dtype) for dtype in set(t.dtype for t in audio.values())},
        "dit_dtype_counts": {str(dtype): sum(1 for t in dit.values() if t.dtype == dtype) for dtype in set(t.dtype for t in dit.values())},
        "config_fields": {k: config[k] for k in ("model_type", "audio_vocab_size", "audio_num_codebooks", "decoder_num_layers", "decoder_num_heads", "decoder_intermediate_size")},
        "strict_dit_state_count": len(dit), "audio_state_count": len(audio),
    }
    generated_path(output / "MANIFEST.json")
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
