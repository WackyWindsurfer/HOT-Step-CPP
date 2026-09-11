"""Benchmark the SGLang DiT+DAV renderer from one fixed hidden stream.

This intentionally bypasses AR and performs no server request.  Construction
and compilation are timed separately from repeated, identical 30-step renders.
Each render is timed at native 44.1 kHz before the final CPU resample to 32 kHz;
the resample is reported separately.  The script is intended for WSL/CUDA and
must be launched by the parent benchmark coordinator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import time
from pathlib import Path

PROCESS_STARTED = time.perf_counter()
import numpy as np
import torch

from sglang_omni.models.minimax_music3.acoustic import (
    MiniMaxMusic3AcousticDecoder,
    resample_waveform,
)
from sglang_omni.models.minimax_music3.chunking import (
    ChunkWindow,
    chunk_windows,
    crop_sample_bounds,
)


CHUNK_RE = re.compile(r"chunk(\d+)_hidden\.pt$")


def load_chunks(directory: Path, pattern: str) -> tuple[list[torch.Tensor], int]:
    paths = []
    for path in directory.glob(pattern):
        match = CHUNK_RE.search(path.name)
        if match:
            paths.append((int(match.group(1)), path))
    paths.sort()
    if [index for index, _ in paths] != list(range(len(paths))):
        raise ValueError("chunk indexes must be contiguous and start at zero")
    chunks = []
    for index, path in paths:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1 or value.shape[2] != 32768:
            raise ValueError(f"{path} has unexpected hidden shape")
        if not torch.isfinite(value.float()).all().item():
            raise ValueError(f"{path} contains non-finite values")
        chunks.append(value[0].contiguous())
    if not chunks:
        raise FileNotFoundError(f"no chunks matching {pattern!r}")
    for index in range(1, len(chunks)):
        previous = chunks[index - 1]
        current = chunks[index]
        if previous.shape[0] < 100 or current.shape[0] < 100:
            raise ValueError("captured windows must contain at least 100 overlap frames")
        if not torch.equal(previous[-100:], current[:100]):
            raise ValueError(f"captured chunk overlap mismatch between {index - 1} and {index}")
    unique_frames = int(chunks[0].shape[0] + sum(chunk.shape[0] - 100 for chunk in chunks[1:]))
    return chunks, unique_frames


def _read_i32(raw: bytes, offset: int) -> tuple[int, int]:
    if offset + 4 > len(raw):
        raise ValueError("truncated hidden-file metadata")
    return struct.unpack_from("<i", raw, offset)[0], offset + 4


def _skip_string(raw: bytes, offset: int) -> int:
    length, offset = _read_i32(raw, offset)
    if length < 0 or offset + length > len(raw):
        raise ValueError("invalid hidden-file string length")
    return offset + length


def _skip_codes(raw: bytes, offset: int) -> int:
    count, offset = _read_i32(raw, offset)
    if count < 0 or offset + count * 4 > len(raw):
        raise ValueError("invalid hidden-file code-vector length")
    return offset + count * 4


def load_native_hidden(path: Path) -> tuple[list[torch.Tensor], int]:
    raw = path.read_bytes()
    if raw[:8] != b"MM3HIDN1":
        raise ValueError(f"{path} is not an MM3HIDN1 file")
    if len(raw) < 36:
        raise ValueError("truncated hidden-file header")
    frames, codebooks, embedding = struct.unpack_from("<qqq", raw, 8)
    if frames < 1 or codebooks != 8 or embedding != 4096:
        raise ValueError(f"unexpected hidden shape [{frames}, {codebooks}, {embedding}]")
    offset = 32
    _, offset = _read_i32(raw, offset)  # eos_hit
    offset = _skip_string(raw, offset)  # model key
    offset = _skip_string(raw, offset)  # full key
    offset = _skip_codes(raw, offset)
    offset = _skip_codes(raw, offset)
    offset = _skip_string(raw, offset)  # lrc
    expected = int(frames) * int(codebooks) * int(embedding) * 4
    if len(raw) - offset != expected:
        raise ValueError(f"hidden payload has {len(raw) - offset} bytes, expected {expected}")
    array = np.frombuffer(raw, dtype="<f4", count=expected // 4, offset=offset)
    hidden = torch.from_numpy(array.reshape(int(frames), int(codebooks) * int(embedding)).copy())
    if not torch.isfinite(hidden).all().item():
        raise ValueError(f"{path} contains non-finite hidden values")
    chunks = [hidden[window.start : window.end].contiguous() for window in chunk_windows(int(frames))]
    return chunks, int(frames)


def render(decoder, chunks: list[torch.Tensor], *, seed: int, steps: int, cfg: float) -> tuple[torch.Tensor, float]:
    previous_latent = None
    previous_condition = None
    outputs = []
    started = time.perf_counter()
    for index, hidden in enumerate(chunks):
        waveform, previous_latent, previous_condition = decoder.decode_with_state(
            hidden,
            seed=seed,
            chunk_idx=index,
            initial_latent=previous_latent,
            initial_condition=previous_condition,
        )
        if not torch.isfinite(waveform).all().item():
            raise RuntimeError(f"non-finite waveform in chunk {index}")
        left, right = crop_sample_bounds(
            ChunkWindow(index, 0, int(hidden.shape[0]), index == 0, index == len(chunks) - 1)
        )
        end = waveform.shape[-1] - right if right else waveform.shape[-1]
        outputs.append(waveform[:, left:end])
    torch.cuda.synchronize()
    return torch.cat(outputs, dim=1), time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True, help="Materialized MM3 checkpoint root")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-dir", type=Path, help="Directory of captured overlapping hidden chunks")
    input_group.add_argument("--native-hidden", type=Path, help="One MM3HIDN1 file; windows are derived from its full stream")
    parser.add_argument("--pattern", default="seed0_chunk*_hidden.pt")
    parser.add_argument("--device", default="cuda:0", help="CUDA device for the renderer")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=1.7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--save-first", type=Path, default=None)
    parser.add_argument("--save-first-44k1", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=Path(__file__).resolve().parent / "bench-long-renderer-result.json")
    args = parser.parse_args()
    if args.repeats < 1 or args.steps < 1:
        raise ValueError("repeats and steps must be positive")
    print("imports complete", flush=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    input_started = time.perf_counter()
    if args.native_hidden is not None:
        chunks, unique_frames = load_native_hidden(args.native_hidden)
        input_source = str(args.native_hidden)
    else:
        chunks, unique_frames = load_chunks(args.input_dir, args.pattern)
        input_source = str(args.input_dir)
    input_s = time.perf_counter() - input_started
    print(f"input loaded: {len(chunks)} windows, {unique_frames} unique frames in {input_s:.3f}s", flush=True)

    load_started = time.perf_counter()
    decoder = MiniMaxMusic3AcousticDecoder(
        str(args.checkpoint), device=args.device, dtype=args.dtype, dit_steps=args.steps,
        dit_cfg_scale=args.cfg, compile_acoustic=False,
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - load_started
    print(f"model load complete: {load_s:.3f}s", flush=True)
    compile_s = 0.0
    if not args.eager:
        compile_started = time.perf_counter()
        window = decoder.dit.aligned_mel_length(200)
        decoder.dit.enable_compiled_blocks(warmup_mel_length=window)
        decoder.dav.enable_compiled_decoder(warmup_mel_length=window)
        torch.cuda.synchronize()
        compile_s = time.perf_counter() - compile_started
        print(f"compile complete: {compile_s:.3f}s", flush=True)
    else:
        print("compile skipped (eager)", flush=True)

    ready_s = time.perf_counter() - PROCESS_STARTED
    runs = []
    first_audio = None
    first_native_audio = None
    for repeat in range(args.repeats):
        waveform, render_s = render(decoder, chunks, seed=args.seed, steps=args.steps, cfg=args.cfg)
        resample_started = time.perf_counter()
        output_32k = resample_waveform(waveform)
        resample_s = time.perf_counter() - resample_started
        if not torch.isfinite(output_32k).all().item():
            raise RuntimeError(f"non-finite 32 kHz output on repeat {repeat}")
        if first_audio is None:
            first_audio = output_32k
            first_native_audio = waveform
        render_hash = hashlib.sha256(waveform.numpy().tobytes()).hexdigest()
        output_hash = hashlib.sha256(output_32k.numpy().tobytes()).hexdigest()
        run = {"repeat": repeat, "render_44k1_s": render_s, "resample_32k_s": resample_s, "samples_44k1": int(waveform.shape[-1]), "samples_32k": int(output_32k.shape[-1]), "channels": int(output_32k.shape[0]), "render_sha256": render_hash, "output32k_sha256": output_hash, "finite": True}
        runs.append(run)
        print(f"repeat {repeat}: render44k1={render_s:.3f}s resample32k={resample_s:.3f}s sha256={render_hash[:16]}", flush=True)

    if args.save_first is not None and first_audio is not None:
        import numpy as np
        from sglang_omni.client.audio import encode_wav
        args.save_first.write_bytes(encode_wav(first_audio.numpy(), 32000))
    if args.save_first_44k1 is not None and first_native_audio is not None:
        from sglang_omni.client.audio import encode_wav
        args.save_first_44k1.write_bytes(encode_wav(first_native_audio.numpy(), 44100))
    result = {"checkpoint": str(args.checkpoint), "input_source": input_source, "input_dir": str(args.input_dir), "pattern": args.pattern, "chunks": len(chunks), "input_frames_unique": unique_frames, "input_frames_window_sum": sum(int(x.shape[0]) for x in chunks), "input_load_s": input_s, "dtype": args.dtype, "compiled": not args.eager, "steps": args.steps, "cfg": args.cfg, "seed": args.seed, "load_s": load_s, "compile_s": compile_s, "runs": runs, "native_comparison_note": "GGML output is 44.1 kHz; SGLang resample timing/output is reported separately at 32 kHz. No speed or numerical parity claim is made until native and SGLang use the same captured hidden stream."}
    result.update(torch=torch.__version__, device=torch.cuda.get_device_name(), seconds_to_ready=ready_s,
                  gpu_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                  gpu_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
    if len({run['render_sha256'] for run in runs}) != 1:
        raise RuntimeError('Repeated renderer outputs changed on identical inputs')
    args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
