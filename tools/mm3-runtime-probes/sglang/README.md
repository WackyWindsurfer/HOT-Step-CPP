# SGLang MiniMax Music 3 renderer probe

This directory preserves the small, local SGLang experiment used to compare
the MiniMax Music 3 DiT/DAV renderer with HOT-Step's native renderer. It is an
isolated probe; it is not imported by the app and contains no model weights.

The source checkout used for the probe was `sgl-project/sglang-omni` at
`65e15007fa54b93880c47728f128ee2f44d32448` ([pinned commit](https://github.com/sgl-project/sglang-omni/tree/65e15007fa54b93880c47728f128ee2f44d32448)). Apply the two patches in
`patches/` to that checkout before running the renderer:

```bash
git apply /path/to/hot-step-cpp/tools/mm3-runtime-probes/sglang/patches/mm3-acoustic-cpu-load.patch
git apply /path/to/hot-step-cpp/tools/mm3-runtime-probes/sglang/patches/mm3-dit-f32-rope.patch
```

The acoustic patch maps DIT and DAV state on CPU before moving the modules to
the requested device, then trims temporary CUDA allocator blocks after load.
The RoPE patch recreates the analytic inverse frequencies and evaluates sine
and cosine in float32 before casting the cached values to the model dtype.
The earlier BF16 phase/trigonometry path produced audible corruption in the
local listening check; this is a quality correction, not a measured speed
claim.

`materialize_checkpoint.py` converts an official checkpoint into the layout
expected by the pinned loader. The source path must exist. The output path must
be new or empty and is the only location the script writes:

```bash
python materialize_checkpoint.py \
  --source /path/to/official-minimax-music3 \
  --output /path/to/new/mm3-materialized
```

The source must contain `language_model/`, `rvq_depth_decoder/`,
`condition_encoder/`, `transformer/`, `tokenizer/`, and `dav.pth`. Existing
Qwen shards, tokenizer files, and DAV state are hardlinked when source and
output are on the same filesystem; the implementation currently requires that
hardlink operation (copy them into an isolated staging filesystem first if
necessary). Renamed RVQ state, the strict DIT state, config/index JSON, and
manifest are generated in the new output directory. The script refuses to
overwrite generated files.

`bench_renderer.py` bypasses AR and replays one fixed hidden stream. Supply
the checkpoint and exactly one input source:

```bash
python bench_renderer.py \
  --checkpoint /path/to/mm3-materialized \
  --native-hidden /path/to/plan.mm3hiddens \
  --dtype bfloat16 --steps 30 --cfg 1.7 --repeats 3 \
  --save-first-44k1 /tmp/sglang-first-44k1.wav \
  --save-first /tmp/sglang-first-32k.wav
```

It reports input read, model load, compilation, each 44.1 kHz DiT/DAV render,
and the separate CPU resample to 32 kHz. Repeated output hashes must match.
Captured chunk directories remain supported, with exact 100-frame adjacent
overlap validation. Native `.mm3hiddens` input is sliced using the same 200
frame/100-frame-hop geometry as the native pipeline.

The recorded WSL environment snapshot is in the experiment logs: Python
3.12.13, SGLang 0.5.19, `cuda-tile==1.6.0rc5`, `torch==2.13.0`,
`torchaudio==2.11.0`, and `flashinfer-python==0.6.18`; `sglang-omni` was
installed from the pinned local checkout. These versions were checked from the
deployed environment's `*.dist-info` metadata. The accompanying pip check
recorded only non-MM3 optional extras missing and the upstream
`diffusers==0.37.0` pin versus installed 0.40.0. Treat this as a dated
environment record, not a validated installation recipe, and re-check a fresh
environment.

The native renderer emits 44.1 kHz audio while the SGLang service payload is
resampled to 32 kHz. Compare renderer timings at 44.1 kHz first, and report
resampling separately. A shared hidden stream establishes the renderer
comparison; it does not establish AR equivalence or overall generation speed.
Native and SGLang use different random-number generators, so passing the same
seed does not create the same initial noise or numerical trajectory; compare
fixed saved hidden inputs and report that limitation.
