# Verify that the weightless ONNX produces the same audio as the 4.86 GB one.
#
# WHAT THIS PROVES
# ----------------
# The shipped mm3-dit-trt.onnx carries no real weights: every one of the 368
# DiT parameters is refit from the selected GGUF at load time.  The claim to
# test is therefore that an engine BUILT from the weightless graph, once
# refit, renders bit-for-bit what the engine built from the full BF16 graph
# rendered.  If the weightless graph had lost or reordered an initializer, the
# refit would either fail loudly or write a weight into the wrong place, and
# the audio would change.
#
# WHAT IT COMPARES
# ----------------
# One deterministic replay of the 273-frame saved plan
# (native-renderer-probe/short-final-input.mm3hiddens, a crop of the adapter
# plan; two windows, L=689 then L=596, so it exercises the shape switch inside
# one profile).  The plan is forced from file and reuse_ar is set, so the
# planner LM contributes nothing new and the only variable is the DiT.
#
#   baseline: native-renderer-probe/integrated-native-warm.wav
#             sha256 e2fe23261e7cfdf206d4d30c4d387be97f48c24b289092ae742852bba96893cf
#             (identical across the cold, warm and cache-reload runs of
#              2026-09-11, so byte equality is a fair bar, not a fluke)
#   new:      <OutDir>/weightless-verify.wav
#
# Pass = identical sha256.  If they differ the script still reports the RMS of
# the sample-wise difference and the relative RMSE, because a tiny non-zero
# delta means "TensorRT picked different kernels" (acceptable, the plan's gate
# is < 1e-3 relative) while a large one means the refit went to the wrong
# tensors (a real failure).
#
# GPU RULE: this is the only part of the weightless-ONNX work that needs the
# GPU.  Do not run it while a benchmark or training job holds the card.
#
# Usage (from the repo root, after the GPU is free):
#   ./tools/mm3-trt-export/verify_weightless_build.ps1
#   ./tools/mm3-trt-export/verify_weightless_build.ps1 -SkipBuild   # reuse engine

[CmdletBinding()]
param(
    [string]$OutDir = 'D:\Ace-Step-Latest\hot-step-cpp\_experiments\2026-09-11-mm3-speed\weightless-onnx',
    [string]$Label  = 'weightless-verify',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$repo      = 'D:\Ace-Step-Latest\hot-step-cpp'
$task      = Join-Path $repo '_experiments\2026-09-11-mm3-speed'
$probe     = Join-Path $task 'native-renderer-probe'
$trtLibs   = Join-Path $repo 'engine\deps\tensorrt_libs'
# The probe's --build target: kREFIT + kTF32 + kSTRONGLY_TYPED, one shared
# optimisation profile (B 1..2, L 3..689, rope 4..690).  That is the same
# recipe mm3-dit-trt.h's own build() uses, which is why an engine from here is
# a valid stand-in for one the app would build on a user's machine.
$builder   = Join-Path $probe 'build\Release\mm3_full_trt_bench.exe'
$onnx      = Join-Path $OutDir 'mm3-dit-trt.onnx'
$manifest  = Join-Path $OutDir 'mm3-dit-trt.engine.json'
$engine    = Join-Path $OutDir 'mm3-dit-trt-weightless.engine'
$baseline  = Join-Path $probe 'integrated-native-warm.wav'
$plan      = 'short-final-input.mm3hiddens'
$produced  = Join-Path $probe "$Label.wav"

foreach ($required in @($builder, $onnx, $manifest, $baseline)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Missing prerequisite: $required" }
}

# The external-data file is never shipped; synthesize it if it is not here.
$data = Join-Path $OutDir 'mm3-dit-trt.onnx.data'
if (-not (Test-Path -LiteralPath $data)) {
    $py   = 'D:/Ace-Step-Latest/mm3-weights/.venv-ref/Scripts/python.exe'
    $deps = Join-Path $repo '_experiments\2026-09-11-mm3-speed\native-renderer-probe\export-deps'
    $env:PYTHONPATH = $deps
    & $py (Join-Path $PSScriptRoot 'make_placeholder_data.py') --onnx $onnx
    if ($LASTEXITCODE -ne 0) { throw 'Could not synthesize the placeholder external data' }
}
if (-not (Test-Path -LiteralPath $trtLibs)) { throw "TensorRT runtime DLLs are missing: $trtLibs" }

# Never race the user's own generations.
$queue = Invoke-RestMethod 'http://127.0.0.1:3001/api/generate/queue' -TimeoutSec 5
if ($queue.running -or $queue.pending) { throw 'The app queue is busy; wait for it to drain' }

$env:PATH = "$trtLibs;" + (Join-Path $repo '_experiments\npm-cache\_npx\52027bd8fc0022aa\node_modules\node\bin') + ";$env:PATH"

# ── 1. Build a TensorRT engine from the weightless graph ────────────────────
# The parser resolves mm3-dit-trt.onnx.data relative to the .onnx path, so all
# 368 initializers read their (zero) bytes from the one shared 64 MiB blob.
# The values do not matter; only that the build sees the right shapes and
# marks all 368 refittable.
if (-not $SkipBuild) {
    if (Test-Path -LiteralPath $engine) { Remove-Item -LiteralPath $engine }
    & $builder --build --onnx $onnx --engine $engine 2>&1 | Tee-Object -FilePath (Join-Path $OutDir 'weightless-build.log')
    if ($LASTEXITCODE -ne 0) { throw "Engine build failed (exit $LASTEXITCODE)" }
}
if (-not (Test-Path -LiteralPath $engine)) { throw "No engine at $engine" }

# mm3-dit-trt.h reads the manifest as "<engine path>.json" when
# MM3_DIT_TRT_ENGINE is set, so give the built engine its own copy.
Copy-Item -LiteralPath $manifest -Destination "$engine.json" -Force

# ── 2. Start an isolated ace-server on :8086 against that engine ────────────
# MM3_DIT_TRT_ENGINE has to be in the server process's environment, so the
# server must be (re)started here rather than reused.
if (Get-NetTCPConnection -LocalPort 8086 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8086 is in use; stop the previous test engine first'
}
$env:MM3_DIT_TRT_ENGINE = $engine
$server = Start-Process -FilePath (Join-Path $repo 'engine\build\Release\ace-server.exe') `
    -ArgumentList '--host 127.0.0.1 --port 8086 --models D:/Ace-Step-Latest/hot-step-cpp/models --max-batch 1' `
    -WorkingDirectory $repo -WindowStyle Hidden `
    -RedirectStandardOutput (Join-Path $OutDir "$Label-server.stdout.log") `
    -RedirectStandardError  (Join-Path $OutDir "$Label-server.stderr.log") -PassThru
Set-Content -LiteralPath (Join-Path $task 'test-server.pid') -Value $server.Id

try {
    for ($i = 0; $i -lt 120; $i++) {
        try { Invoke-RestMethod 'http://127.0.0.1:8086/mm3/props' -TimeoutSec 2 | Out-Null; break }
        catch { Start-Sleep -Milliseconds 500 }
    }
    $props = Invoke-RestMethod 'http://127.0.0.1:8086/mm3/props' -TimeoutSec 10
    $props.dit_runtime | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $OutDir "$Label-dit-runtime.json")
    if (-not $props.dit_runtime.available) {
        throw "TensorRT DiT is not available: $($props.dit_runtime.reason)"
    }

    # ── 3. Replay the 273-frame plan through the refitted weightless engine ──
    # Same harness, same request, same saved plan as the baseline run.
    & (Join-Path $probe 'run-integrated-monitored.ps1') -Label $Label -Backend tensorrt -Plan $plan
    if ($LASTEXITCODE -ne 0) { throw "Replay failed (exit $LASTEXITCODE)" }
}
finally {
    if (-not $server.HasExited) { $server.Kill(); $server.WaitForExit() }
    Remove-Item Env:MM3_DIT_TRT_ENGINE -ErrorAction SilentlyContinue
}

# ── 4. Compare ──────────────────────────────────────────────────────────────
if (-not (Test-Path -LiteralPath $produced)) { throw "Replay produced no audio at $produced" }
$expected = (Get-FileHash -LiteralPath $baseline -Algorithm SHA256).Hash
$actual   = (Get-FileHash -LiteralPath $produced -Algorithm SHA256).Hash
$report = [ordered]@{
    engine          = $engine
    engine_bytes    = (Get-Item -LiteralPath $engine).Length
    baseline        = $baseline
    baseline_sha256 = $expected
    produced        = $produced
    produced_sha256 = $actual
    identical       = ($expected -eq $actual)
}

if (-not $report.identical) {
    # 16-bit PCM WAV, 44-byte canonical header from the same writer, so the
    # payload lines up sample for sample.
    $a = [IO.File]::ReadAllBytes($baseline)
    $b = [IO.File]::ReadAllBytes($produced)
    if ($a.Length -ne $b.Length) { $report.length_mismatch = @($a.Length, $b.Length) }
    else {
        $n = ($a.Length - 44) / 2
        $sumSquaredError = 0.0; $sumSquaredRef = 0.0; $peak = 0
        for ($i = 0; $i -lt $n; $i++) {
            $o = 44 + $i * 2
            $x = [BitConverter]::ToInt16($a, $o)
            $y = [BitConverter]::ToInt16($b, $o)
            $d = [double]($x - $y)
            $sumSquaredError += $d * $d; $sumSquaredRef += [double]$x * $x
            if ([math]::Abs($d) -gt $peak) { $peak = [math]::Abs($d) }
        }
        $report.rmse          = [math]::Sqrt($sumSquaredError / $n)
        $report.relative_rmse = [math]::Sqrt($sumSquaredError / [math]::Max($sumSquaredRef, 1e-12))
        $report.max_abs_delta_lsb = $peak
        $report.verdict = if ($report.relative_rmse -lt 1e-3) {
            'kernel-level difference only; within the plan gate'
        } else {
            'FAIL: the refit did not reproduce the prepared engine'
        }
    }
}

$report | ConvertTo-Json -Depth 5 | Tee-Object -FilePath (Join-Path $OutDir "$Label-comparison.json")
if (-not $report.identical -and ($null -eq $report.relative_rmse -or $report.relative_rmse -ge 1e-3)) { exit 1 }
