<#
.SYNOPSIS
    Generates MSVC import libraries (.lib) for TensorRT DLLs from their own
    export tables.

.DESCRIPTION
    NVIDIA's TensorRT SDK download bundles nvinfer_10.lib / nvonnxparser_10.lib,
    but those import libs are not named as distributable in the SLA — only the
    runtime DLLs and the public headers are. Rather than host or redistribute
    NVIDIA's .lib files, this script derives an equivalent import lib directly
    from a DLL's own export table (dumpbin /exports -> a .def listing every
    exported name -> lib.exe /def), a standard, purely mechanical technique
    for linking against a DLL that didn't ship its own import lib. The
    resulting .lib exposes the same symbols the DLL already exports — nothing
    is copied from NVIDIA's SDK archive.

    Locates dumpbin.exe / lib.exe via vswhere (falls back to PATH, e.g. inside
    a Developer PowerShell / VsDevCmd session).

.PARAMETER Dll
    One or more DLL paths to process (e.g. nvinfer_10.dll, nvonnxparser_10.dll).

.PARAMETER OutDir
    Directory to write <name>.lib (and the intermediate .def) into.

.EXAMPLE
    ./make-import-libs.ps1 `
        -Dll engine\deps\tensorrt_libs\nvinfer_10.dll, engine\deps\tensorrt_libs\nvonnxparser_10.dll `
        -OutDir _experiments\2026-09-11-mm3-speed\trt-sdk\generated-libs
#>
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Dll,

    [Parameter(Mandatory = $true)]
    [string]$OutDir
)

$ErrorActionPreference = "Stop"

function Find-VcTool([string]$toolName) {
    # 1) Already on PATH (Developer shell / VsDevCmd).
    $onPath = Get-Command $toolName -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }

    # 2) Ask vswhere for the latest VS install, then find the tool under its
    #    MSVC toolchain (pick the highest MSVC version present).
    $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "$toolName not on PATH and vswhere.exe not found at $vswhere"
    }
    $vsInstall = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vsInstall) {
        throw "vswhere found no VS install with the VC x86/x64 tools component"
    }
    $msvcRoot = Join-Path $vsInstall "VC\Tools\MSVC"
    $verDir = Get-ChildItem $msvcRoot -Directory | Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1
    if (-not $verDir) { throw "No MSVC toolchain found under $msvcRoot" }
    $toolPath = Join-Path $verDir.FullName "bin\Hostx64\x64\$toolName"
    if (-not (Test-Path $toolPath)) { throw "$toolName not found at $toolPath" }
    return $toolPath
}

$dumpbin = Find-VcTool "dumpbin.exe"
$libExe = Find-VcTool "lib.exe"
Write-Host "dumpbin: $dumpbin"
Write-Host "lib:     $libExe"

New-Item -ItemType Directory -Force $OutDir | Out-Null

$results = @()

foreach ($dllPath in $Dll) {
    if (-not (Test-Path $dllPath)) { throw "DLL not found: $dllPath" }
    $dllName = [System.IO.Path]::GetFileName($dllPath)
    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($dllPath)
    $defPath = Join-Path $OutDir "$baseName.def"
    $libPath = Join-Path $OutDir "$baseName.lib"

    Write-Host ""
    Write-Host "== $dllName =="

    # dumpbin /exports lines look like:
    #   ordinal hint RVA      name
    #         1    0 0239E8A0 ?DeserializeContextFromFile@cask6_myelin@@YA...@Z
    # Some entries are forwarders ("(forwarded to ...)") or have no RVA
    # (data-only, "RVA" column blank) — capture the name token regardless.
    $exportsRaw = & $dumpbin /exports $dllPath 2>&1
    $names = New-Object System.Collections.Generic.List[string]
    foreach ($line in $exportsRaw) {
        # Match "<ordinal> <hint> <rva-or-blank> <name> [forwarding info]"
        if ($line -match '^\s*\d+\s+[0-9A-Fa-f]+\s+[0-9A-Fa-f]*\s+(\S+)') {
            $names.Add($Matches[1])
        }
    }
    if ($names.Count -eq 0) {
        throw "Parsed zero exports from $dllName — dumpbin output format may have changed"
    }
    Write-Host "Parsed $($names.Count) exported names"

    $defLines = @("LIBRARY $baseName", "EXPORTS") + $names
    Set-Content -Path $defPath -Value $defLines -Encoding ASCII

    & $libExe "/def:$defPath" "/machine:x64" "/out:$libPath" | Out-Null
    if (-not (Test-Path $libPath)) { throw "lib.exe did not produce $libPath" }

    $size = (Get-Item $libPath).Length
    Write-Host "Wrote $libPath ($size bytes)"

    $results += [PSCustomObject]@{
        Dll         = $dllName
        Lib         = $libPath
        ExportCount = $names.Count
        Size        = $size
    }
}

$results
