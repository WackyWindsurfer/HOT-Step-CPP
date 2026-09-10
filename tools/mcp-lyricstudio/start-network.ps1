param(
    [string]$Address = '127.0.0.1',
    [int]$Port = 3012,
    [string]$NodePath = 'node',
    [string]$RuntimeDirectory = $PSScriptRoot,
    [string]$TokenFile = "$env:LOCALAPPDATA/HOT-Step/discussion-network/token.txt",
    [switch]$Background
)
$ErrorActionPreference = 'Stop'
$major = (& $NodePath -p 'process.versions.node.split(".")[0]')
if ($LASTEXITCODE -ne 0 -or $major -ne '22') { throw 'Select a Node 22 executable with -NodePath and dependencies built for Node 22.' }
$tokenParent = Split-Path -Parent $TokenFile
New-Item -ItemType Directory -Force -Path $tokenParent | Out-Null
if (-not (Test-Path -LiteralPath $TokenFile)) {
    $bytes = New-Object byte[] 32
    $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
    try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
    [IO.File]::WriteAllText($TokenFile, [Convert]::ToBase64String($bytes))
}
$env:HOTSTEP_COLLAB_HOST = $Address
$env:HOTSTEP_COLLAB_PORT = "$Port"
$env:HOTSTEP_COLLAB_ALLOWED_HOSTS = $Address
$env:HOTSTEP_COLLAB_TOKEN_FILE = (Resolve-Path -LiteralPath $TokenFile).Path
$env:HOTSTEP_COLLAB_DB = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../data/collaboration.db'))
$nodeArgs = @('--preserve-symlinks', '--preserve-symlinks-main', '--import', 'tsx', 'src/network-server.ts')
if ($Background) {
    if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
        throw "Port $Port already has a listener. Check it before starting another discussion server."
    }
    $process = Start-Process -FilePath $NodePath -ArgumentList $nodeArgs -WorkingDirectory $RuntimeDirectory -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $tokenParent 'stdout.log') -RedirectStandardError (Join-Path $tokenParent 'stderr.log')
    $process.Id | Set-Content -LiteralPath (Join-Path $tokenParent 'server.pid')
    Start-Sleep -Milliseconds 800
    if ($process.HasExited) { throw "Server exited. Read $(Join-Path $tokenParent 'stderr.log')." }
    Write-Output "Discussion server PID $($process.Id)"
} else {
    Push-Location $RuntimeDirectory
    try { & $NodePath @nodeArgs } finally { Pop-Location }
}
Write-Output "Viewer: http://${Address}:${Port}/"
Write-Output "MCP: http://${Address}:${Port}/mcp"
Write-Output "Access token file: $TokenFile"
