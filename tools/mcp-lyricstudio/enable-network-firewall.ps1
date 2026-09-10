#Requires -RunAsAdministrator
param(
    [Parameter(Mandatory = $true)][string]$Address,
    [int]$Port = 3012
)
$ErrorActionPreference = 'Stop'
$adapter = Get-NetIPAddress -AddressFamily IPv4 -IPAddress $Address -ErrorAction Stop
$ruleName = "HOT-Step-Discussions-$Address-$Port"
if (Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue) {
    Write-Output "Rule $ruleName already exists."
    exit 0
}
New-NetFirewallRule -Name $ruleName -DisplayName "HOT-Step discussions LAN ($Address port $Port)" `
    -Direction Inbound -Action Allow -Protocol TCP -LocalAddress $Address -LocalPort $Port `
    -RemoteAddress LocalSubnet -InterfaceAlias $adapter.InterfaceAlias -Profile Any | Out-Null
Write-Output "Allowed TCP $Port from the local subnet on $($adapter.InterfaceAlias)."
