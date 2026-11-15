# Starts the live loop (through twin.supervisor) in its OWN PowerShell window, so closing an editor,
# a chat or another terminal cannot take it down. Refuses to start if a loop is already running
# (only ONE live process per team).
#
#   .\start_live.ps1                                   # the arms below, holdout 0
#   .\start_live.ps1 -Arms 'v5tcs=...:100'         # override the arms
#   .\start_live.ps1 -Dry                              # same window setup, sends nothing
#
# Output also goes to artifacts\logs\live_<date>.log. Stop it with Ctrl+C in its window.
param(
    [string]$Arms = 'v5tcs=v5-hybrid+m2-intent+m3-live+trend2+cold8+seedtrend:100',
    [switch]$Dry
)

$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'twin\.(live|supervisor)' }
if ($running) {
    Write-Host 'A live loop is already running; not starting another one:' -ForegroundColor Red
    $running | Select-Object ProcessId, CreationDate, CommandLine | Format-Table -AutoSize -Wrap
    exit 1
}

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$dryArg = ''
if ($Dry) { $dryArg = ' --dry' }
$title = 'TWIN LIVE - DO NOT CLOSE'
if ($Dry) { $title = 'TWIN DRY RUN' }
$command = "`$host.UI.RawUI.WindowTitle = '$title'; " +
           "`$env:TWIN_ARMS = '$Arms'; `$env:TWIN_HOLDOUT = '0'; " +
           "& '$python' -m twin.supervisor$dryArg"

Start-Process powershell.exe -WorkingDirectory $PSScriptRoot -ArgumentList '-NoExit', '-NoProfile', '-Command', $command
Write-Host "Started in a new window ('$title'), arms: $Arms"
