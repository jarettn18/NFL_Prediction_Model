# Weekly betting agent: grade last week's paper bets, then pick this week's bets against opening lines.
# Run by Windows Task Scheduler (see README). Needs the ODDS_API_KEY user environment variable.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8  # so PowerShell decodes Python's UTF-8 table borders
$env:COLUMNS = "200"  # keep rich tables from wrapping in the log
if (-not $env:ODDS_API_KEY) {
    $env:ODDS_API_KEY = [Environment]::GetEnvironmentVariable("ODDS_API_KEY", "User")
}

New-Item -ItemType Directory -Force (Join-Path $root "logs") | Out-Null
$log = Join-Path $root ("logs\weekly_{0:yyyy-MM-dd}.log" -f (Get-Date))
$nflpred = Join-Path $root ".venv\Scripts\nflpred.exe"

# Plain-text log: stderr ErrorRecords (e.g. Python warnings) become ordinary lines,
# and box-drawing table borders become ASCII.
# Box-drawing characters (U+2500-U+257F), built at runtime so this file stays ASCII for PowerShell 5.1.
$boxH = "[{0}{1}{2}]" -f [char]0x2500, [char]0x2501, [char]0x2550
$boxV = "[{0}{1}{2}]" -f [char]0x2502, [char]0x2503, [char]0x2551
$boxAny = "[{0}-{1}]" -f [char]0x2500, [char]0x257F
filter ConvertTo-PlainText {
    "$_" -replace $boxH, '-' -replace $boxV, '|' -replace $boxAny, '+'
}

"=== $(Get-Date -Format s) grading ledger ===" | Out-File -Append -Encoding utf8 $log
& $nflpred ledger 2>&1 | ConvertTo-PlainText | Out-File -Append -Encoding utf8 $log

# ledger already refreshed the data, so bets reuses it.
"=== $(Get-Date -Format s) picking bets ===" | Out-File -Append -Encoding utf8 $log
& $nflpred bets --no-refresh 2>&1 | ConvertTo-PlainText | Out-File -Append -Encoding utf8 $log
$code = $LASTEXITCODE

"=== $(Get-Date -Format s) done (exit $code) ===" | Out-File -Append -Encoding utf8 $log
exit $code
