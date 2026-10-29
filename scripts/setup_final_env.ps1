param([string]$Python = "python")

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$environmentPath = Join-Path $projectRoot ".venv-final"
$environmentPython = Join-Path $environmentPath "Scripts\python.exe"
$lockPath = Join-Path $projectRoot "requirements-final-env.lock.txt"

$pythonVersion = & $Python -c "import platform; print(platform.python_version())"
if ($LASTEXITCODE -ne 0 -or $pythonVersion.Trim() -ne "3.13.14") {
    throw "The final environment requires Python 3.13.14; found '$pythonVersion'. Pass -Python with the matching interpreter."
}

if (-not (Test-Path -LiteralPath $environmentPath)) {
    & $Python -m venv $environmentPath
    if ($LASTEXITCODE -ne 0) { throw "Virtual environment creation failed." }
}
if (-not (Test-Path -LiteralPath $environmentPython)) {
    throw "$environmentPath already exists without a usable Python interpreter. It has not been overwritten."
}
$venvConfig = Get-Content -LiteralPath (Join-Path $environmentPath "pyvenv.cfg")
if (-not ($venvConfig -match '^include-system-site-packages\s*=\s*false\s*$')) {
    throw "The existing environment must not inherit global site-packages."
}
$environmentVersion = & $environmentPython -c "import platform; print(platform.python_version())"
if ($environmentVersion.Trim() -ne "3.13.14") {
    throw "The existing environment uses Python $environmentVersion instead of 3.13.14."
}

$buildTemp = Join-Path $environmentPath ".build-temp"
$packageCache = Join-Path $environmentPath ".pip-cache"
New-Item -ItemType Directory -Path $buildTemp,$packageCache -Force | Out-Null
$savedEnvironment = @{}
foreach ($variableName in @("TMP", "TEMP", "TMPDIR", "PIP_CACHE_DIR")) {
    $savedEnvironment[$variableName] = [Environment]::GetEnvironmentVariable($variableName, "Process")
}
try {
    # Keep wheel build/rename operations on the same workspace volume.
    $env:TMP = $buildTemp
    $env:TEMP = $buildTemp
    $env:TMPDIR = $buildTemp
    $env:PIP_CACHE_DIR = $packageCache
    & $environmentPython -m pip install --upgrade "pip==26.1.2" "setuptools==80.9.0" "wheel==0.45.1" "packaging==25.0"
    if ($LASTEXITCODE -ne 0) { throw "Pinned build-tool installation failed." }
    & $environmentPython -m pip install --no-build-isolation --requirement $lockPath
    if ($LASTEXITCODE -ne 0) { throw "Final environment installation failed." }
    & $environmentPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "Dependency consistency check failed." }
} finally {
    foreach ($variableName in $savedEnvironment.Keys) {
        [Environment]::SetEnvironmentVariable($variableName, $savedEnvironment[$variableName], "Process")
    }
}

Write-Host "Final environment ready: $environmentPython"
Write-Host "Next: & '$environmentPython' scripts/check_final.py"
