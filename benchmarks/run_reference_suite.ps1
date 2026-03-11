param(
    [ValidateSet("quick", "standard", "strict")]
    [string]$Mode = "standard",
    [switch]$EnforceGate,
    [switch]$WriteBaseline,
    [switch]$SkipTests,
    [switch]$SkipE2E
)

$ErrorActionPreference = "Stop"

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

$python = "c:/Users/mickg/smart-light/.venv/Scripts/python.exe"
$script = "benchmarks/reference_suite.py"

$args = @($script, "--mode", $Mode)
if ($EnforceGate) { $args += "--enforce-gate" }
if ($WriteBaseline) { $args += "--write-baseline" }
if ($SkipTests) { $args += "--skip-tests" }
if ($SkipE2E) { $args += "--skip-e2e" }

& $python @args
exit $LASTEXITCODE
