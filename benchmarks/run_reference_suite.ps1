param(
    [ValidateSet("quick", "standard", "strict")]
    [string]$Mode = "standard",
    [switch]$EnforceGate,
    [switch]$WriteBaseline
)

$ErrorActionPreference = "Stop"

$env:TF_ENABLE_ONEDNN_OPTS = "0"
$env:TF_CPP_MIN_LOG_LEVEL = "2"

$python = "c:/Users/mickg/smart-light/.venv/Scripts/python.exe"
$script = "benchmarks/reference_suite.py"

if ($EnforceGate -and $WriteBaseline) {
    & $python $script --mode $Mode --enforce-gate --write-baseline
}
elseif ($EnforceGate) {
    & $python $script --mode $Mode --enforce-gate
}
elseif ($WriteBaseline) {
    & $python $script --mode $Mode --write-baseline
}
else {
    & $python $script --mode $Mode
}
exit $LASTEXITCODE
