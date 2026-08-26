$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$env:UV_CACHE_DIR = Join-Path $projectRoot ".uv-cache"

function Assert-LastExitCode {
    param([string]$Step)
    if ($LASTEXITCODE -ne 0) {
        throw "$Step failed with exit code $LASTEXITCODE."
    }
}

Push-Location $projectRoot
try {
    uv sync --locked --extra dev --extra hf --extra service --extra viz --extra documents --extra multimodal
    Assert-LastExitCode "uv sync"
    uv run python -c "import sys; assert sys.version_info[:3] == (3, 11, 16), sys.version; print(sys.version)"
    Assert-LastExitCode "Python version check"
    uv run ruff format --check .
    Assert-LastExitCode "Ruff format check"
    uv run ruff check .
    Assert-LastExitCode "Ruff lint"
    uv run mypy
    Assert-LastExitCode "mypy"
    uv run python evals/validate_splits.py
    Assert-LastExitCode "Agent evaluation inventories"
    uv run python evals/run_agent_eval.py `
        --predictions evals/fixtures/qwen25_0_5b_dev_smoke_predictions.jsonl `
        --case-id da-001 --case-id ms-011 --case-id rt-001 `
        --case-id ts-001 --case-id ts-d-001
    Assert-LastExitCode "Evaluator fixture replay"
    uv run pytest -q --cov=minigpt --cov=hf_lora --cov=agent `
        --cov=minigpt_service --cov=evals --cov-report=term-missing `
        --cov-report=json:coverage.json
    Assert-LastExitCode "pytest"
    uv run python scripts/check_critical_coverage.py coverage.json
    Assert-LastExitCode "Critical module coverage"
}
finally {
    Pop-Location
}
