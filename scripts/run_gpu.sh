#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source "${VENV_PATH:-$HOME/continuous-batching.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be set in the environment." >&2
    exit 1
fi

python scripts/gpu_preflight.py --minimum-vram-gb "${MINIMUM_VRAM_GB:-20}"
mkdir -p results/raw results/logs

exec continuous-batching-server "$@"
