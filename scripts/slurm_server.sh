#!/usr/bin/env bash
#SBATCH -p 3090-gcondo,gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH -J continuous-batching
#SBATCH -o results/logs/%x_%j.out
#SBATCH -e results/logs/%x_%j.err

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

bash scripts/run_gpu.sh \
    --model-name "${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}" \
    --host "${HOST:-127.0.0.1}" \
    --port "${PORT:-8000}" \
    --device cuda \
    --dtype "${DTYPE:-bfloat16}" \
    --mode "${MODE:-continuous}" \
    --max-batch-size "${MAX_BATCH_SIZE:-16}" \
    --static-batch-size "${STATIC_BATCH_SIZE:-8}" \
    --max-seq-len "${MAX_SEQ_LEN:-2048}" \
    --kv-cache-budget-tokens "${KV_CACHE_BUDGET_TOKENS:-32768}" \
    --metrics-path "${METRICS_PATH:-results/raw/server.jsonl}"
