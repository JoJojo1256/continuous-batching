#!/usr/bin/env bash
#SBATCH --job-name=continuous-batching-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=02:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
mkdir -p results/logs results/raw

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda "${PYTHON_MODULE:-python/3.11.11-5e66}"
source "${VENV_PATH:-$HOME/continuous-batching.venv}/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be exported before submitting this job." >&2
    exit 1
fi

port="${PORT:-8000}"
job_id="${SLURM_JOB_ID:-manual}"
server_log="results/logs/continuous-batching-server_${job_id}.log"
metrics_path="results/raw/continuous-batching-server_${job_id}.jsonl"
client_path="results/raw/continuous-batching-smoke_${job_id}.jsonl"

bash scripts/run_gpu.sh \
    --model-name "${MODEL_NAME:-meta-llama/Llama-3.1-8B-Instruct}" \
    --host 127.0.0.1 \
    --port "$port" \
    --device cuda \
    --dtype "${DTYPE:-bfloat16}" \
    --mode continuous \
    --max-batch-size "${MAX_BATCH_SIZE:-8}" \
    --max-seq-len "${MAX_SEQ_LEN:-512}" \
    --kv-cache-budget-tokens "${KV_CACHE_BUDGET_TOKENS:-4096}" \
    --metrics-path "$metrics_path" \
    >"$server_log" 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

server_url="http://127.0.0.1:$port"
for _ in $(seq 1 120); do
    if ! kill -0 "$server_pid" 2>/dev/null; then
        echo "Server exited before becoming ready; see $server_log" >&2
        exit 1
    fi
    if python -c \
        "import httpx; httpx.get('$server_url/metrics', timeout=2).raise_for_status()" \
        >/dev/null 2>&1; then
        break
    fi
    sleep 5
done

python -c \
    "import httpx; httpx.get('$server_url/metrics', timeout=2).raise_for_status()" \
    >/dev/null

continuous-batching-loadgen \
    --url "$server_url" \
    --prompts bench/prompts.txt \
    --output "$client_path" \
    --requests "${REQUESTS:-8}" \
    --arrival closed \
    --concurrency "${CONCURRENCY:-2}" \
    --length-workload bimodal \
    --min-output-tokens "${MIN_OUTPUT_TOKENS:-8}" \
    --max-output-tokens "${MAX_OUTPUT_TOKENS:-32}" \
    --warmup-requests "${WARMUP_REQUESTS:-2}" \
    --trials 3

echo "Smoke benchmark complete: $client_path"
