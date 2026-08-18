#!/usr/bin/env bash
#SBATCH --job-name=continuous-batching-compare
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
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
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}127.0.0.1,localhost"
export no_proxy="$NO_PROXY"

job_id="${SLURM_JOB_ID:-manual}"
port="${PORT:-8000}"
server_url="http://127.0.0.1:$port"
server_pid=""

stop_server() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill "$server_pid"
        wait "$server_pid" 2>/dev/null || true
    fi
    server_pid=""
}
trap stop_server EXIT

wait_for_server() {
    local server_log="$1"
    for _ in $(seq 1 120); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Server exited before becoming ready; see $server_log" >&2
            exit 1
        fi
        if python -c \
            "import httpx; response = httpx.get('$server_url/metrics', timeout=2); response.raise_for_status(); assert 'request_count' in response.json()" \
            >/dev/null 2>&1; then
            return
        fi
        sleep 5
    done
    echo "Server did not become ready within 10 minutes; see $server_log" >&2
    exit 1
}

for mode in sequential static continuous; do
    server_log="results/logs/continuous-batching-compare_${job_id}_${mode}.log"
    metrics_path="results/raw/continuous-batching-compare_${job_id}_${mode}_server.jsonl"

    bash scripts/run_gpu.sh \
        --model-name "${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}" \
        --host 127.0.0.1 \
        --port "$port" \
        --device cuda \
        --dtype "${DTYPE:-bfloat16}" \
        --mode "$mode" \
        --max-batch-size "${MAX_BATCH_SIZE:-8}" \
        --static-batch-size "${STATIC_BATCH_SIZE:-8}" \
        --max-seq-len "${MAX_SEQ_LEN:-512}" \
        --kv-cache-budget-tokens "${KV_CACHE_BUDGET_TOKENS:-4096}" \
        --metrics-path "$metrics_path" \
        >"$server_log" 2>&1 &
    server_pid=$!
    wait_for_server "$server_log"

    for concurrency in ${CONCURRENCIES:-1 2 4 8}; do
        for workload in ${WORKLOADS:-uniform bimodal}; do
            output="results/raw/continuous-batching-compare_${job_id}_${mode}_c${concurrency}_${workload}.jsonl"
            continuous-batching-loadgen \
                --url "$server_url" \
                --prompts bench/prompts.txt \
                --output "$output" \
                --requests "${REQUESTS:-32}" \
                --arrival closed \
                --concurrency "$concurrency" \
                --length-workload "$workload" \
                --min-output-tokens "${MIN_OUTPUT_TOKENS:-8}" \
                --max-output-tokens "${MAX_OUTPUT_TOKENS:-32}" \
                --warmup-requests "${WARMUP_REQUESTS:-8}" \
                --trials 3
        done
    done

    stop_server
    sleep 2
done

echo "Comparison complete: results/raw/continuous-batching-compare_${job_id}_*.jsonl"
