# Continuous Batching from First Principles

A correctness-first continuous-batching inference server and performance study built
with PyTorch, Hugging Face Transformers, and FastAPI. The implementation focuses on
iteration-level scheduling: finished sequences leave the active batch immediately and
queued requests take their slots on the next decoding iteration. The accompanying
load generator, sweep runner, and analysis tools make static-versus-continuous
throughput and latency comparisons reproducible.

The final Brown Oscar A40 comparison measured 2,304 requests across 24
configurations. At concurrency 8, continuous batching delivered 52.41-53.74 output
tokens/second versus 33.46-34.11 sequentially, while reducing p99 latency relative
to static batching. See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full
architecture, methodology, results, debugging history, and interpretation.

## Architecture

`src/continuous_batching` separates scheduling from model execution so every policy
uses the same tokenization, forward pass, and sampling behavior:

```text
FastAPI -> bounded request queue -> BatchScheduler
                                  -> ModelEngine.prefill/decode_step
                                  -> fixed-slot KVCacheManager
                                  -> per-request Sampler
                                  -> JSONL metrics
```

The server supports three scheduling modes:

- `sequential` admits one request at a time.
- `static` admits a fixed group and waits for every sequence in the group to finish
  before admitting more work.
- `continuous` performs FCFS admission every decoding iteration, evicts EOS or
  max-token completions, and fills the newly available slots immediately.

Admission reserves each sequence's full prompt-plus-output token capacity. If the
configured cache budget cannot accommodate the next request, that request remains
queued rather than overallocating. The cache manager provides deterministic fixed-slot
ownership, capacity accounting, free-list reuse, and full state clearing on eviction.

Ragged batches use explicit attention masks and position IDs. Device and dtype
selection are isolated in `ModelEngine`. Greedy output is token-identical across
sequential, static, and continuous modes regardless of batch composition or arrival
order: the scheduler controls *when* a sequence advances, never *how* its token is
computed. Sampled generation uses a request-local seeded generator so results are also
independent of batch order.

The portable Transformers reference engine currently recomputes each active
sequence's visible token prefix on every iteration while tracking logical cache
occupancy. It intentionally does not implement paged attention, fused KV-cache
kernels, prefix caching, CUDA graphs, distributed execution, or multi-GPU serving.

## Install

Python 3.11 is the target runtime. Install the pinned environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Tests use tiny local fake models and require neither a GPU nor network access:

```bash
python -m pytest
```

They cover ragged masks and positions, greedy equality across scheduling modes and
arrival orders, reproducible per-request sampling, slot reuse without state leakage,
budget-limited queuing, immediate continuous replacement, one unique output per
input, deterministic workload generation, and in-process synchronous and streaming
FastAPI behavior.

## Run the server

The model is swappable through the command line:

```bash
continuous-batching-server \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device cuda --dtype bfloat16 \
  --mode continuous --max-batch-size 16 \
  --max-seq-len 2048 --kv-cache-budget-tokens 32768 \
  --metrics-path results/raw/server.jsonl
```

Use `--mode sequential` as a single-request baseline or `--mode static
--static-batch-size 8` for a fixed-batch comparison. The API exposes:

- `POST /generate_sync` for one JSON response.
- `POST /generate` for server-sent token events.
- `GET /metrics` for throughput, configurable SLO goodput, p50/p99 latency, and the
  active-batch trace, histogram, and time average.

The request queue is bounded. Overload returns HTTP 429. Invalid requests and
sequences that exceed the configured context or total cache-token budget return HTTP
422.

## Benchmark methodology

Keep prompts fixed, discard warmup requests, and run at least three measured trials.
The load generator supports closed-loop concurrency and open-loop Poisson arrivals,
plus uniform, bimodal, and seeded sampled output-length distributions:

```bash
continuous-batching-loadgen \
  --url http://127.0.0.1:8000 \
  --prompts bench/prompts.txt \
  --arrival closed --concurrency 8 \
  --length-workload bimodal \
  --warmup-requests 4 --trials 3 \
  --output results/raw/continuous_c8_bimodal.jsonl
```

Analyze one or more raw JSONL files:

```bash
continuous-batching-analyze \
  results/raw/continuous_c8_bimodal.jsonl \
  --output-dir results/continuous_c8_bimodal
```

The analyzer derives summary JSON and CSV plus latency histograms entirely from raw
records. Request and server records include arrival, first-token and finish
timestamps, TTFT, TPOT, end-to-end latency, output token counts, active batch size,
configuration, package versions, git state, CUDA runtime, and hardware.

For a standard comparison sweep, start one static server and one continuous server,
then run:

```bash
continuous-batching-sweep \
  --static-url http://127.0.0.1:8001 \
  --continuous-url http://127.0.0.1:8002 \
  --prompts bench/prompts.txt \
  --output-dir results/raw/sweep
```

The sweep covers both modes at concurrency 1, 2, 4, 8, 16, and 32 across all three
output-length workloads. Raw JSONL is gitignored; derived summaries and figures can be
generated from retained experiment records.

## GPU workflows

[`GPU_ACCESS.md`](GPU_ACCESS.md) documents Brown Oscar and standalone Linux CUDA
setup. On a Linux GPU host, the convenience wrapper performs the CUDA/VRAM preflight
before starting the server:

```bash
bash env/setup_linux_gpu.sh
export HF_TOKEN="<read-only-token>"
bash scripts/run_gpu.sh \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --mode continuous --max-batch-size 16
```

On Brown Oscar, `scripts/slurm_smoke.sh` starts a continuous-batching server,
waits for it to become ready, and records a small three-trial load-generator run
with the public `Qwen/Qwen2.5-7B-Instruct` model by default. See
[`GPU_ACCESS.md`](GPU_ACCESS.md) for login, setup, and submission commands.
After that gate passes, `scripts/slurm_compare.sh` records a compact
sequential-versus-static-versus-continuous comparison while keeping only one
model resident on the GPU at a time.

Generate the matrix summary and figure from a completed comparison:

```bash
continuous-batching-compare-analyze \
  results/raw/continuous-batching-compare_<job-id>_*_c*.jsonl \
  --output-dir results/compare_<job-id>
```

Accept the model's license before downloading it. Keep Hugging Face tokens, model
weights, raw measurements, scheduler logs, and profiler output out of the repository.
