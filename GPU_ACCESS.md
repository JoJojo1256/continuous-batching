# GPU access

The target benchmark requires an NVIDIA CUDA GPU. FPGA instances and graphics-only
hosts are not substitutes.

## Brown Oscar

Connect from PowerShell and clone or update the repository on the login node:

```powershell
ssh <brown-username>@ssh.ccv.brown.edu
```

```bash
git clone https://github.com/JoJojo1256/continuous-batching.git
cd continuous-batching
```

If the clone already exists, run `git pull --ff-only` instead. Do not run model
workloads on the login node.

Use a GPU interactive allocation for setup and debugging:

```bash
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh
export HF_TOKEN="<read-only-token>"
export HF_HOME="$HOME/scratch/hf_cache"
bash scripts/run_gpu.sh \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --mode continuous --max-batch-size 16
```

For the first recorded end-to-end smoke benchmark, submit:

```bash
sbatch scripts/slurm_smoke.sh
```

The smoke job defaults to the public `Qwen/Qwen2.5-7B-Instruct` model so model
access does not block GPU validation. To run the gated Llama model instead,
accept its license, create a read-only Hugging Face token, and submit with
`MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct` and `HF_TOKEN` exported.

After the smoke job succeeds, run the single-GPU scheduling comparison:

```bash
sbatch scripts/slurm_compare.sh
```

The comparison loads one server at a time for sequential, static, and continuous
modes. It covers concurrency 1, 2, 4, and 8 with uniform and bimodal output
lengths, 32 requests per measured trial, eight warmups, and three measured trials.
This keeps multiple request waves queued even at concurrency 8 so continuous slot
refill is exercised. Override
`CONCURRENCIES`, `WORKLOADS`, `REQUESTS`, or the model and server settings with
exported environment variables.

For a standalone server run, submit:

```bash
export HF_TOKEN="<read-only-token>"
sbatch scripts/slurm_server.sh
```

Override `MODEL_NAME`, `MODE`, `MAX_BATCH_SIZE`, `PORT`, `HF_HOME`, or `VENV_PATH`
with exported environment variables. The default 8B model should be attempted first
on a 24 GiB Ampere GPU. The exploratory account uses the general `gpu` partition and
has four CPU cores. Store model weights and raw results under Oscar scratch storage,
copy important results off Oscar, and keep them out of the repository.

## Standalone Linux CUDA host

This path supports an approved Azure NVIDIA VM or another Ubuntu CUDA machine:

```bash
git clone https://github.com/JoJojo1256/continuous-batching.git
cd continuous-batching
bash env/setup_linux_gpu.sh
export HF_TOKEN="<read-only-token>"
bash scripts/run_gpu.sh \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --mode continuous
```

The preflight exits before model download when CUDA is unavailable or the selected GPU
has less than 20 GiB of VRAM. Set `MINIMUM_VRAM_GB` to change that threshold.

## Azure requirements

Azure NP-series VMs are FPGA-backed and cannot run this CUDA/PyTorch project. Request
an NVIDIA quota family instead:

- For comfortable headroom, use one A100 VM such as `Standard_NC24ads_A100_v4`.
- For a lower-cost 24 GiB option, use one full A10 VM such as
  `Standard_NV36ads_A10_v5`.

Before provisioning, confirm that the subscription owner permits the workload, that
you have Contributor access to a dedicated resource group, that the region has quota
for the exact NVIDIA VM family, and that the hourly cost and shutdown plan are
approved.

Always use `az vm deallocate` when an Azure benchmark session ends. Shutting down only
inside Linux may continue billing allocated compute.
