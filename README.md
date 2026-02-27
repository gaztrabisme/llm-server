# llm-server

Local LLM inference server using llama.cpp, optimized for MoE model offloading on a single-GPU consumer workstation.

## What This Is

A production-ready setup for running **Qwen3.5-35B-A3B** (Mixture-of-Experts, ~3B active params per token) on an **RTX 5080 16GB** via llama.cpp with partial expert offloading. Achieves **~75 tok/s** generation speed at Q4_K_M quantization with only +2.1% perplexity loss vs the Q8_0 reference.

Includes a full benchmarking framework, Docker builds for Blackwell GPUs, and 6 sessions of documented optimization experiments.

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 5080 16GB GDDR7 (Blackwell, sm_120) |
| CPU | AMD Ryzen 9 9950X (32 threads) |
| RAM | 128 GB DDR5-4800 |
| CUDA | 13.1, driver 590.48.01 |

## Quick Start

### 1. Download the model

```bash
# Using huggingface-cli (install via: pip install huggingface-hub)
huggingface-cli download \
  unsloth/Qwen3.5-35B-A3B-GGUF \
  Qwen3.5-35B-A3B-Q4_K_M.gguf \
  --local-dir ./models
```

### 2. Build the Docker image

```bash
# Pin to a known-good llama.cpp commit (recommended)
docker build \
  -f docker/Dockerfile.llama-cpp \
  --build-arg LLAMA_CPP_REF=b8149 \
  -t llm-server/llama-cpp:latest-fit \
  docker/
```

### 3. Run the server

```bash
docker compose --profile llama-cpp up
```

Or run directly:

```bash
docker run --gpus all --ipc host \
  -v ./models:/models \
  -p 8080:8080 \
  llm-server/llama-cpp:latest-fit \
  -m /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 \
  --fit on \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

### 4. Query the API

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

The server exposes an OpenAI-compatible API at `localhost:8080`:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/messages` (Anthropic Messages API)

## Performance

### Benchmark Results

All configs: 20 threads, 65K context, `--no-mmap`, KV cache q8_0.

| Config | Quant | Strategy | tok/s | VRAM |
|--------|-------|----------|-------|------|
| **fit-nobatch** | **Q4_K_M** | **`--fit on`, no -b/-ub** | **74.7** | **14.6 GB** |
| C7 | Q4_K_M | `--n-cpu-moe 24` (manual) | 69.6 | 14.9 GB |
| MXFP4_MOE | MXFP4 | `--fit on`, no -b/-ub | 49.5 | 14.5 GB |
| Q4_K_L | Q4_K_L | `--fit on`, no -b/-ub | 41.4 | 14.5 GB |
| C4r | Q8_0 | `--fit on` (auto) | 40.5 | 14.7 GB |
| C1 | Q8_0 | full offload | 35.7 | 8.1 GB |
| 27B dense | Q4_K_M | `--fit on` | 7.4 | 14.1 GB |

### Quantization Quality

WikiText-2 perplexity and KL divergence vs Q8_0 reference:

| Quant | PPL | vs Q8_0 | Mean KLD | Same Top-1 % |
|-------|-----|---------|----------|--------------|
| Q8_0 (reference) | 6.5342 | — | — | — |
| **Q4_K_M** | **6.6688** | **+2.1%** | **0.028** | **92.4%** |
| Q4_K_L (bartowski) | 6.6125 | +1.2% | 0.018 | 94.2% |
| MXFP4_MOE | ~5.96 | ~-0.6%* | 0.037 | 91.0% |
| UD-Q4_K_XL | 7.1702 | +9.7% | 0.109 | 86.2% |

*Partial evaluation (40 chunks) due to memory leak in MXFP4 dequant path.

### Key Findings

- **KV cache q8_0 is a free lunch**: < 0.4% PPL difference, +12-38% throughput
- **`--fit on` without `-b/-ub` is fastest**: batch flags consume VRAM that `--fit` needs for expert layers
- **Q4_K_M is the best quant for 16GB VRAM**: alternatives are either slower (Q4_K_L -44%, MXFP4 -34%) or worse quality (UD-Q4_K_XL +9.7% PPL)
- **MoE >> dense on consumer hardware**: 35B-A3B MoE is 10x faster than 27B dense AND has better quality
- **UD-Q4_K_XL is NOT recommended** for MoE models: 3.9x worse KLD than Q4_K_M
- **Speculative decoding**: no compatible draft model (vocab mismatch), ngram self-speculation provides no benefit

## Project Structure

```
llm-server/
├── CLAUDE.md                     # Detailed project docs and decisions
├── docker-compose.yml            # One-command server startup
├── docker/
│   ├── Dockerfile.llama-cpp      # Mainline llama.cpp (CUDA 12.8, sm_120)
│   └── Dockerfile.ik-llama-cpp   # ik_llama.cpp fork (deprecated)
├── scripts/
│   ├── bench.sh                  # Benchmark runner (4 workloads × N runs)
│   ├── run-experiment.sh         # Multi-step experiment orchestrator
│   ├── compare-results.py        # Parse & compare benchmark JSONs
│   ├── quant-quality.sh          # PPL + KLD quality evaluation
│   └── download-model.sh         # HuggingFace model downloader
├── configs/                      # 40+ env files for different experiments
│   ├── llama-cpp-baseline.env
│   ├── llama-cpp-s006-e4-fit-nobatch.env  # Current winner
│   └── ...
├── benchmarks/                   # Results (gitignored)
│   ├── *.json                    # Speed benchmark results
│   ├── perplexity/               # PPL logs + WikiText-2 dataset
│   └── kl-divergence/            # KLD base logits + comparison logs
├── models/                       # GGUF files (gitignored)
└── docs/dev/                     # 6 sessions of optimization research
    ├── session-index.md
    ├── 001-research-and-setup/
    ├── 002-infra-and-benchmark/
    ├── 003-optimization-sweep/
    ├── 004-speedup-investigation/
    ├── 005-qwen35-migration/
    └── 006-community-followup/
```

## Benchmarking

Run a speed benchmark:

```bash
./scripts/bench.sh llama-cpp s006-e4-fit-nobatch
```

Run a full experiment (PPL + KLD + speed):

```bash
./scripts/run-experiment.sh e3
```

Compare results:

```bash
python3 scripts/compare-results.py benchmarks/
```

## Key Flags Explained

| Flag | Purpose |
|------|---------|
| `--fit on` | Auto-split model between GPU and CPU based on available VRAM |
| `-fa on` | Flash attention (required for KV cache quantization) |
| `-ctk q8_0 -ctv q8_0` | KV cache quantization — free throughput gain |
| `-t 20` | CPU threads (optimal for 32-core Ryzen 9 9950X) |
| `--no-mmap` | Load full model into RAM upfront for consistent performance |
| `--jinja` | Enable Jinja2 chat templates |
| `-c 65536` | Context length (native max for Qwen3.5, extendable to 262K with YaRN) |

**Do NOT use** `-b 4096 -ub 4096` with `--fit on` — batch buffers consume VRAM that `--fit` needs for expert layer allocation.

## Optimization History

| Session | Date | What | Result |
|---------|------|------|--------|
| 001 | Feb 17 | Research & setup | Hardware analysis, llama.cpp evaluation |
| 002 | Feb 22 | Infrastructure + A/B benchmark | llama.cpp wins over ik_llama fork, 22 tok/s baseline |
| 003 | Feb 24 | Optimization sweep | New flags negligible, rebuild regression detected |
| 004 | Feb 25 | Speedup investigation | 20 threads optimal (+27%), speculative decoding research |
| 005 | Feb 25 | Model migration (Qwen3.5-35B-A3B) | 3.2x speedup to ~70 tok/s, Q4_K_M validated |
| 006 | Feb 26-27 | Community follow-up (7 experiments) | fit-nobatch ~75 tok/s, all quant alternatives tested |

See `docs/dev/` for detailed findings from each session.

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 5080, should work on any GPU with enough VRAM)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- For Blackwell GPUs (RTX 50-series): build from source with CUDA 12.8+ for native sm_120 support

## License

This is a personal infrastructure project. The llama.cpp server it wraps is MIT-licensed. Model weights are subject to their respective licenses (Qwen3.5: Apache 2.0).
