# llm-server

Local LLM inference server using llama.cpp, optimized for MoE model offloading on a single-GPU consumer workstation.

## What This Is

A production-ready setup for running **Qwen3.5-35B-A3B** (Mixture-of-Experts, ~3B active params per token) on an **RTX 5080 16GB** via llama.cpp with partial expert offloading. Achieves **~50 tok/s** generation speed at UD-Q4_K_XL quantization with only +0.9% perplexity loss vs the Q8_0 reference.

Includes a smart proxy with sampling presets and cold start, a benchmarking framework (`llm-bench`), Docker builds for Blackwell GPUs, and 12 sessions of documented optimization experiments.

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
# Activate venv with huggingface-hub
source .venv/bin/activate
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('unsloth/Qwen3.5-35B-A3B-GGUF',
                'Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf',
                local_dir='./models')
"
```

### 2. Build the Docker image

```bash
docker build \
  -f docker/Dockerfile.llama-cpp \
  --build-arg LLAMA_CPP_REF=b8322 \
  -t llm-server/llama-cpp:b8322 \
  docker/
```

### 3. Run via proxy (recommended)

The proxy handles cold start, idle shutdown, and sampling presets:

```bash
# Install proxy deps (one-time)
source .venv/bin/activate
pip install fastapi uvicorn httpx pyyaml

# Start proxy — it will auto-start the llama-server container on first request
python proxy.py
```

### 4. Query the API

```bash
# Basic request (no preset — pass-through)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'

# With sampling preset
curl "http://localhost:8080/v1/chat/completions?mode=coding" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Write a Python fibonacci function"}]
  }'
```

### Alternative: Run without proxy

```bash
# Direct Docker Compose (llama-server on port 8081)
docker compose --profile llama-cpp up

# Or direct Docker run
docker run --gpus all --ipc host \
  -v ./models:/models:ro \
  -p 8080:8080 \
  llm-server/llama-cpp:b8322 \
  -m /models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  -c 65536 \
  --fit on \
  --fit-target 0 \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

## Proxy

`proxy.py` is a FastAPI reverse proxy that sits in front of llama-server on port 8080:

- **Sampling presets**: Select via `?mode=<name>` or `X-Mode` header. Presets defined in `presets.yaml`
- **Cold start**: Auto-starts the container on first request, polls health, forwards when ready (~10-20s)
- **Idle shutdown**: Stops container after 30min of no requests to free VRAM
- **Streaming**: Full SSE pass-through for `stream: true`

### Available Presets

| Mode | Temperature | top_p | presence_penalty | max_tokens | Use Case |
|------|-------------|-------|------------------|------------|----------|
| `thinking` | 1.0 | 0.95 | 1.5 | 32768 | General reasoning (default) |
| `coding` | 0.6 | 0.95 | 0.0 | 32768 | Precise code generation |
| `vision` | 1.0 | 0.95 | 1.5 | 81920 | Multimodal (needs bigger token budget) |
| `instruct` | 0.7 | 0.8 | 1.5 | 32768 | Non-thinking mode |

Presets are defaults — any parameter you specify in your request overrides the preset.

## API

The server exposes an OpenAI-compatible API:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/messages` (Anthropic Messages API)

Proxy-specific endpoints:
- `/health` — proxy + backend status
- `/presets` — list available sampling presets

## Performance

### Production Config (S012)

**~50 tok/s** with UD-Q4_K_XL (Unsloth Dynamic 2.0) + `--fit on --fit-target 0`, 14.7 GB VRAM.

### Quantization Quality

| Quant | Size | PPL | vs Q8_0 | Mean KLD | Same Top-1 % |
|-------|------|-----|---------|----------|--------------|
| Q8_0 (reference) | 36.9 GB | 6.5342 | -- | -- | -- |
| **UD-Q4_K_XL** | **22.2 GB** | **6.5918** | **+0.9%** | **0.0137** | **94.7%** |
| Q4_K_M (Unsloth) | 21.2 GB | 6.6053 | +1.1% | 0.0192 | 93.5% |
| AesSedai Q4_K_M | ~21 GB | 6.3949 | -2.1% | 0.0095 | 95.7% |

### Key Findings

- **KV cache q8_0 is a free lunch**: < 0.3% PPL difference across all context lengths (4k-128k)
- **`--fit on --fit-target 0`**: Maximizes GPU layers by eliminating default VRAM margin. Safe on dedicated inference
- **Do NOT use** `-b/-ub` batch flags with `--fit on` — they hurt TG by ~35%
- **20 threads optimal** for Ryzen 9 9950X (Session 004 sweep)
- **MoE >> dense**: 35B-A3B MoE is 10x faster than 27B dense with better quality

## Project Structure

```
llm-server/
├── proxy.py                     # Smart proxy (presets, cold start, idle shutdown)
├── presets.yaml                 # Sampling presets from Qwen3.5 model card
├── docker-compose.yml           # Container orchestration
├── docker/
│   └── Dockerfile.llama-cpp     # llama.cpp CUDA build (sm_120)
├── llm_bench/                   # Python benchmarking framework
│   ├── commands/                # speed, quality, eval, compare, report, setup
│   └── core/                    # config, docker, gpu, hardware, results, stats
├── configs/                     # 77+ env files for experiments
│   └── llama-cpp-s012-production.env  # Current production
├── scripts/                     # Legacy benchmark scripts
├── benchmarks/                  # Results (gitignored)
├── models/                      # GGUF files (gitignored)
└── docs/dev/                    # 12 sessions of optimization research
```

## Benchmarking

```bash
# Install the CLI
pip install -e .

# Speed benchmark
llm-bench speed --env configs/llama-cpp-s012-production.env

# Quick speed test
llm-bench speed --env configs/llama-cpp-s012-production.env --quick

# Compare results
llm-bench compare

# Full setup info
llm-bench setup
```

## Key Flags

| Flag | Purpose |
|------|---------|
| `--fit on` | Auto-split model between GPU and CPU based on available VRAM |
| `--fit-target 0` | Eliminate default 1024 MiB VRAM margin (safe on dedicated box) |
| `-fa on` | Flash attention (required for KV cache quantization) |
| `-ctk q8_0 -ctv q8_0` | KV cache quantization -- free throughput gain |
| `-t 20` | CPU threads (optimal for 32-core Ryzen 9 9950X) |
| `--no-mmap` | Load full model into RAM upfront for consistent performance |
| `--jinja` | Enable Jinja2 chat templates |
| `-c 65536` | Context length (native max 262K, extendable to 1M via YaRN) |

## Requirements

- NVIDIA GPU with CUDA support (tested on RTX 5080, should work on any GPU with enough VRAM)
- Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- Python 3.12+ with venv for proxy and benchmarking tools
- For Blackwell GPUs (RTX 50-series): build from source with CUDA 12.8+ for native sm_120 support

## License

This is a personal infrastructure project. The llama.cpp server it wraps is MIT-licensed. Model weights are subject to their respective licenses (Qwen3.5: Apache 2.0).
