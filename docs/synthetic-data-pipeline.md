# Synthetic Data Pipeline: JD-CV Pair Generation

Reference document for downstream projects that need to generate synthetic job description and CV data at scale using the local llm-server.

## Pipeline Overview

```
Job Titles (bulk) → JDs (JSON/XML) → CVs at varied match levels (JSON/XML) → Match Assessments
```

Each stage produces structured output (JSON or XML). The pipeline generates JD-CV pairs with controlled match quality (strong match, partial match, weak match, no match) for training matching/ranking models.

## Infrastructure

### Server

- **Model**: Qwen3.5-35B-A3B Q4_K_M (~20 GB)
- **Engine**: llama.cpp via Docker (`llm-server/llama-cpp:latest`)
- **API**: OpenAI-compatible at `http://localhost:8080/v1/chat/completions`
- **Base throughput**: ~70 tok/s single stream (Q4_K_M, `--n-cpu-moe 24`, RTX 5080 16GB)

### Hardware

| Component | Spec |
|-----------|------|
| GPU | RTX 5080 16GB GDDR7 (PCIe 5.0 x16) |
| CPU | AMD Ryzen 9 9950X (32 threads) |
| RAM | 128 GB DDR5-4800 |

## Token Budget Per Pair

| Step | Output tokens (est.) | Input tokens (est.) | Notes |
|------|---------------------|---------------------|-------|
| Job title | ~20 | ~100 | Batch 100 titles per prompt |
| JD (JSON/XML) | 400-600 | ~200 | One JD per request |
| CV (JSON/XML) | 600-1000 | ~400 | One CV per request, match level in prompt |
| Match assessment | 150-300 | ~800 | JD + CV in context |
| **Total per pair** | **~1200-1900** | **~1400** | |

Rough estimate: **~1500 output tokens per JD-CV pair**.

## Throughput Estimates for 100K Pairs

Total output tokens: ~150M

| Configuration | Aggregate tok/s | Wall time | Notes |
|---------------|----------------|-----------|-------|
| 1 slot, baseline | 70 | ~25 days | Current production config |
| 4 slots (`-np 4`) | 200-250 | 7-9 days | Recommended starting point |
| 4 slots + speculative decoding | 300-400 | 4-6 days | Needs draft model setup |
| 8 slots (`-np 8`) | 350-450 | 4-5 days | Requires reduced context per slot |
| 8 slots + speculation | 500-600 | 3-4 days | Best case |

All estimates assume 24/7 operation. Real throughput depends on prompt complexity and output length variance.

## Server Launch Configurations

### Standard (single stream, max quality)

Current production config. Use for testing and small batches.

```bash
docker run --gpus all --ipc host \
  -v /path/to/models:/models:ro \
  -p 8080:8080 \
  llm-server/llama-cpp:latest \
  -m /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 -ngl 999 --n-cpu-moe 24 \
  -fa on -t 20 -b 4096 -ub 4096 \
  --no-mmap --jinja -ctk q8_0 -ctv q8_0
```

### Batch generation (4 parallel slots)

Recommended for the data pipeline. 4 concurrent requests, 8K context per slot (plenty for JD/CV generation).

```bash
docker run --gpus all --ipc host \
  -v /path/to/models:/models:ro \
  -p 8080:8080 \
  llm-server/llama-cpp:latest \
  -m /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 32768 -ngl 999 --n-cpu-moe 24 \
  -np 4 \
  -fa on -t 20 -b 4096 -ub 4096 \
  --no-mmap --jinja -ctk q8_0 -ctv q8_0
```

### High throughput (8 parallel slots, reduced context)

Maximum throughput for bulk generation. 4K context per slot.

```bash
docker run --gpus all --ipc host \
  -v /path/to/models:/models:ro \
  -p 8080:8080 \
  llm-server/llama-cpp:latest \
  -m /models/Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 32768 -ngl 999 --n-cpu-moe 24 \
  -np 8 \
  -fa on -t 20 -b 4096 -ub 4096 \
  --no-mmap --jinja -ctk q8_0 -ctv q8_0
```

## Optimization Recommendations

### 1. Reduce context per slot

JDs and CVs don't need 65K context. 4K-8K per slot is sufficient. This frees VRAM for more parallel slots. Total context (`-c`) is shared across all slots: `-c 32768 -np 8` = 4096 tokens per slot.

### 2. Batch job titles

Generate 50-100 job titles per prompt instead of one at a time. This amortizes prompt processing cost and produces more tokens per request.

### 3. Pipeline parallelism

Run stages concurrently: while slots 1-4 generate JDs for batch N, slots 5-8 generate CVs for batch N-1 (already-completed JDs). The downstream project should implement a queue/worker pattern.

### 4. Structured output via JSON schema

llama.cpp supports grammar-constrained generation. Pass a JSON schema to guarantee valid output and skip impossible tokens:

```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "model": "qwen3.5",
    "messages": [{"role": "user", "content": "Generate a job description for a Senior Backend Engineer"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "job_description",
        "schema": {
          "type": "object",
          "properties": {
            "title": {"type": "string"},
            "department": {"type": "string"},
            "requirements": {"type": "array", "items": {"type": "string"}},
            "responsibilities": {"type": "array", "items": {"type": "string"}}
          },
          "required": ["title", "department", "requirements", "responsibilities"]
        }
      }
    }
  }'
```

### 5. Speculative decoding (not yet configured)

JSON/XML output is highly predictable (repeated keys, tags, structural tokens). Speculative decoding with a small draft model (Qwen3-1.7B) could give 1.5-2x speedup on this workload. Requires downloading the draft model and adding `--model-draft` flag. This is not yet tested on our setup — see llm-server CLAUDE.md "Ready to Test" section.

### 6. N-gram speculation (zero cost)

Add `--spec-type ngram-simple` to the server launch command. Uses patterns from existing context to predict next tokens. Particularly effective when generating many documents with the same JSON/XML schema since keys and tags repeat. No additional VRAM or model required.

## API Usage

The server exposes an OpenAI-compatible API. Any OpenAI SDK or HTTP client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

response = client.chat.completions.create(
    model="qwen3.5",
    messages=[{"role": "user", "content": "Generate a job description..."}],
    temperature=0.7,
    max_tokens=1024,
)
```

For concurrent generation, use `asyncio` with `httpx` or a thread pool with the OpenAI SDK. The server handles queueing internally via its slot system — just send requests concurrently up to the `-np` limit.

## Thinking Mode

Qwen3.5 has thinking mode enabled by default. For data generation, you may want to disable it to save tokens (thinking tokens count toward output but aren't useful for structured data generation). Add to the prompt or system message:

```
/no_think
```

Or set in the API call:
```json
{"messages": [{"role": "system", "content": "/no_think\nYou are a data generation assistant..."}]}
```

This can significantly reduce output token count and speed up generation.

## Quality Notes

- **Q4_K_M quality**: PPL 6.6688 on WikiText-2, only +2.1% above Q8_0 baseline (6.5342). Negligible quality loss for structured data generation.
- **Do NOT use UD-Q4_K_XL**: PPL 7.1702 (+9.7%), confirmed worse than standard Q4_K_M on MoE architectures.
- For maximum quality at lower throughput, switch to Q8_0 with `--fit on` (~40 tok/s). See llm-server CLAUDE.md for the alternative config.

## Recommended Test Plan

Before committing to 100K pairs:

1. **Generate 10 pairs end-to-end** with 1 slot — verify output quality and format
2. **Generate 100 pairs** with 4 slots — measure actual throughput and validate concurrency
3. **Enable n-gram speculation** — compare throughput with and without
4. **Scale to 1000 pairs** — verify pipeline stability over hours
5. **Extrapolate** — confirm throughput matches estimates before launching full 100K run
