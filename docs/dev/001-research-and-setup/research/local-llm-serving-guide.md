# Serving Local LLMs for Application Use Cases (February 2026)

**Target Setup**: Single NVIDIA RTX 5080 (16GB VRAM), 128GB RAM, Linux
**Use Cases**: Classification, Synthetic Data Generation, Chatbot, Agentic Workflows

---

## Table of Contents

1. [llama-server (llama.cpp Built-in Server)](#1-llama-server-llamacpp-built-in-server)
2. [Model Routing / Multiplexing](#2-model-routing--multiplexing)
3. [Application Integration Patterns](#3-application-integration-patterns)
4. [Docker Architecture](#4-docker-architecture)
5. [Performance Tuning for Serving](#5-performance-tuning-for-serving)
6. [Recommended Architecture for Your Setup](#6-recommended-architecture-for-your-setup)

---

## 1. llama-server (llama.cpp Built-in Server)

### 1.1 OpenAI-Compatible API Endpoints

llama-server provides comprehensive OpenAI API compatibility, making it a drop-in replacement for applications built against the OpenAI SDK. The following endpoints are supported:

**OpenAI-Compatible Endpoints:**
| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completions (ChatGPT-style) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Text embeddings (configurable pooling: none/mean/cls/last/rank) |
| `GET /v1/models` | List available models |
| `POST /v1/responses` | Response API |

**Anthropic-Compatible Endpoints:**
| Endpoint | Description |
|---|---|
| `POST /v1/messages` | Anthropic Messages API |
| `POST /v1/messages/count_tokens` | Token counting |

**Native llama.cpp Endpoints:**
| Endpoint | Description |
|---|---|
| `GET /health` | Server health check |
| `POST /completion` | Native text completion |
| `POST /tokenize` | Text to tokens |
| `POST /detokenize` | Tokens to text |
| `POST /apply-template` | Apply chat template to conversation |
| `POST /embedding` | Native embeddings endpoint (all pooling types) |
| `POST /reranking` | Document reranking by relevance |
| `POST /infill` | Code infilling |
| `GET /props` | Server properties (chat template info, etc.) |
| `GET /slots` | Monitor slot processing state |
| `GET /metrics` | Prometheus-compatible metrics |
| `POST /slots/{id}` | Save/restore/erase prompt cache per slot |
| `GET/POST /lora-adapters` | LoRA adapter management |

Sources: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [Unsloth deployment docs](https://unsloth.ai/docs/basics/inference-and-deployment/llama-server-and-openai-endpoint)

### 1.2 Concurrent Request Handling and Continuous Batching

**Continuous batching** is enabled by default (`--cont-batching`) and is the key feature that allows llama-server to handle multiple concurrent requests efficiently.

**How it works:**
- Without continuous batching, even with multiple parallel slots, the server answers only one request at a time
- With continuous batching, new requests can begin processing as soon as a slot becomes available, without waiting for all current requests to complete
- Tokens from different requests are batched together in a single forward pass through the model

**Key configuration flags:**

```bash
llama-server \
  -m model.gguf \
  -c 8192 \              # Total context size (shared across all slots)
  -np 4 \                # Number of parallel slots (concurrent requests)
  -b 2048 \              # Logical batch size (max tokens per batch, default 2048)
  -ub 512 \              # Physical batch size (actual GPU batch, default 512)
  -t 8 \                 # CPU threads for generation
  -tb 8 \                # CPU threads for batch/prompt processing
  --threads-http 4 \     # HTTP request processing threads
  -ngl 99 \              # GPU layers (99 or "all" for full offload)
  -fa auto \             # Flash attention (on|off|auto)
  --cont-batching        # Enabled by default
```

**Context size planning for parallel slots:**
- Total context (`-c`) is divided among slots: with `-c 8192 -np 4`, each slot gets ~2048 tokens
- For 4 parallel requests each needing up to 4096 tokens of context, set `-c 16384 -np 4`
- Add extra KV space (~10-20%) for fragmentation when using continuous batching

**Performance characteristics:**
- llama.cpp maintains flat, consistent throughput regardless of concurrent load (unlike vLLM which scales throughput with concurrency)
- At low concurrency (1-4 users), llama.cpp provides excellent single-request latency
- Best suited for single-user to moderate concurrent loads on single-GPU setups

Sources: [Batching explanation discussion](https://github.com/ggml-org/llama.cpp/discussions/4130), [Continuous batching discussion](https://github.com/ggml-org/llama.cpp/discussions/10170), [Red Hat vLLM vs llama.cpp](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)

### 1.3 Embeddings Endpoint

llama-server supports embeddings through both native and OpenAI-compatible endpoints:

```bash
# Start server with an embedding model
llama-server -m nomic-embed-text-v1.5.Q8_0.gguf --port 8081 --embedding

# Or with a chat model that also supports embeddings
llama-server -m model.gguf --embedding
```

**OpenAI-compatible usage:**
```bash
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "nomic-embed"}'
```

**Pooling options:** `none` (per-token), `mean`, `cls`, `last`, `rank`
**Normalization:** Euclidean norm applied automatically for pooled embeddings

Embedding models are lightweight (typically 100-400MB for GGUF quantized versions) and can run alongside or separately from your generation models.

Sources: [Embeddings tutorial](https://github.com/ggml-org/llama.cpp/discussions/7712), [llama-server-embeddings](https://github.com/fabiomatricardi/llama-server-embeddings)

### 1.4 Multiple Model Serving (Router Mode)

As of late 2025, llama.cpp server has **built-in router mode** for multi-model management:

```bash
# Start in router mode (no model specified upfront)
llama-server --models-dir ./models --models-max 2 -c 8192 -ngl 99

# Or with a custom models directory
llama-server --models-dir /path/to/gguf/files
```

**Key features:**
- **Auto-discovery**: Scans `--models-dir` or `~/.cache/llama.cpp` for GGUF files
- **On-demand loading**: Models load when first requested via the `model` field in API requests
- **LRU eviction**: When `--models-max` is reached, the least-recently-used model unloads automatically
- **Multi-process architecture**: Each model runs in its own process (crash isolation)
- **Web UI dropdown**: Built-in UI shows available models for selection

**Configuration flags:**

| Flag | Description |
|---|---|
| `--models-dir PATH` | Directory containing GGUF files |
| `--models-max N` | Max simultaneously loaded models (default: 4) |
| `--no-models-autoload` | Disable auto-loading; require explicit API calls |
| `--models-preset config.ini` | Per-model settings override |

**Model presets (config.ini):**
```ini
[classification-model]
model = /models/qwen2.5-3b-instruct-q5_k_m.gguf
ctx-size = 2048
temp = 0.0

[generation-model]
model = /models/qwen2.5-14b-instruct-q4_k_m.gguf
ctx-size = 8192
temp = 0.7
```

**API usage:**
```bash
# Chat with a specific model
curl http://localhost:8080/v1/chat/completions \
  -d '{"model": "qwen2.5-14b-instruct-q4_k_m", "messages": [...]}'

# Manually load/unload
curl -X POST http://localhost:8080/models/load \
  -d '{"model": "my-model.gguf"}'
curl -X POST http://localhost:8080/models/unload \
  -d '{"model": "my-model.gguf"}'
```

**Limitation for 16GB VRAM:** With `--models-max 2`, you can keep a small model (e.g., 3B Q5) and a medium model (e.g., 8B Q4) loaded simultaneously. Larger models require swapping.

Sources: [HuggingFace blog: Model Management](https://huggingface.co/blog/ggml-org/model-management-in-llamacpp), [llama.cpp router mode](https://medium.com/coding-nexus/llama-cpp-server-gets-router-mode-switch-models-on-the-fly-without-restarting-d3a159dd567a)

### 1.5 Structured Output / JSON Mode / Grammar-Constrained Generation

llama.cpp supports three approaches to structured output:

**1. JSON Schema (recommended for most use cases):**
```bash
# Server-side default
llama-server -m model.gguf --json-schema '{"type":"object","properties":{"label":{"type":"string","enum":["positive","negative","neutral"]}}}'

# Per-request via API
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "messages": [{"role": "user", "content": "Classify: great product"}],
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "classification",
        "schema": {
          "type": "object",
          "properties": {
            "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "confidence": {"type": "number"}
          },
          "required": ["label", "confidence"]
        }
      }
    }
  }'
```

**2. GBNF Grammar (for non-JSON structured output):**
```bash
# Force output to match a custom grammar
llama-server -m model.gguf --grammar 'root ::= "yes" | "no"'
```

**3. JSON Mode (simple, no schema enforcement):**
```bash
curl http://localhost:8080/v1/chat/completions \
  -d '{
    "messages": [...],
    "response_format": {"type": "json_object"}
  }'
```

**Performance notes:**
- Grammar-constrained generation filters token probabilities at each step, guaranteeing syntactic correctness
- Lazy grammars allow deferring grammar activation until specific conditions are met
- XGrammar optimization (2025) achieves up to 100x speedup by splitting vocabulary into context-independent (~99%) and context-dependent (~1%) token sets
- Minimal overhead for simple schemas; complex nested schemas may add slight latency

Sources: [Grammar and Structured Output](https://deepwiki.com/ggml-org/llama.cpp/7.3-grammar-and-structured-output), [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/), [llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)

### 1.6 Throughput vs. Latency Configuration Tradeoffs

| Optimize For | Settings | Effect |
|---|---|---|
| **Low latency (chatbot)** | `-np 1 -c 4096 -fa on` | Single slot, all resources to one request |
| **Throughput (batch jobs)** | `-np 8 -c 16384 -b 4096 -ub 1024` | Many slots, larger batches |
| **Balanced** | `-np 4 -c 8192 -b 2048 -ub 512` | Default-like configuration |
| **Memory efficient** | `-ctk q8_0 -ctv q8_0` | Quantized KV cache, ~50% less cache VRAM |
| **Max context** | `-c 32768 -ctk q4_0 -ctv q4_0 -np 1` | Aggressive cache quantization |

**Speculative decoding** for latency-sensitive applications:
```bash
llama-server \
  -m large-model.gguf \
  -md draft-model.gguf \    # Small draft model (e.g., 0.5B-1B)
  --draft 16 \              # Tokens to draft per step
  -ngl 99 -ngld 99          # Both models on GPU
```

This can yield 2-4x speedup for "high draftability" prompts (e.g., code generation, structured output).

Sources: [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), [Speculative decoding discussion](https://github.com/ggml-org/llama.cpp/discussions/10466)

---

## 2. Model Routing / Multiplexing

### 2.1 Running Different Models for Different Tasks

With 16GB VRAM, you cannot run multiple large models simultaneously. The strategy is:

**Option A: Built-in Router Mode (simplest)**
- Use llama-server's router mode with `--models-max` to auto-swap
- LRU eviction handles transitions automatically
- Best for development and low-traffic applications

**Option B: llama-swap (production-grade)**
- External Go binary that sits in front of llama-server instances
- More configuration control, groups, TTL, aliases

**Option C: LiteLLM Proxy (unified gateway)**
- Python-based proxy that normalizes APIs across providers
- Can route to local llama-server AND cloud APIs as fallback
- Supports load balancing, rate limiting, cost tracking

### 2.2 llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) is a lightweight proxy that manages multiple llama-server instances:

```yaml
# config.yaml
models:
  classifier:
    cmd: llama-server --port ${PORT} -m /models/qwen2.5-3b-q5_k_m.gguf -c 2048 -ngl 99
    ttl: 300  # Unload after 5 minutes of inactivity

  generator:
    cmd: llama-server --port ${PORT} -m /models/qwen2.5-14b-q4_k_m.gguf -c 8192 -ngl 99
    ttl: 600

  embedder:
    cmd: llama-server --port ${PORT} -m /models/nomic-embed-text-v1.5-q8_0.gguf --embedding
    ttl: 0  # Never auto-unload

# Groups allow concurrent models
groups:
  always-on:
    swap: false  # Members run concurrently
    members:
      - embedder

  gpu-heavy:
    swap: true   # Only one active at a time
    members:
      - classifier
      - generator
```

**Key features:**
- **Model aliasing**: Map friendly names (e.g., `gpt-4o-mini`) to local models
- **Groups**: `swap: false` keeps models running concurrently; `swap: true` (default) swaps on request
- **TTL**: Auto-unload idle models to free VRAM
- **Filters**: Rewrite request components before upstream
- **Docker support**: Nightly CUDA images available
- **Hot-reload**: Watch config file for changes with `-watch-config`

```bash
# Run llama-swap
docker run -it --rm --runtime nvidia -p 9292:8080 \
  -v /path/to/models:/models \
  -v /path/to/config.yaml:/app/config.yaml \
  ghcr.io/mostlygeek/llama-swap:cuda
```

Sources: [llama-swap GitHub](https://github.com/mostlygeek/llama-swap), [KDnuggets tutorial](https://www.kdnuggets.com/how-to-run-multiple-llms-locally-using-llama-swap-on-a-single-server)

### 2.3 LiteLLM Proxy

[LiteLLM](https://github.com/BerriAI/litellm) provides a unified OpenAI-compatible gateway:

```yaml
# litellm_config.yaml
model_list:
  - model_name: classifier
    litellm_params:
      model: openai/qwen2.5-3b
      api_base: http://localhost:8080/v1
      api_key: sk-xxx

  - model_name: generator
    litellm_params:
      model: openai/qwen2.5-14b
      api_base: http://localhost:8080/v1
      api_key: sk-xxx

  - model_name: gpt-4o-fallback
    litellm_params:
      model: gpt-4o
      api_key: os.environ/OPENAI_API_KEY

router_settings:
  routing_strategy: simple-shuffle
  num_retries: 2
  fallbacks:
    - classifier: [gpt-4o-fallback]
```

```bash
litellm --config litellm_config.yaml --port 4000
```

**When to use LiteLLM over llama-swap:**
- You need cloud API fallback when local models are overloaded
- You want unified cost tracking across local + cloud
- You need rate limiting, guardrails, or logging
- You are integrating with frameworks that expect a standard OpenAI endpoint

Sources: [LiteLLM docs](https://docs.litellm.ai/docs/simple_proxy), [LiteLLM routing](https://docs.litellm.ai/docs/routing)

### 2.4 Practical Model Sizing for RTX 5080 (16GB VRAM)

| Model Size | Quantization | VRAM (approx) | Context Room | Best For |
|---|---|---|---|---|
| 1-3B | Q5_K_M | 1.5-2.5 GB | Plenty | Classification, simple extraction |
| 7-8B | Q4_K_M | 4.5-5.5 GB | 8-16K context | Chatbot, general tasks |
| 14B | Q4_K_M | 8-9 GB | 4-8K context | Generation, complex reasoning |
| 20B | Q4_K_M | 12-13 GB | 2-4K context | Best quality, limited context |
| 32B+ | Q3_K_M / Q2_K | 14-16 GB | Minimal | Not recommended for serving |
| Embedding | Q8_0 | 0.3-0.5 GB | N/A | Always-on alongside main model |

**Rule of thumb for VRAM budget:**
- Model weights: primary consumer
- KV cache: `hidden_size x context_length x 2 (K+V) x num_layers x bytes_per_element`
- For 8B model at FP16 KV, 8K context: ~2 GB KV cache
- For 8B model at Q8_0 KV, 8K context: ~1 GB KV cache
- System overhead: ~500MB

Sources: [GGUF quantization guide](https://apatero.com/blog/gguf-quantized-models-complete-guide-2025), [VRAM discussion](https://github.com/ggml-org/llama.cpp/discussions/9784), [Context VRAM analysis](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)

---

## 3. Application Integration Patterns

### 3.1 Classification

**Approach A: LLM Prompt-Based Classification (simpler, less accurate)**

Use a small model (1-3B) with structured output for direct classification:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx")

response = client.chat.completions.create(
    model="qwen2.5-3b",
    messages=[
        {"role": "system", "content": "Classify the sentiment. Respond with JSON."},
        {"role": "user", "content": "This product is amazing!"}
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "classification",
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["label", "confidence"]
            }
        }
    },
    temperature=0.0
)
```

**Approach B: Embeddings + Classifier (more accurate, recommended)**

Recent research (2025) shows embeddings outperform direct prompting by ~49.5% higher accuracy for classification tasks, with better calibration, lower latency, and lower cost.

```python
import numpy as np
from openai import OpenAI
from sklearn.linear_model import LogisticRegression

# Embedding client (lightweight model, always running)
embed_client = OpenAI(base_url="http://localhost:8081/v1", api_key="sk-xxx")

def get_embedding(text):
    response = embed_client.embeddings.create(
        input=text, model="nomic-embed-text"
    )
    return response.data[0].embedding

# Train classifier on labeled examples
train_embeddings = [get_embedding(text) for text in train_texts]
clf = LogisticRegression().fit(train_embeddings, train_labels)

# Classify new text
new_embedding = get_embedding("This product is terrible")
prediction = clf.predict([new_embedding])
```

**Recommendation:** Use embeddings + classifier for production classification. Use LLM prompting for quick prototyping or when you lack labeled training data.

Sources: [Embeddings vs Prompting for Classification](https://arxiv.org/html/2504.04277), [llama.cpp embeddings tutorial](https://github.com/ggml-org/llama.cpp/discussions/7712)

### 3.2 Synthetic Data Generation

**Batching strategy for throughput:**

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx")

async def generate_one(prompt: str, semaphore: asyncio.Semaphore):
    async with semaphore:
        response = await client.chat.completions.create(
            model="qwen2.5-14b",
            messages=[
                {"role": "system", "content": "Generate a realistic customer review."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.9,
            max_tokens=512
        )
        return response.choices[0].message.content

async def batch_generate(prompts: list[str], max_concurrent: int = 4):
    """Send requests concurrently; llama-server handles batching internally."""
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [generate_one(p, semaphore) for p in prompts]
    return await asyncio.gather(*tasks)

# Match max_concurrent to server's -np (parallel slots) value
results = asyncio.run(batch_generate(prompts, max_concurrent=4))
```

**Server configuration for synthetic data throughput:**
```bash
llama-server \
  -m qwen2.5-14b-instruct-q4_k_m.gguf \
  -c 16384 \        # Enough context for 4 slots x 4096 tokens each
  -np 4 \           # 4 parallel generation slots
  -b 4096 \         # Larger logical batch
  -ub 1024 \        # Larger physical batch
  -ngl 99 \         # Full GPU offload
  -fa on \          # Flash attention for efficiency
  -ctk q8_0 \       # Quantized KV cache to save VRAM
  -ctv q8_0
```

**Tips for maximizing throughput:**
- Match your async concurrency to the server's `-np` value
- Use consistent system prompts across requests to benefit from KV cache reuse
- Set `cache_prompt: true` in requests (enabled by default)
- For large batches, consider running overnight with a larger model swapped in

Sources: [Throughput optimization](https://itecsonline.com/post/vllm-vs-ollama-vs-llama.cpp-vs-tgi-vs-tensort), [CPU inference benchmarks](https://tiffena.me/blog/llm-cpu-only-inference-benchmark-llama.cpp-server-flags/)

### 3.3 Chatbot

**Context management best practices:**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx")

class ChatSession:
    def __init__(self, system_prompt: str, max_history: int = 20):
        self.system_prompt = system_prompt
        self.history = []
        self.max_history = max_history

    def chat(self, user_message: str) -> str:
        self.history.append({"role": "user", "content": user_message})

        # Trim history to avoid exceeding context window
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-self.max_history:])

        response = client.chat.completions.create(
            model="qwen2.5-8b",
            messages=messages,
            stream=True,  # Stream for responsive UX
            temperature=0.7,
            max_tokens=1024
        )

        full_response = ""
        for chunk in response:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                print(token, end="", flush=True)  # Real-time output

        self.history.append({"role": "assistant", "content": full_response})
        return full_response
```

**Server configuration for chatbot:**
```bash
llama-server \
  -m qwen2.5-8b-instruct-q4_k_m.gguf \
  -c 8192 \         # Good context for conversation
  -np 1 \           # Single user = single slot for lowest latency
  -ngl 99 \
  -fa on \
  --cache-prompt     # Enabled by default; reuses system prompt KV cache
```

**Streaming:** Set `"stream": true` in the API request. llama-server sends Server-Sent Events (SSE) in the same format as OpenAI, so any OpenAI SDK client handles streaming natively.

**KV cache reuse for chatbot:** When consecutive requests share the same system prompt prefix, the server reuses the cached KV values for that prefix. This means the system prompt is processed once and reused across turns, significantly reducing time-to-first-token for subsequent messages.

Sources: [Context management](https://discuss.huggingface.co/t/correct-way-to-pass-context-to-llama-cpp-server/95531), [llama.cpp streaming guide](https://blog.steelph0enix.dev/posts/llama-cpp-guide/)

### 3.4 Agentic Workflows (Tool Calling)

llama-server supports OpenAI-style function calling when started with the `--jinja` flag:

```bash
llama-server \
  --jinja \                         # Enable Jinja template processing for tool calls
  -fa on \
  -m qwen2.5-7b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 8192
```

**Natively supported models for tool calling:**
- Llama 3.1 / 3.2 / 3.3 (with built-in tools: wolfram_alpha, web_search, code_interpreter)
- Functionary v3.1 / v3.2
- Hermes 2 / 3
- Qwen 2.5 and Qwen 2.5 Coder
- Mistral Nemo
- Firefunction v2
- Command R7B
- DeepSeek R1 (work-in-progress)

**Generic format:** For models not in the above list, a generic tool-calling format is used automatically (less efficient, more tokens consumed).

**API usage:**
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="sk-xxx")

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="qwen2.5-7b",
    messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a tool
if response.choices[0].finish_reason == "tool_calls":
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Call: {tool_call.function.name}({tool_call.function.arguments})")
```

**Parallel tool calling:** Enable with `"parallel_tool_calls": true` in the request for models that support it.

**Framework integration:**
- **OpenAI Python SDK**: Works directly with `base_url` pointed at llama-server
- **LangChain / LangGraph**: Full integration via `ChatOpenAI(base_url=...)`
- **llama-cpp-agent**: Dedicated framework for structured function calls with grammar-based enforcement
- **Instructor**: Works with llama-cpp-python for validated structured outputs

**Caution:** Extreme KV quantization (e.g., `-ctk q4_0`) substantially degrades tool-calling performance. Use at least `-ctk q8_0` for agentic workloads.

Sources: [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md), [Agentic coding tutorial](https://github.com/ggml-org/llama.cpp/discussions/14758), [llama-cpp-agent](https://github.com/Maximilian-Winter/llama-cpp-agent)

### 3.5 Client Libraries That Work With llama-server

| Library | Language | Notes |
|---|---|---|
| **OpenAI SDK** | Python, Node.js | Drop-in; just change `base_url`. Recommended default choice. |
| **LangChain** | Python, JS | Use `ChatOpenAI(base_url="http://localhost:8080/v1")` |
| **LangGraph** | Python | Full agentic workflows via OpenAI-compatible interface |
| **Instructor** | Python | Structured output validation with Pydantic models |
| **llama-cpp-python** | Python | Direct bindings (alternative to HTTP server approach) |
| **llama-cpp-agent** | Python | Specialized for function calling and structured output |
| **node-llama-cpp** | Node.js | Native bindings with `LlamaChatSession` for conversation |
| **LiteLLM** | Python | Unified client for multi-provider routing |

---

## 4. Docker Architecture

### 4.1 Should the LLM Server Be a Separate Container?

**Yes.** Running the LLM server as a separate container is the recommended pattern for these reasons:

1. **Resource isolation**: GPU memory management is cleaner when the LLM process is isolated
2. **Independent scaling**: Restart the LLM server without affecting application containers
3. **Model updates**: Swap models without rebuilding application images
4. **Multiple consumers**: Multiple application containers can share one LLM server
5. **Crash isolation**: If the LLM process crashes (OOM, CUDA errors), your application stays up

### 4.2 Docker Compose Patterns

**Basic: Single LLM server + application**

```yaml
# docker-compose.yml
services:
  llm-server:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models
    environment:
      - LLAMA_ARG_MODEL=/models/qwen2.5-8b-instruct-q4_k_m.gguf
      - LLAMA_ARG_CTX_SIZE=8192
      - LLAMA_ARG_N_GPU_LAYERS=99
      - LLAMA_ARG_FLASH_ATTN=true
      - LLAMA_ARG_HOST=0.0.0.0
      - LLAMA_ARG_PORT=8080
      - LLAMA_ARG_PARALLEL=4
      - LLAMA_ARG_CONT_BATCHING=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  app:
    build: ./app
    environment:
      - LLM_BASE_URL=http://llm-server:8080/v1
      - LLM_API_KEY=sk-xxx
    depends_on:
      - llm-server
    ports:
      - "3000:3000"
```

**Advanced: Multiple model servers with llama-swap**

```yaml
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:cuda
    ports:
      - "9292:8080"
    volumes:
      - ./models:/models
      - ./llama-swap-config.yaml:/app/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  embeddings:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    environment:
      - LLAMA_ARG_MODEL=/models/nomic-embed-text-v1.5-q8_0.gguf
      - LLAMA_ARG_N_GPU_LAYERS=99
      - LLAMA_ARG_EMBEDDING=true
      - LLAMA_ARG_HOST=0.0.0.0
      - LLAMA_ARG_PORT=8080
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    ports:
      - "8081:8080"
    restart: unless-stopped

  app:
    build: ./app
    environment:
      - LLM_BASE_URL=http://llama-swap:8080/v1
      - EMBED_BASE_URL=http://embeddings:8080/v1
    depends_on:
      - llama-swap
      - embeddings
```

### 4.3 GPU Sharing Between Containers

On a single RTX 5080, multiple containers can share the GPU but with caveats:

**Default behavior (no special config):**
- Docker with `--gpus all` gives each container access to the full GPU
- CUDA driver handles context switching between processes
- **No memory isolation**: containers can OOM each other
- This works fine for: one heavy model + one lightweight embedding model

**NVIDIA MPS (Multi-Process Service):**
- Allows multiple CUDA contexts to truly share GPU resources simultaneously
- Reduces context-switching overhead
- Experimental in NVIDIA device plugin (as of v0.15.0)
- Setup:
  ```bash
  # On host (not in container)
  nvidia-cuda-mps-control -d
  ```
- Best for: running a small embedding model alongside a larger generation model

**NVIDIA Time-Slicing:**
- Divides GPU time into intervals allocated to different containers
- No memory isolation (same as default)
- Primarily a Kubernetes feature via NVIDIA device plugin
- For Docker Compose on a single machine, the default CUDA behavior is effectively time-slicing already

**Practical recommendation for your setup:**
- Run the embedding model server in one container (~0.5 GB VRAM)
- Run the main generation model via llama-swap in another container (~5-13 GB VRAM)
- Both containers mount `--gpus all`; the CUDA driver handles sharing
- Total VRAM usage stays within 16 GB budget

### 4.4 Networking Between Containers

```yaml
# Docker Compose automatically creates a bridge network
# Containers reference each other by service name

# From app container:
# - LLM server: http://llm-server:8080
# - Embeddings: http://embeddings:8080
# - llama-swap: http://llama-swap:8080

# Only expose ports you need externally
# Internal container-to-container traffic uses Docker's internal DNS
```

Sources: [llama.cpp Docker docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md), [llama-cpp-docker](https://github.com/fboulnois/llama-cpp-docker), [ServiceStack production guide](https://docs.servicestack.net/ai-server/llama-server), [GPU sharing guide](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)

---

## 5. Performance Tuning for Serving

### 5.1 Prompt Caching

Prompt caching is **enabled by default** in llama-server and is one of the most impactful optimizations.

**How it works:**
1. First request: Full processing of all tokens; KV cache stored in the assigned slot
2. Subsequent requests with same prefix: Server detects the common prefix and skips reprocessing those tokens
3. Only the new suffix (changed portion) is computed

**Example impact:**
- 1000-token system prompt: processed once, reused across all requests
- Without caching: full 1000-token prefill every request
- With caching: only new user message tokens are processed

**Configuration:**
```bash
llama-server \
  --cache-prompt \              # Enabled by default
  --cache-reuse 256 \           # Min chunk size for cache reuse via KV shifting
  -sps 0.5 \                    # Slot-prompt-similarity threshold (0.0-1.0)
  --slot-save-path /tmp/slots   # Optional: persist slots to disk
```

**Slot-prompt-similarity (`-sps`):**
- Default: 0.5 (slot reused if >= 50% of prompt matches cached content)
- Set to 0.0 to disable automatic slot matching
- Use `"id_slot": N` in request body for explicit slot assignment

**Slot persistence (advanced):**
```bash
# Save a slot's KV cache to disk
curl -X POST 'http://localhost:8080/slots/0?action=save'

# Restore later (instant context reload)
curl -X POST 'http://localhost:8080/slots/0?action=restore'
```

Sources: [KV cache reuse tutorial](https://github.com/ggml-org/llama.cpp/discussions/13606), [System prompt caching](https://github.com/ggml-org/llama.cpp/discussions/8947)

### 5.2 KV Cache Management

**KV cache VRAM formula:**
```
KV_cache_bytes = num_layers x 2 (K and V) x hidden_size x context_length x bytes_per_element
```

**Example for Qwen2.5-7B (28 layers, hidden_size 3584):**
- FP16 KV, 8K context: `28 x 2 x 3584 x 8192 x 2 bytes = ~3.3 GB`
- Q8_0 KV, 8K context: `28 x 2 x 3584 x 8192 x 1 byte = ~1.6 GB`
- Q4_0 KV, 8K context: `28 x 2 x 3584 x 8192 x 0.5 bytes = ~0.8 GB`

**Quantized KV cache options:**

| Type | Bytes | Quality Impact | Recommended For |
|---|---|---|---|
| `f16` (default) | 2 | None | When VRAM allows |
| `q8_0` | 1 | Negligible | General use, best tradeoff |
| `q5_0` / `q5_1` | 0.625 | Minor | Moderate compression |
| `q4_0` / `q4_1` | 0.5 | Noticeable for tool calling | Batch jobs, simple generation |

```bash
# Recommended: q8_0 for both K and V caches
llama-server -m model.gguf -ctk q8_0 -ctv q8_0 -c 8192 -ngl 99
```

**The `-fit` flag:** Automatically adjusts parameters to fit within device memory:
```bash
llama-server -m model.gguf -fit -c 8192
```

Sources: [VRAM analysis](https://github.com/ggml-org/llama.cpp/discussions/9784), [KV cache quantization](https://github.com/ggml-org/llama.cpp/discussions/5932), [KV quantization in Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)

### 5.3 Prefill vs. Generation Optimization

LLM inference has two distinct phases:

**Prefill (prompt processing):**
- Compute-bound: processes all input tokens in parallel
- Benefits from: larger `-ub` (ubatch size), more `-tb` threads, flash attention
- Bottleneck for: long system prompts, synthetic data generation with long instructions

**Generation (token-by-token output):**
- Memory-bandwidth-bound: generates one token at a time
- Benefits from: faster VRAM, fewer memory copies, speculative decoding
- Bottleneck for: chatbot latency, long-form generation

**Optimization by phase:**
```bash
# Optimize for prefill-heavy workloads (batch generation, classification)
llama-server -m model.gguf -ub 1024 -b 4096 -tb 16 -fa on -ngl 99

# Optimize for generation-heavy workloads (chatbot, streaming)
llama-server -m model.gguf -ub 512 -np 1 -fa on -ngl 99 \
  -md draft-model.gguf --draft 16  # Speculative decoding
```

### 5.4 Context Length vs. Throughput Tradeoffs

**The fundamental tradeoff:** Longer context = more VRAM for KV cache = less room for parallel slots or larger models.

| Config | Context | Parallel Slots | Use Case |
|---|---|---|---|
| `-c 4096 -np 4` | 1024/slot | 4 concurrent | Classification, short tasks |
| `-c 8192 -np 2` | 4096/slot | 2 concurrent | Balanced chatbot |
| `-c 16384 -np 1` | 16384 | 1 at a time | Long-form generation |
| `-c 32768 -np 1 -ctk q4_0 -ctv q4_0` | 32768 | 1 at a time | Document analysis |

**Warning:** Once model + KV cache exceeds VRAM, layers spill to CPU RAM. This causes a dramatic slowdown from ~50-100 tok/s to ~2-5 tok/s. Always ensure the total fits in VRAM.

**VRAM budget for RTX 5080 (16 GB):**
```
16 GB total
 - ~0.5 GB system/CUDA overhead
 - Model weights (varies by size/quantization)
 - KV cache (varies by context/type)
 = Must all fit in ~15.5 GB
```

### 5.5 Flash Attention

Flash attention is a significant optimization that should be enabled for most use cases:

```bash
llama-server -m model.gguf -fa on  # or -fa auto (default)
```

**Benefits:**
- Reduces memory usage for attention computation
- Improves generation speed, especially for longer contexts
- Supported by most recent models
- Auto mode detects compatibility

### 5.6 Monitoring and Metrics

```bash
# Enable Prometheus metrics endpoint
llama-server -m model.gguf --metrics

# Access metrics
curl http://localhost:8080/metrics

# Check slot utilization
curl http://localhost:8080/slots  # (disable with --slots-endpoint-disable in production)

# Health check
curl http://localhost:8080/health
```

Sources: [Performance discussion](https://github.com/ggml-org/llama.cpp/discussions/15013), [NVIDIA LM Studio blog](https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/)

---

## 6. Recommended Architecture for Your Setup

### 6.1 Suggested Stack

```
                    +-------------------+
                    |   Your App(s)     |
                    |  (Python/Node.js) |
                    +--------+----------+
                             |
                    OpenAI SDK (base_url)
                             |
              +--------------+--------------+
              |                             |
    +---------v---------+      +-----------v-----------+
    |   llama-swap       |      |  Embedding Server     |
    |   (port 9292)     |      |  (port 8081)          |
    |   Routes to:      |      |  nomic-embed-text     |
    |   - classifier    |      |  Always running       |
    |   - generator     |      |  ~0.5 GB VRAM         |
    |   - chatbot       |      +-----------------------+
    |   - agent         |
    +-------------------+
```

### 6.2 Model Recommendations by Task

| Task | Model | Quantization | VRAM | Config |
|---|---|---|---|---|
| **Classification** | Qwen2.5-3B-Instruct | Q5_K_M | ~2.5 GB | `-c 2048 -np 8 -fa on --json-schema ...` |
| **Synthetic Data** | Qwen2.5-14B-Instruct | Q4_K_M | ~9 GB | `-c 8192 -np 4 -fa on -ctk q8_0` |
| **Chatbot** | Qwen2.5-7B-Instruct or Llama-3.3-8B | Q4_K_M | ~5 GB | `-c 8192 -np 2 -fa on --jinja` |
| **Agentic** | Qwen2.5-7B-Instruct | Q5_K_M | ~5.5 GB | `-c 8192 -np 2 --jinja -fa on` |
| **Embeddings** | nomic-embed-text-v1.5 | Q8_0 | ~0.3 GB | `--embedding --port 8081` |

### 6.3 llama-swap Configuration for All Tasks

```yaml
# llama-swap-config.yaml
models:
  classifier:
    cmd: >
      llama-server --port ${PORT}
      -m /models/qwen2.5-3b-instruct-q5_k_m.gguf
      -c 2048 -np 8 -ngl 99 -fa on
    ttl: 300
    aliases:
      - gpt-4o-mini-classify

  generator:
    cmd: >
      llama-server --port ${PORT}
      -m /models/qwen2.5-14b-instruct-q4_k_m.gguf
      -c 8192 -np 4 -ngl 99 -fa on -ctk q8_0 -ctv q8_0
    ttl: 600
    aliases:
      - gpt-4o-generate

  chatbot:
    cmd: >
      llama-server --port ${PORT} --jinja
      -m /models/qwen2.5-7b-instruct-q4_k_m.gguf
      -c 8192 -np 2 -ngl 99 -fa on
    ttl: 300
    aliases:
      - gpt-4o-chat

  agent:
    cmd: >
      llama-server --port ${PORT} --jinja
      -m /models/qwen2.5-7b-instruct-q5_k_m.gguf
      -c 8192 -np 2 -ngl 99 -fa on
    ttl: 300
    aliases:
      - gpt-4o-agent

groups:
  gpu-models:
    swap: true
    members:
      - classifier
      - generator
      - chatbot
      - agent
```

### 6.4 Docker Compose for Production

```yaml
# docker-compose.llm.yml
services:
  llama-swap:
    image: ghcr.io/mostlygeek/llama-swap:cuda
    container_name: llama-swap
    ports:
      - "9292:8080"
    volumes:
      - ./models:/models
      - ./llama-swap-config.yaml:/app/config.yaml
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  embeddings:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    container_name: embeddings
    ports:
      - "8081:8080"
    volumes:
      - ./models:/models
    environment:
      - LLAMA_ARG_MODEL=/models/nomic-embed-text-v1.5-q8_0.gguf
      - LLAMA_ARG_N_GPU_LAYERS=99
      - LLAMA_ARG_EMBEDDING=true
      - LLAMA_ARG_HOST=0.0.0.0
      - LLAMA_ARG_PORT=8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
```

```bash
# Start the stack
docker compose -f docker-compose.llm.yml up -d

# Test classification
curl http://localhost:9292/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "classifier", "messages": [{"role": "user", "content": "Classify: great product"}]}'

# Test embeddings
curl http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "model": "nomic-embed-text"}'
```

---

## Key Takeaways

1. **llama-server is production-ready** with full OpenAI API compatibility, continuous batching, embeddings, structured output, and tool calling support.

2. **Router mode is built-in** as of late 2025 -- you can dynamically load/unload multiple models without external tools. For more control, use llama-swap.

3. **For classification, prefer embeddings + classifier** over direct LLM prompting. Research shows ~49.5% higher accuracy with better calibration and lower latency.

4. **For 16GB VRAM**, target Q4_K_M quantization for 7-14B models. Use Q8_0 KV cache quantization to reclaim VRAM for longer contexts. Always keep model + KV cache within VRAM to avoid catastrophic slowdowns from CPU spillover.

5. **Docker is the right pattern**: Run llama-server (or llama-swap) as a separate container. Your application containers connect via the OpenAI SDK over the Docker network.

6. **The OpenAI Python SDK is the best client**: Just set `base_url="http://llama-server:8080/v1"` and your code works identically whether pointing at OpenAI's cloud or your local server.

7. **For agentic workflows**, use models with native tool-calling support (Qwen 2.5, Llama 3.x, Hermes) and start the server with `--jinja`. Avoid aggressive KV quantization for tool calling.

---

## Sources

- [llama.cpp server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp function-calling docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/function-calling.md)
- [llama.cpp Docker docs](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- [llama.cpp grammars README](https://github.com/ggml-org/llama.cpp/blob/master/grammars/README.md)
- [Model Management in llama.cpp (HuggingFace Blog)](https://huggingface.co/blog/ggml-org/model-management-in-llamacpp)
- [llama-swap GitHub](https://github.com/mostlygeek/llama-swap)
- [LiteLLM Proxy Docs](https://docs.litellm.ai/docs/simple_proxy)
- [vLLM vs llama.cpp (Red Hat)](https://developers.redhat.com/articles/2025/09/30/vllm-or-llamacpp-choosing-right-llm-inference-engine-your-use-case)
- [KV Cache Reuse Tutorial](https://github.com/ggml-org/llama.cpp/discussions/13606)
- [Continuous Batching Discussion](https://github.com/ggml-org/llama.cpp/discussions/10170)
- [Batching Explanation](https://github.com/ggml-org/llama.cpp/discussions/4130)
- [Embeddings vs Prompting for Classification (arXiv)](https://arxiv.org/html/2504.04277)
- [GGUF Quantization Guide](https://apatero.com/blog/gguf-quantized-models-complete-guide-2025)
- [Context vs VRAM Analysis](https://medium.com/@lyx_62906/context-kills-vram-how-to-run-llms-on-consumer-gpus-a785e8035632)
- [ServiceStack LLM Hosting Guide](https://docs.servicestack.net/ai-server/llama-server)
- [NVIDIA GPU Sharing](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/gpu-sharing.html)
- [Speculative Decoding Discussion](https://github.com/ggml-org/llama.cpp/discussions/10466)
- [RTX 5080 LLM Benchmarks](https://www.microcenter.com/site/mc-news/article/benchmarking-ai-on-nvidia-5080.aspx)
- [LLM Inference Engines Comparison](https://itecsonline.com/post/vllm-vs-ollama-vs-llama.cpp-vs-tgi-vs-tensort)
- [Grammar and Structured Output (DeepWiki)](https://deepwiki.com/ggml-org/llama.cpp/7.3-grammar-and-structured-output)
- [Constrained Decoding Guide](https://www.aidancooper.co.uk/constrained-decoding/)
- [NVIDIA LM Studio + llama.cpp on Blackwell](https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/)
