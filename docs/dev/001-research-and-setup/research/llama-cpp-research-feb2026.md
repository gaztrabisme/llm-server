# llama.cpp Research Report -- February 2026

**Latest version**: b8070 (released February 16, 2026)
**Repository**: https://github.com/ggml-org/llama.cpp
**Research date**: February 17, 2026

---

## 1. Blackwell / RTX 5080 Support

### Compute Capability and Architecture Codes

The RTX 5080 uses NVIDIA's **Blackwell architecture** with **compute capability 12.0**. However, there is a critical distinction between NVIDIA's compute capability numbering and the CMake architecture code:

- **Compute capability**: 12.0 (what `nvidia-smi` and `deviceQuery` report)
- **CUDA architecture code for CMake**: The correct value depends on your CUDA toolkit version:
  - **CUDA 12.8+**: Use `CMAKE_CUDA_ARCHITECTURES=120` (sm_120). CUDA 12.8 introduced sm_100, sm_101, and sm_120 for Blackwell GPUs.
  - **CUDA 13.x**: Full sm_120 support. llama.cpp b8070 ships Windows binaries for both CUDA 12 and CUDA 13.

There was historical confusion in early 2025 where some guides incorrectly suggested using `CMAKE_CUDA_ARCHITECTURES=90` for Blackwell -- this is **wrong** (sm_90 is Hopper/H100, not Blackwell consumer). The NVIDIA Software Migration Guide for Blackwell RTX GPUs explicitly recommends `CMAKE_CUDA_ARCHITECTURES=120` with CUDA 12.8+.

### Driver Requirements

- **Minimum driver**: R570 or higher (your driver 590.48.01 is compatible)
- **CUDA toolkit**: 12.8 minimum for native Blackwell support; CUDA 13.1 (which you have) provides full support

### Native Blackwell Kernel Support

llama.cpp has **working Blackwell support** but with some nuances:

1. **Basic CUDA inference**: Fully functional. Models run correctly on RTX 5080.
2. **CUDA Graphs**: Enabled by default for batch-size-1 inference. Up to 35% throughput improvement.
3. **Flash Attention**: Works on Blackwell. CUDA FA kernels boost throughput by up to 15%.
4. **MXFP4 (experimental)**: PR #17906 added experimental native MXFP4 support for Blackwell, leveraging the 5th-generation Tensor Cores for up to 25% faster prompt processing. This is for MXFP4-quantized models (like gpt-oss-120b).
5. **NVFP4 MoE kernels**: A feature request (Issue #18250) for SM120-native NVFP4 MoE kernels was **closed as not planned** on Feb 6, 2026. The MXFP4 path via PR #17906 is the intended approach.

### Known Issues

- **Issue #18398**: A compilation bug where Blackwell support broke after PR #17906 was reported. Building with `CMAKE_CUDA_ARCHITECTURES="120a-real"` or just `120` caused "Unsupported gpu architecture" errors in some configurations.
- **Issue #18090**: A regression in b7410+ caused `GGML_ASSERT(addr) failed` on RTX 5090 Blackwell that worked in b7376.
- **Workaround**: If you encounter build issues, try `CMAKE_CUDA_ARCHITECTURES=all-major` or build from the latest HEAD which should have fixes.

### Build Flags for RTX 5080

```bash
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON
cmake --build build --config Release
```

For broader compatibility (supporting older GPUs too):
```bash
-DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;120"
```

---

## 2. Latest llama.cpp Features

### 2.1 MoE (Mixture of Experts) Offloading

**Status**: Fully supported with fine-grained control.

MoE models (DeepSeek V3, Qwen3-235B, etc.) are particularly amenable to partial offloading because only a fraction of expert parameters are activated per token (e.g., DeepSeek V3 671B uses only 37B active parameters per forward pass).

**Key mechanisms**:

| Flag | Purpose |
|------|---------|
| `-ngl 999` | Assign all layers to GPU initially |
| `-ot "exps=CPU"` | Offload all routed experts to CPU |
| `--n-cpu-moe N` | Offload top N layers' MoE experts to CPU |
| `-ot "blk\.([0-9]\|[1-2][0-9]\|30)\.=CUDA0,exps=CPU"` | Assign specific layers to GPU, experts to CPU |
| `-b 4096 -ub 4096` | Recommended batch sizes for MoE models |
| `--no-mmap` | Recommended for offloading scenarios |
| `-fit off` | Disable auto-fit for manual tuning |

**Strategy**: The optimal approach is to assign all "always active" components (attention, dense FFN, shared experts) to GPU, and offload routed expert FFN to CPU. This works because routed experts are only partially activated each token.

**Multi-GPU example**:
```bash
-ot "blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU"
```

**Performance note**: GPU prompt processing via disaggregated offload is faster than CPU only above a threshold batch size (default: 32 tokens). Use `-b 4096 -ub 4096` for MoE models.

### 2.2 Tensor Splitting / Layer Offloading

**Override-tensor (`-ot`) system**: Uses regex selectors to assign individual tensors to specific devices. This is far more granular than simple layer counting.

```bash
# Offload only FFN expert tensors to CPU
-ot "ffn_(up|gate|down)_exps=CPU"

# Multiple expressions as comma-separated list
-ot "blk\.(0|1|2|3)\.=CUDA0,blk\.(4|5|6|7)\.=CUDA1,exps=CPU"
```

**Tensor split for multi-GPU**:
- `--tensor-split 0.5,0.5` -- split 50/50 across two GPUs
- `--tensor-split-cpu-last` -- control CPU offload ordering
- `--tensor-split-layer-sort` -- sort layers by size before splitting

### 2.3 CUDA Backend Optimizations

Recent optimizations from NVIDIA partnership (2025):

| Optimization | Speedup | Details |
|-------------|---------|---------|
| **CUDA Graphs** | Up to 35% | Groups GPU operations into single CPU call; enabled by default for batch-size-1 |
| **Flash Attention kernels** | Up to 15% | Improved attention processing; reduced memory bandwidth |
| **GPU token sampling** | Variable | Moves sampling to GPU, reducing CPU-GPU round trips |
| **Memory management** | Variable | New scheme allocates additional memory to GPU |
| **NVFP4/FP8 quantization** | Up to 35% combined | New low-precision formats for Blackwell/Ada GPUs |
| **MXFP4 on Blackwell** | Up to 25% prompt processing | Uses 5th-gen Tensor Cores natively |

### 2.4 Flash Attention Support

**Status**: Fully implemented across CUDA, Metal, Vulkan, and SYCL backends.

**Build flags**:
- `GGML_CUDA_FA=ON` -- Enable Flash Attention (enabled by default in recent builds)
- `GGML_CUDA_FA_ALL_QUANTS=ON` -- Enable FA for all KV cache quantization types (increases binary size but needed for Q4_0/Q8_0 KV cache)

**Supported head dimensions**: 64 and 128 (without tensor cores). With tensor cores, additional sizes may be supported.

**Runtime flag**: `-fa` or `--flash-attn` to enable at runtime.

**GQA (Grouped Query Attention)**: Supported. Recent improvements in late 2025 added better performance for GQA models where the ratio between Q head counts is not a power of 2.

**Late 2025 enhancements**: Non-padded attention masks, Volta tensor core compatibility, refactored kernel configuration for ROCm compatibility, out-of-bounds checks, and tile shape defect fixes.

### 2.5 KV Cache Quantization

**Status**: Fully supported with multiple quantization levels.

**Flags**:
```bash
--cache-type-k q8_0    # Quantize K cache (default: f16)
--cache-type-v q8_0    # Quantize V cache (default: f16)
```

**Supported types**: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1

**Impact**:

| Cache Type | VRAM Reduction | Perplexity Impact |
|-----------|---------------|-------------------|
| f16 (default) | Baseline | None |
| q8_0 | ~50% | Negligible (+0.002-0.05 ppl) |
| q4_0 | ~67% | Noticeable (+0.206-0.25 ppl) |

**Recommendation**: q8_0 is safe for nearly all use cases and should be the default choice for VRAM-constrained setups. q4_0 is usable for creative/casual tasks where slight quality loss is acceptable.

### 2.6 Speculative Decoding

**Status**: Multiple methods supported.

**Methods available**:

1. **Draft model** (`--spec-type draft`): Use a smaller model to generate candidates. A 0.5B draft model achieves up to 2.5x speedup at 10 draft tokens.
2. **n-gram cache** (`--spec-type ngram-cache`): Statistical prediction from token history.
3. **n-gram simple** (`--spec-type ngram-simple`): Pattern matching in token history, minimal overhead.
4. **n-gram map-k** (`--spec-type ngram-map-k`): Frequency-based drafting with configurable minimum hits.
5. **n-gram mod** (`--spec-type ngram-mod`): Lightweight (~16 MB), constant memory/compute, shared across server slots. Recommended for text iteration, reasoning models, and summarization.

**Key flags**:
```bash
--spec-type ngram-mod     # Speculative decoding type
--draft 16                # Number of tokens to draft (default: 16)
--draft-min 0             # Minimum draft tokens
--spec-ngram-size-n 12    # Lookup n-gram size
--spec-ngram-size-m 48    # Draft m-gram size
```

**Performance**: With ~60% acceptance rate, Llama 3.1 8B achieved 182 tok/s vs 90-100 tok/s without speculation. Coding tasks benefit most.

**Eagle-3**: Being integrated into llama.cpp, providing ~2-2.5x speedup over native autoregressive decoding.

### 2.7 Server Mode (llama-server) Features

**Status**: Production-ready with extensive feature set.

#### OpenAI-Compatible API Endpoints

| Endpoint | Description |
|----------|-------------|
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |
| `POST /v1/embeddings` | Embeddings |
| `GET /v1/models` | Model listing |

#### Anthropic API Compatibility (NEW -- merged Dec 22, 2025)

| Endpoint | Description |
|----------|-------------|
| `POST /v1/messages` | Anthropic Messages API (streaming, tool use) |
| `POST /v1/messages/count_tokens` | Token counting |

This is significant: **Claude Code and other Anthropic-compatible clients can now connect directly to llama-server.** Tool use with `tool_use` and `tool_result` content blocks is supported.

#### Additional Endpoints

- `GET /health` -- Health check
- `GET /metrics` -- Prometheus-compatible metrics
- `GET /slots` -- Request queue status
- `POST /completion` -- Native completion endpoint
- `POST /tokenize` / `POST /detokenize` -- Tokenization
- `POST /embedding` -- Embeddings
- `POST /reranking` -- Document relevance scoring
- `POST /infill` -- Code infilling
- `GET/POST /lora-adapters` -- LoRA adapter management

#### Concurrent Request Handling

- **Slot-based scheduling**: `-np N` / `--parallel N` sets the number of parallel slots
- **Continuous batching**: Enabled by default (`-cb`)
- **Batch size control**: `-b 2048` (logical), `-ub 512` (physical)
- Active slots are batched together in a single `llama_decode()` call

#### Advanced Server Features

- **Tool calling / Function calling**: Works with any model via chat template extraction
- **Structured output**: JSON schema constraints (`-j`), BNF grammar (`--grammar`)
- **Reasoning/Thinking**: `--reasoning-format deepseek` for reasoning models (DeepSeek R1, etc.)
- **Multimodal**: Vision model support via `--mmproj`
- **LoRA hot-loading**: `--lora` with runtime scaling
- **API authentication**: `--api-key`
- **HTTPS**: `--ssl-key-file`, `--ssl-cert-file`
- **Web UI**: Built-in (`--webui`, enabled by default)
- **Model routing**: `--models-dir` for multi-model serving
- **Prompt caching**: `--cache-prompt` for reusing cached prompts

---

## 3. Build Options for RTX 5080

### Recommended Build Configuration

```bash
# Clone latest
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with CUDA for RTX 5080 (Blackwell, sm_120)
cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j$(nproc)
```

### Key Build Flags

| Flag | Default | Purpose |
|------|---------|---------|
| `GGML_CUDA=ON` | OFF | Enable CUDA backend |
| `CMAKE_CUDA_ARCHITECTURES="120"` | auto | Target Blackwell (sm_120) |
| `GGML_CUDA_FA_ALL_QUANTS=ON` | OFF | Flash attention for all KV cache quant types |
| `GGML_CUDA_FORCE_MMQ=ON` | OFF | Use matrix-multiply quantized kernels (reduces VRAM, slower for large batches) |
| `GGML_CUDA_FORCE_CUBLAS=ON` | OFF | Force cuBLAS (higher memory, potentially faster) |
| `GGML_NATIVE=OFF` | ON | Disable native CPU optimizations for portability |

### CUDA Toolkit Compatibility

| CUDA Version | sm_120 Support | Notes |
|-------------|---------------|-------|
| < 12.8 | No | Cannot compile for Blackwell |
| 12.8 | Yes | First version with sm_120 support |
| 13.0 | Yes | Full support |
| 13.1 (yours) | Yes | Full support, latest |

### Multi-Architecture Build

To build a binary that works on multiple GPU generations:
```bash
-DCMAKE_CUDA_ARCHITECTURES="75-real;86-real;89-real;120"
```

This covers: Turing (RTX 20xx), Ampere (RTX 30xx), Ada (RTX 40xx), and Blackwell (RTX 50xx).

### If Using a Specific CUDA Installation

```bash
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.1/bin/nvcc
```

---

## 4. Performance Considerations

### RTX 5080 Benchmark Data

**Llama 7B Q4_0** (fully in VRAM):

| Metric | Without FA | With FA |
|--------|-----------|---------|
| Prompt processing (pp512) | ~7,400-9,600 tok/s | ~7,000-8,000 tok/s |
| Token generation (tg128) | ~183-185 tok/s | ~190-192 tok/s |

**General RTX 5080 performance**: ~132 tok/s token generation on typical models, competitive with RTX 4090 despite having less VRAM (16GB vs 24GB).

### Bottlenecks for Models Exceeding VRAM

#### 1. PCIe Bandwidth (Primary Bottleneck)

When model weights or KV cache spill to system RAM, PCIe becomes the critical bottleneck:

- **PCIe 4.0 x16**: ~32 GB/s theoretical (your setup)
- **PCIe 5.0 x16**: ~64 GB/s theoretical
- **GPU memory bandwidth (RTX 5080)**: ~960 GB/s (GDDR7)

The 30x difference between GPU memory bandwidth and PCIe bandwidth explains why partial offloading causes dramatic slowdowns.

**Practical impact**: A model dropping from full VRAM to partial offload can see 5-30x slowdown (e.g., 40 tok/s down to 8 tok/s with only 25/36 layers in VRAM on an 8B model).

**U-shaped performance curve**: At very low GPU utilization, performance can actually be worse than CPU-only (~2.6 tok/s) because PCIe transfer overhead dominates without enough GPU work to amortize it. Performance improves as more layers are offloaded to GPU, but only after crossing a threshold.

#### 2. System RAM Bandwidth

- DDR5-5600: ~90 GB/s (dual channel)
- DDR4-3200: ~50 GB/s (dual channel)
- This limits CPU-side MoE expert computation speed

#### 3. Context Length vs VRAM

KV cache grows linearly with context length and can quickly consume available VRAM:

| Context | KV Cache (f16, 8B model) | KV Cache (q8_0) | KV Cache (q4_0) |
|---------|------------------------|-----------------|-----------------|
| 4K | ~0.5 GB | ~0.25 GB | ~0.17 GB |
| 32K | ~4 GB | ~2 GB | ~1.3 GB |
| 128K | ~16 GB | ~8 GB | ~5.3 GB |

With 16GB VRAM on the RTX 5080, long contexts will force model layer offloading. Use KV cache quantization (q8_0) to reclaim VRAM.

#### 4. PCIe Gen 4 vs Gen 5 Impact

The practical difference for llama.cpp inference is surprisingly small for models that fit in VRAM (~6% for prompt processing). It matters much more for partial offloading scenarios where large amounts of weight data must shuttle between CPU and GPU each token. With 300GB of model weights on CPU over PCIe 4.0, expect ~10 seconds per batch just for data transfer.

### Optimization Strategies for 16GB VRAM

1. **Use KV cache quantization**: `--cache-type-k q8_0 --cache-type-v q8_0` (halves KV cache VRAM)
2. **Use aggressive model quantization**: Q4_K_M or Q3_K_M for larger models
3. **For MoE models**: Offload experts to CPU with `-ot "exps=CPU"`, keep attention on GPU
4. **Flash Attention**: Always enable (`-fa`) -- reduces memory requirements
5. **Speculative decoding**: Use ngram-mod for free speedup on repetitive tasks
6. **MXFP4**: For Blackwell-optimized models, leverage native Tensor Core support

### What Fits in 16GB VRAM?

Approximate model size limits (weights only, before KV cache):

| Model Size | Quantization | Approx Size | Fits in 16GB? |
|-----------|-------------|-------------|---------------|
| 7-8B | Q4_K_M | ~4.5 GB | Yes, with room for large context |
| 7-8B | Q8_0 | ~8 GB | Yes, moderate context |
| 13B | Q4_K_M | ~7.5 GB | Yes, limited context |
| 14B | Q4_K_M | ~8 GB | Yes, limited context |
| 32B | Q4_K_M | ~18 GB | No, needs partial offload |
| 70B | Q4_K_M | ~40 GB | No, heavy offload needed |
| 70B | Q3_K_M | ~30 GB | No, heavy offload needed |

---

## 5. Docker Support

### Official Docker Images

llama.cpp provides official Docker images via GitHub Container Registry:

| Image | Contents |
|-------|----------|
| `ghcr.io/ggml-org/llama.cpp:full-cuda` | llama-cli, llama-completion, conversion tools |
| `ghcr.io/ggml-org/llama.cpp:light-cuda` | llama-cli, llama-completion only |
| `ghcr.io/ggml-org/llama.cpp:server-cuda` | llama-server only |

Build-tagged versions are also available (e.g., `ghcr.io/ggml-org/llama.cpp:server-cuda-b8070`).

**Platform**: linux/amd64 only. No ARM64+CUDA images.

### Building Locally

```bash
# Server with CUDA
docker build -t local/llama.cpp:server-cuda \
  --target server \
  -f .devops/cuda.Dockerfile .

# Full suite with CUDA
docker build -t local/llama.cpp:full-cuda \
  --target full \
  -f .devops/cuda.Dockerfile .
```

**Build arguments**:
- `CUDA_VERSION`: Default 12.4.0 (you may want to override for 13.1 / Blackwell)
- `CUDA_DOCKER_ARCH`: Defaults to cmake auto-detection

### Running with GPU

**Prerequisites**: NVIDIA Container Toolkit (`nvidia-container-toolkit`) must be installed (you have v1.18.2).

```bash
# Run server with GPU
docker run --gpus all \
  -v /path/to/models:/models \
  -p 8080:8080 \
  ghcr.io/ggml-org/llama.cpp:server-cuda \
  -m /models/model.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -ngl 999 \
  -fa \
  --cache-type-k q8_0 \
  --cache-type-v q8_0

# Run CLI inference
docker run --gpus all \
  -v /path/to/models:/models \
  ghcr.io/ggml-org/llama.cpp:full-cuda \
  --run -m /models/model.gguf \
  -p "Hello, world" \
  -n 512 \
  -ngl 999
```

### Important Caveats

1. **Pre-built images use CUDA 12.4 by default** -- for RTX 5080 (Blackwell/sm_120), you likely need to **build locally** with CUDA 12.8+ or 13.x to get native Blackwell support rather than relying on PTX JIT compilation.
2. **GPU-enabled images are not CI-tested** beyond building successfully.
3. **Custom CUDA versions require local builds** -- the pre-built images do not offer CUDA 13.x variants as of Feb 2026.

### Recommended Docker Approach for RTX 5080

Build a custom image with CUDA 13.1:

```bash
# Clone llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp

# Build with your CUDA version
docker build -t llama-cpp:server-cuda13 \
  --target server \
  --build-arg CUDA_VERSION=13.1.0 \
  --build-arg CUDA_DOCKER_ARCH=120 \
  -f .devops/cuda.Dockerfile .
```

(Note: Verify that the CUDA 13.1 base image exists on Docker Hub as `nvidia/cuda:13.1.0-devel-ubuntu24.04`. If not, you may need to adjust the Dockerfile base image or use CUDA 12.8.)

---

## 6. Notable Recent Developments (Late 2025 - Early 2026)

| Date | Feature |
|------|---------|
| Dec 2025 | Anthropic Messages API (`/v1/messages`) merged (PR #17570) |
| Dec 2025 | Full acceleration on Android/ChromeOS via new GUI binding |
| Dec 2025 | MXFP4 experimental Blackwell support (PR #17906) |
| Late 2025 | Profile-Guided Speculative Decoding implementation |
| Late 2025 | CUDA FlashAttention generalized: non-padded masks, Volta support |
| Jan 2026 | Delta-net graph deduplication for Qwen family (b7885) |
| Feb 2026 | b8070 release with CUDA 12/13 binaries |

### ik_llama.cpp Fork

Worth noting: the [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork offers additional MoE-specific optimizations including:
- `--merge-qkv` (merged Q/K/V attention tensors)
- `-gr` (graph reuse)
- `-smgs` (split mode graph scheduling)
- `-mla 3` (multi-head latent attention for DeepSeek models)
- `-sm graph` (graph split mode for multi-GPU)

These may provide better MoE performance in some configurations but are not in mainline llama.cpp.

---

## 7. Uncertainties and Caveats

Items flagged as uncertain or potentially outdated:

1. **CUDA 13.1 Docker base image availability**: It is unclear whether `nvidia/cuda:13.1.0-devel-ubuntu24.04` exists on Docker Hub. CUDA 13.x is very new. You may need to use CUDA 12.8 base images instead.

2. **Pre-built GHCR image CUDA version**: The pre-built `ghcr.io/ggml-org/llama.cpp:server-cuda` images default to CUDA 12.4. Whether they include PTX for sm_120 (allowing JIT compilation on Blackwell) or require native sm_120 compilation is not definitively documented.

3. **MXFP4 stability**: PR #17906 is described as "experimental" and Issue #18398 reported compilation breakage after it was merged. Current stability on b8070 is not confirmed.

4. **Benchmark numbers**: The RTX 5080 benchmark numbers cited are from various sources with different llama.cpp versions, driver versions, and configurations. Your actual performance will vary.

5. **Anthropic API compatibility depth**: While `/v1/messages` is merged, the maintainers note "no strong claims of compatibility with the Anthropic API spec." Complex tool-calling chains may have edge cases.

6. **CMAKE_CUDA_ARCHITECTURES value**: There has been community confusion about whether to use `90`, `120`, or `all-major`. The definitive answer per NVIDIA's migration guide is `120` with CUDA 12.8+. Using `all-major` is the safest option if you encounter issues.

---

## Sources

- [NVIDIA Blog: LM Studio Accelerates LLM Performance with RTX GPUs and CUDA 12.8](https://blogs.nvidia.com/blog/rtx-ai-garage-lmstudio-llamacpp-blackwell/)
- [NVIDIA Technical Blog: Optimizing llama.cpp AI Inference with CUDA Graphs](https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/)
- [NVIDIA Technical Blog: Open Source AI Tool Upgrades Speed Up LLM and Diffusion Models on RTX PCs](https://developer.nvidia.com/blog/open-source-ai-tool-upgrades-speed-up-llm-and-diffusion-models-on-nvidia-rtx-pcs)
- [NVIDIA Developer Forums: Software Migration Guide for Blackwell RTX GPUs](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [NVIDIA Developer Forums: CUDA Toolkit 12.8 -- What GPU is sm_120?](https://forums.developer.nvidia.com/t/cuda-toolkit-12-8-what-gpu-is-sm-120/322128)
- [NVIDIA Blackwell Compatibility Guide](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [llama.cpp GitHub Releases](https://github.com/ggml-org/llama.cpp/releases)
- [llama.cpp Build Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp Docker Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md)
- [llama.cpp Server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
- [llama.cpp Speculative Decoding Documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [llama.cpp Issue #18250: SM120 NVFP4 MoE Kernels](https://github.com/ggml-org/llama.cpp/issues/18250)
- [llama.cpp Issue #18398: Blackwell Support Broken After PR #17906](https://github.com/ggml-org/llama.cpp/issues/18398)
- [llama.cpp Issue #18090: Regression on RTX 5090 Blackwell](https://github.com/ggml-org/llama.cpp/issues/18090)
- [llama.cpp PR #17570: Anthropic Messages API Support](https://github.com/ggml-org/llama.cpp/pull/17570)
- [Hugging Face Blog: Anthropic Messages API in llama.cpp](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp)
- [Hugging Face Blog: Performant MoE CPU Inference with GPU Acceleration](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [llama.cpp Discussion #15013: Performance on NVIDIA CUDA](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [DeepWiki: Flash Attention and Optimizations](https://deepwiki.com/ggml-org/llama.cpp/7.4-flash-attention-and-optimizations)
- [DeepWiki: HTTP Server](https://deepwiki.com/ggml-org/llama.cpp/5.2-http-server)
- [Kombitz: Understanding NVIDIA's sm_80 to sm_120](https://www.kombitz.com/2025/11/23/understanding-nvidias-sm_80-to-sm_120-in-plain-english/)
- [Guide: Compiling on RTX 5090 + VS 2026 + CUDA 13.1](https://github.com/abetlen/llama-cpp-python/discussions/2115)
- [llama-cpp-python Issue #2028: Building for RTX 50 Blackwell GPU](https://github.com/abetlen/llama-cpp-python/issues/2028)
- [Weekly GitHub Report for llama.cpp: Jan 25 - Feb 1, 2026](https://buttondown.com/weekly-project-news/archive/weekly-github-report-for-llamacpp-january-25-2026-2813/)
- [Phoronix: llama.cpp AI Performance with the GeForce RTX 5090](https://www.phoronix.com/review/nvidia-rtx5090-llama-cpp)
- [Best Local LLMs for Every NVIDIA RTX 50 Series GPU](https://apxml.com/posts/best-local-llms-for-every-nvidia-rtx-50-series-gpu)
