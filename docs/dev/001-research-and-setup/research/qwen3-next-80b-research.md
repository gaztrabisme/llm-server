# Qwen3-Next-80B-A3B: Comprehensive Research Report

**Date**: February 17, 2026
**Status**: The model **exists and is publicly available**. Released by Alibaba's Qwen team in September 2025.

---

## 1. Architecture Details

### Confirmed: This is a Mixture of Experts (MoE) Model

Qwen3-Next-80B-A3B is a **hybrid MoE** model with several architectural innovations beyond standard MoE designs.

### Core Specifications (from `config.json`)

| Parameter | Value |
|---|---|
| **Architecture** | `Qwen3NextForCausalLM` |
| **Total Parameters** | ~80B |
| **Active Parameters** | ~3.9B per token (3.75% activation) |
| **Non-Embedding Parameters** | ~79B |
| **Hidden Size** | 2048 |
| **Number of Layers** | 48 |
| **Vocabulary Size** | 151,936 |
| **Max Context Length** | 262,144 tokens (native), extensible to ~1M with YaRN |
| **Precision** | bfloat16 |

### MoE Configuration

| Parameter | Value |
|---|---|
| **Total Routed Experts** | 512 |
| **Shared Experts** | 1 |
| **Active Experts per Token** | 10 (routed) + 1 (shared) = 11 |
| **Expert Intermediate Size (`moe_intermediate_size`)** | 512 |
| **Shared Expert Intermediate Size** | 512 |
| **Decoder Sparse Step** | 1 (every layer is MoE) |

### Individual Expert Size Calculation

Each expert is an FFN with a SwiGLU-style architecture (gate + up + down projections):
- **Gate projection**: `hidden_size x moe_intermediate_size` = 2048 x 512 = 1,048,576 params
- **Up projection**: 2048 x 512 = 1,048,576 params
- **Down projection**: 512 x 2048 = 1,048,576 params
- **Per-expert total**: ~3.15M parameters (~3M params)
- **All 512 routed experts**: 512 x 3.15M = ~1.61B params per layer
- **Across 48 layers**: 48 x 1.61B = ~77.4B params in routed experts alone

This confirms that the **vast majority** (~97%) of the 80B parameters are in the routed expert FFNs, with only ~2.6B in attention layers, embeddings, shared experts, and other components.

**Per-expert weight size**:
- BF16: ~6.3 MB per expert (3.15M params x 2 bytes)
- Q8_0: ~3.15 MB per expert
- **Per-layer expert block (all 512)**: ~3.2 GB (BF16), ~1.6 GB (Q8_0)
- **Active experts per token per layer (10+1 shared)**: ~69 MB (BF16), ~35 MB (Q8_0)

### Hybrid Attention Architecture

This is **not a standard transformer**. The 48 layers follow a repeating pattern:

**Layout**: `12 x (3 x (Gated DeltaNet -> MoE) + 1 x (Gated Attention -> MoE))`

- **36 layers** use **Gated DeltaNet** (linear attention) -- constant memory per token, no growing KV cache
- **12 layers** use **Gated Attention** (standard grouped-query attention) -- standard KV cache

**Gated DeltaNet (linear attention) heads**:
- Linear attention V heads: 32
- Linear attention QK heads: 16
- Head dimension: 128

**Gated Attention (full attention) heads**:
- Query heads: 16
- KV heads: 2 (GQA ratio 8:1)
- Head dimension: 256
- Rotary embedding dimension: 64 (partial_rotary_factor = 0.25)

This hybrid design is why the KV cache grows very slowly with context length -- only 12 of 48 layers maintain a traditional KV cache.

---

## 2. GGUF Availability

### Confirmed: GGUF versions are widely available

**Official Qwen repositories**:
- [Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct-GGUF)
- [Qwen/Qwen3-Next-80B-A3B-Thinking-GGUF](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Thinking-GGUF)

**Community quantizations**:
- [unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF)
- [bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-Next-80B-A3B-Instruct-GGUF)

### File Sizes by Quantization

| Quantization | File Size |
|---|---|
| BF16 | 159 GB |
| **Q8_0** | **84.8 GB** |
| Q8_K_XL (UD) | 93.1 GB |
| Q6_K | 65.5 GB |
| Q6_K_XL (UD) | 68.5 GB |
| Q5_K_M | 56.8 GB |
| Q4_K_M | 48.5 GB |
| Q4_K_XL (UD) | 46.1 GB |
| Q3_K_M | 38.3 GB |
| Q2_K | 29.2 GB |
| IQ1_S (UD) | 22.9 GB |

**Q8_0 at 84.8 GB fits in 128GB RAM with room to spare.**

---

## 3. Memory Requirements

### Model Weight Size at Q8_0

- **Total weights**: 84.8 GB
- **Remaining RAM after loading**: 128 - 84.8 = ~43 GB (for OS, KV cache, etc.)

### KV Cache Size Estimates

The hybrid attention architecture makes KV cache much smaller than a comparable dense model. Only 12 of 48 layers maintain a standard KV cache (the Gated Attention layers). The other 36 layers use Gated DeltaNet, which maintains a fixed-size recurrent state regardless of sequence length.

**Gated Attention KV cache per token (12 layers, GQA with 2 KV heads, head_dim 256)**:
- Per-layer KV: 2 (K+V) x 2 (KV heads) x 256 (head_dim) = 1,024 elements
- 12 layers: 12 x 1,024 = 12,288 elements per token
- In FP16: 12,288 x 2 bytes = ~24 KB per token

| Context Length | KV Cache (FP16) | KV Cache (Q8_0) |
|---|---|---|
| 4K tokens | ~96 MB | ~48 MB |
| 8K tokens | ~192 MB | ~96 MB |
| 32K tokens | ~768 MB | ~384 MB |
| 128K tokens | ~3 GB | ~1.5 GB |
| 256K tokens | ~6 GB | ~3 GB |

### RAM Breakdown for Q8_0 with CPU Offloading

| Component | Size |
|---|---|
| Model weights (Q8_0) | 84.8 GB |
| KV cache (32K context, FP16) | ~0.8 GB |
| KV cache (128K context, FP16) | ~3 GB |
| OS + llama.cpp overhead | ~3-5 GB |
| **Total at 32K context** | **~89 GB** |
| **Total at 128K context** | **~93 GB** |

**128GB system has sufficient headroom for Q8_0 at up to 128K context.**

### VRAM Requirements for Active Experts Only

Active computation per token involves:
- 10 routed experts + 1 shared expert per layer x 48 layers
- Attention layers (both DeltaNet and GQA)
- Embeddings, norms, router weights

At Q8_0:
- Active expert weights per layer: 11 experts x ~3.15 MB = ~34.6 MB
- Across 48 layers: ~1.66 GB
- Non-expert components (attention, embeddings, norms, routers): ~2-3 GB
- **Total active parameter footprint**: ~3.9B params x 1 byte (Q8) = ~3.9 GB
- **Plus KV cache and compute buffers**: ~1-2 GB

**Estimated VRAM for active-only inference: ~5-6 GB -- well within 16 GB.**

---

## 4. MoE Offloading Strategy Analysis

### Theory: Load All Weights in RAM, Keep Active Experts in VRAM

This is **exactly the strategy that llama.cpp was designed to support** for MoE models, and it is **feasible**.

### llama.cpp MoE Offloading Mechanisms

1. **`-ot "exps=CPU"`**: Keeps ALL routed expert FFN weights in system RAM while attention layers, shared experts, embeddings, and normalization layers stay on GPU.

2. **`--n-cpu-moe N`**: Keeps the expert weights of the first N layers in CPU (finer control).

3. **Tensor override (`-ot`)**: Regex-based assignment of specific tensor groups to CPU or GPU. Example:
   ```
   -ngl 999 -ot "exps=CPU" -b 4096 -ub 4096
   ```

### How MoE Expert Offloading Works

**Regular (dense) model offloading**: Entire layers are placed on GPU or CPU. Bandwidth-bottlenecked because every token must pass through every layer.

**MoE expert offloading**: Fundamentally more efficient:
- The **router** (small, on GPU) determines which experts to activate
- Only the **selected experts' weights** are transferred from CPU to GPU
- Attention layers and other "always-on" components stay permanently on GPU
- Only 10/512 experts (~2%) needed per token -- transferring a tiny fraction of total weights

### PCIe 5.0 x16 Bandwidth Analysis

RTX 5080 has PCIe 5.0 x16: **~64 GB/s** bidirectional.

**Per-token expert transfer at Q8_0**:
- 10 activated experts per layer x 48 layers = 480 expert activations per token
- Each expert: ~3.15 MB at Q8_0
- Total transfer per token: 480 x 3.15 MB = **~1.51 GB per token**

**Transfer time**:
- At 64 GB/s: 1.51 GB / 64 = **~23.6 ms per token**
- Theoretical maximum throughput: **~42 tokens/sec** (transfer-limited)

Caveats:
1. **Expert caching**: llama.cpp does not (as of current builds) dynamically cache individual experts in VRAM between tokens
2. **Batched transfer**: With batch processing, multiple tokens share expert activations
3. **Prefetch and overlap**: Some implementations overlap compute and transfer

### Real-World Performance Benchmarks

| Hardware | Quant | Config | Token Gen (t/s) | Prompt (t/s) |
|---|---|---|---|---|
| RTX PRO 6000 (96GB) single GPU | Q8_0 | Full VRAM | 132.09 | 3,563 |
| RTX PRO 6000 + RTX 6000 Ada dual | Q8_0 | Dual GPU | 118.63 | 2,770 |
| DGX Spark (GB10) | MXFP4 | Partial offload | 45.93 | 1,242 |
| M2 Ultra (unified memory) | Q8_0 | Full offload | 43.78 | 1,338 |
| Strix Halo (96GB DDR5, iGPU) | Q4_K_M | 49 GPU layers | 15.3 | 57 |
| Strix Halo (96GB DDR5, iGPU) | Q8_0 | Full offload | 10.9 | N/A |
| Ryzen AI 9 HX PRO 370 (CPU-only) | Q4_K_M | CPU only | 7.74 | N/A |

**Estimated for RTX 5080 16GB + 128GB DDR5, Q8_0, `--cpu-moe`**:
- **Token generation: 15-30 t/s** (speculative, no exact benchmark exists)
- **Prompt processing: 200-500 t/s** (batch amortization helps significantly)

### Recommended llama.cpp Launch Command

```bash
./llama-server \
  -m ./Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf \
  -c 32768 \
  -ngl 999 \
  -ot "exps=CPU" \
  -fa on \
  -t 16 \
  -b 4096 \
  -ub 4096 \
  --no-mmap \
  --jinja
```

Key flags:
- `-ngl 999`: Put all layers on GPU first
- `-ot "exps=CPU"`: Override expert tensors to CPU (attention stays on GPU)
- `-b 4096 -ub 4096`: Large batch size for efficient prompt processing
- `--no-mmap`: Load entire model into RAM (important for MoE offload performance)
- `-fa on`: Flash attention
- `-t 16`: Use 16 CPU threads (half of 32 cores; tune this)

---

## 5. Model Variants

### Instruct vs Thinking

- **Qwen3-Next-80B-A3B-Instruct**: Standard instruction-following model. Best for API serving, classification, synthetic data generation.
- **Qwen3-Next-80B-A3B-Thinking**: Chain-of-thought reasoning model. Better for complex reasoning tasks, agentic workflows.

### Qwen3-Coder-Next

Same 80B architecture, code-optimized. Same memory profile. Choose if workload is code-focused.

---

## 6. Alternative Models for 128GB RAM + 16GB VRAM

| Model | Total | Active | Q8_0 Size | Fit in 128GB? |
|---|---|---|---|---|
| **Qwen3-Next-80B-A3B** | 80B | 3.9B | 84.8 GB | Yes (ideal) |
| Qwen3-Coder-Next | 80B | 3B | 84.8 GB | Yes |
| Qwen3-30B-A3B | 30.5B | 3.3B | ~30 GB | Yes (easy) |
| Mixtral 8x7B | 46.7B | 12.9B | ~47 GB | Yes |
| Mixtral 8x22B | 141B | 39B | ~141 GB | No at Q8 |
| DeepSeek-V3/R1 | 671B | 37B | ~670 GB | No |

**Qwen3-Next-80B-A3B is the ideal model for this hardware configuration.**

---

## Sources

- [Qwen/Qwen3-Next-80B-A3B-Instruct (HuggingFace)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [Qwen/Qwen3-Next-80B-A3B-Instruct config.json](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/blob/main/config.json)
- [unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF](https://huggingface.co/unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF)
- [NVIDIA Blog: Qwen3-Next Hybrid MoE Architecture](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
- [Hardware Corner: llama.cpp Qwen3 Speed Boost](https://www.hardware-corner.net/llama-cpp-update-qwen3-speed-boost/)
- [llama.cpp Issue #19396: Qwen3-Next-80B 8-bit on 96GB System](https://github.com/ggml-org/llama.cpp/issues/19396)
- [Performant MoE CPU Inference Guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [vLLM Blog: Qwen3-Next Support](https://blog.vllm.ai/2025/09/11/qwen3-next.html)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/abs/2505.09388)
