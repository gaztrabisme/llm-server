# MoE (Mixture of Experts) Model Offloading Strategies for Consumer GPUs

**Research compiled: February 2026**
**Target hardware: RTX 5080 (16GB VRAM, PCIe 5.0 x16) / 128GB RAM / 32 CPU cores**

---

## Table of Contents

1. [MoE Offloading in llama.cpp](#1-moe-offloading-in-llamacpp)
2. [PCIe Bandwidth Analysis](#2-pcie-bandwidth-analysis)
3. [Alternative Offloading Tools](#3-alternative-offloading-tools)
4. [Optimization Techniques](#4-optimization-techniques)
5. [Benchmarks and Real-World Results](#5-benchmarks-and-real-world-results)
6. [Practical Recommendations for RTX 5080 + 128GB RAM](#6-practical-recommendations-for-rtx-5080--128gb-ram)
7. [References](#7-references)

---

## 1. MoE Offloading in llama.cpp

### 1.1 How llama.cpp Handles MoE vs Dense Models

MoE models are fundamentally different from dense models for offloading purposes. A dense 70B model uses all 70B parameters every forward pass. An MoE model like DeepSeek-V3 (671B total / 37B active) or Qwen3-235B-A22B (235B total / 22B active) only activates a small subset of parameters per token. This makes MoE models **far more amenable to CPU offloading** than dense models because the performance penalty for offloading inactive experts is much lower.

llama.cpp treats MoE models with this key architectural insight:

- **Attention layers** and **shared experts** are used on every single token -- they are "always active" and should be prioritized for GPU placement.
- **Routed expert FFN tensors** are large, sparsely activated, and less impactful when offloaded to CPU.
- The model essentially has a "hot path" (attention + shared experts + router) and a "cold path" (routed expert FFN weights).

### 1.2 Expert-Level vs Layer-Level Granularity

llama.cpp supports **tensor-level granularity** for offloading, not just layer-level. This is the critical distinction that enables efficient MoE inference:

- **Layer-level offloading** (`-ngl`): The traditional approach. Setting `-ngl 20` puts 20 layers on GPU, the rest on CPU. For MoE models, this is suboptimal because each layer contains both the always-active attention tensors AND the massive expert FFN tensors.
- **Tensor-level offloading** (`-ot` / `--override-tensor`): Allows regex-based control over which tensors go to which device. This is the preferred approach for MoE models.

You **can** keep active components (attention, shared experts) on GPU while placing routed experts on CPU, **within the same layer**. This is not "expert-level" in the sense of choosing specific experts, but it is "tensor-type-level" -- you can offload all routed expert FFN weights to CPU while keeping attention weights on GPU across all layers.

### 1.3 Key Flags and Options

#### Core offloading flags:

```bash
# Layer-level GPU offload (traditional, suboptimal for MoE)
-ngl 999                    # Put all layers on GPU

# Tensor-level override (preferred for MoE)
-ot "exps=CPU"              # Offload all routed expert FFN tensors to CPU
# OR equivalently:
--n-cpu-moe N               # Offload N layers' MoE experts to CPU (counts from highest layer)

# Combined: all layers "on GPU" but expert FFNs on CPU
-ngl 999 -ot "exps=CPU"

# Granular regex control
-ot "blk\.([0-9]|[1-2][0-9]|30)\.=CUDA0,exps=CPU"
# Layers 0-30 on GPU, routed experts on CPU

# Multi-GPU distribution
-ot "blk\.([0-9])\.=CUDA0,blk\.(1[0-9])\.=CUDA1,exps=CPU"
```

#### Important `-ot` syntax notes:

- The pattern `exps` matches routed expert FFN tensors (`ffn_up_exps`, `ffn_gate_exps`, `ffn_down_exps`).
- It does **not** match shared experts (`ffn_xxx_shexp.weight`), which are always-active and should stay on GPU.
- Use comma-separated patterns in a single `-ot` flag rather than multiple `-ot` flags (better Docker compatibility).
- Regex patterns follow standard regex syntax.

#### Batch size and prompt processing flags:

```bash
-b 4096                     # Logical batch size (default 2048)
-ub 4096                    # Physical batch size (default 512)
-fa on                      # Flash attention (recommended)
--no-mmap                   # Disable memory mapping (often faster for offloading)
-t 16                       # CPU threads
```

#### Disaggregated prompt processing (op offload):

llama.cpp performs **disaggregated prompt processing**: when the prompt batch is large enough, it copies CPU-assigned weights to the GPU temporarily and processes the entire prompt on GPU. This is often faster than split CPU/GPU processing.

- Default trigger threshold: **32 tokens** (standard llama.cpp)
- ik_llama.cpp threshold: **32 * (total_experts / active_experts)** tokens
  - DeepSeek V3: 32 * (256/8) = 1024 tokens minimum
  - Qwen3-235B: 32 * (128/8) = 512 tokens minimum

The environment variable `GGML_OP_OFFLOAD_MIN_BATCH` can override this threshold.

#### Automated fitting (recent feature):

```bash
--fit                       # Enable auto-fitting (default: on in recent builds)
--fit-target 1024           # Target free VRAM per GPU in MiB
--fit-ctx <min_ctx>         # Minimum context size
```

The auto-fitter prioritizes dense tensors over sparse MoE tensors for GPU placement and can automatically determine optimal tensor placement. Setting any manual `-ngl`, `-ot`, or `--tensor-split` disables it.

### 1.4 ik_llama.cpp Fork Improvements

The [ikawrakow/ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) fork has several MoE-specific improvements:

- **Smarter MoE batch threshold**: MoE matrix multiplications only offloaded to GPU if batch size exceeds `32 * (total_experts / active_experts)`. This prevents wasteful GPU transfers for small batches.
- **`--merge-qkv`**: Merges Q/K/V attention tensors for better memory access.
- **`-gr` (graph reuse)**: Avoids rebuilding compute graphs.
- **`-mla 3`**: Multi-head latent attention for DeepSeek models.
- **Graph split mode** (`-sm graph`): Rudimentary tensor parallelism across multiple GPUs.
- **NUMA-aware execution**: Scripts for binding to CPU sockets and disabling NUMA balancing.
- Reported **~2x prefill speedup** over standard llama.cpp for MoE models in some configurations.

---

## 2. PCIe Bandwidth Analysis

### 2.1 Theoretical vs Practical Bandwidth

| PCIe Version | Theoretical (x16) | Practical (measured) |
|-------------|-------------------|---------------------|
| PCIe 3.0   | ~16 GB/s          | ~12-14 GB/s         |
| PCIe 4.0   | ~32 GB/s          | ~25-28 GB/s         |
| PCIe 5.0   | ~64 GB/s          | ~50-55 GB/s         |

The RTX 5080 uses PCIe 5.0 x16, giving approximately **50-55 GB/s practical bandwidth** between system RAM and GPU VRAM.

For comparison, the RTX 5080's internal GDDR7 memory bandwidth is **960 GB/s** -- roughly 17-19x faster than the PCIe link.

### 2.2 Expert Transfer Time Calculation

For a typical MoE expert at different quantization levels (assuming ~3B parameters per expert):

| Quantization | Bytes/Param | Expert Size | Transfer Time (PCIe 5.0) | Transfer Time (PCIe 4.0) |
|-------------|-------------|-------------|--------------------------|--------------------------|
| FP16        | 2           | ~6 GB       | ~120 ms                  | ~240 ms                  |
| Q8_0        | ~1          | ~3 GB       | ~60 ms                   | ~120 ms                  |
| Q4_K_M      | ~0.56       | ~1.7 GB     | ~34 ms                   | ~68 ms                   |
| IQ3_XXS     | ~0.39       | ~1.2 GB     | ~24 ms                   | ~48 ms                   |

### 2.3 Impact on Generation Latency

For interactive use, the target is typically **30-100 ms per token** (10-33 tokens/second).

- At Q8, transferring a single 3B-param expert takes ~60 ms over PCIe 5.0.
- Most MoE models activate **2-8 experts per token per layer**, and have **32-60+ layers**.
- If experts are NOT cached, the latency is catastrophic: transferring 2 experts x 32 layers = 64 expert loads x 60 ms = **3,840 ms per token** (0.26 tok/s).

This is why **expert caching is absolutely critical**. Without caching, naive expert offloading is 30-100x too slow for interactive use.

The saving grace: expert activation patterns show strong **temporal locality** (the same experts tend to be reused across consecutive tokens) and **frequency skew** (a small number of "hot" experts handle most tokens). With effective caching:

- Studies show **85-96% cache hit rates** are achievable with modest VRAM budgets.
- Only cache-miss experts need transfer, reducing effective transfer overhead by 5-15x.
- PCIe 5.0's 2x bandwidth improvement over PCIe 4.0 directly halves transfer latency for misses.

### 2.4 PCIe 5.0 Advantage

PCIe 5.0 provides a meaningful advantage over PCIe 4.0 for MoE offloading:

- Expert transfer time is halved (60 ms -> 30 ms for Q8 3B-param experts).
- The 50-55 GB/s practical bandwidth can be **fully saturated** with pinned memory and DMA transfers.
- For models with high expert reuse rates, PCIe 5.0 can bring cache-miss penalties below the computation time, effectively hiding transfer latency.

However, even PCIe 5.0 bandwidth is **orders of magnitude slower than on-chip GPU memory**. The strategy must be to minimize transfers, not to brute-force them.

---

## 3. Alternative Offloading Tools

### 3.1 ktransformers

[ktransformers](https://github.com/kvcache-ai/ktransformers) (by KVCache.ai / Tsinghua University) is the **most purpose-built tool for MoE CPU/GPU hybrid inference**.

**Key features:**
- Intel AMX and AVX-512/AVX2 optimized CPU kernels for INT4/INT8 quantized inference.
- NUMA-aware memory management.
- Asynchronous CPU-GPU task scheduling to minimize overhead.
- Published at SOSP 2025.

**Performance (DeepSeek-R1 671B, single 24GB GPU):**
- Prefill: up to **286 tokens/s** (v0.3, selective 6-expert mode)
- Decode: up to **14 tokens/s** on RTX 4090D 24GB + 382GB DRAM
- On a 3090Ti 24GB + 96GB DDR5: ~3.2 tok/s decode

**For a 16GB GPU like RTX 5080:**
- Decode speed would be lower than the 24GB results due to less VRAM for caching.
- Estimate: **2-5 tok/s** for DeepSeek-R1 671B with aggressive quantization, depending on RAM speed and CPU.
- ktransformers is reportedly **~2x faster than llama.cpp** for MoE models with CPU/GPU hybrid inference.

**Limitations:**
- Requires Intel CPUs with AMX for peak performance (Xeon Gold/Sapphire Rapids or newer).
- AMD CPU support exists via AVX2/AVX-512 but without AMX-level speedups.
- 16GB VRAM may limit the amount of attention/shared expert layers that fit on GPU.

**Supported models (as of Feb 2026):** DeepSeek-R1/V3, Qwen3MoE, Llama 4, Mixtral, Kimi K2.5, GLM-5, MiniMax-M2.5.

### 3.2 llama-swap

[llama-swap](https://github.com/mostlygeek/llama-swap) is a **model management proxy**, not an inference engine. It sits in front of llama.cpp (or vLLM, etc.) and provides:

- Automatic model swapping based on incoming requests.
- OpenAI/Anthropic API compatibility.
- Single YAML configuration for multiple model configs.
- LRU eviction for loaded models.
- Useful for switching between a smaller fast model and a larger MoE model on the same hardware.

It does not perform its own MoE offloading -- it delegates to the underlying inference engine.

### 3.3 vLLM

[vLLM](https://github.com/vllm-project/vllm) supports MoE models but its **expert-level CPU offloading is still under development** as of early 2026:

- General `--cpu-offload-gb` flag exists but has reported issues with MoE models (tensor device mismatch errors).
- Expert Parallelism (EP) is a P0 priority on the 2025 roadmap.
- **LvLLM** (NUMA extension of vLLM) added EP support in v1.7.0 (Feb 2026).
- vLLM excels at **high-throughput multi-user serving** but is less optimized for single-user MoE offloading on consumer hardware.
- Weight offloading v2 with async prefetching uses separate CUDA streams for overlapping weight onloading with kernel execution.

**Verdict:** Not recommended for single-user consumer MoE offloading currently. Better suited for datacenter deployments with full GPU memory.

### 3.4 ExLlamaV2

[ExLlamaV2](https://github.com/turboderp-org/exllamav2) is a high-performance inference library for consumer GPUs, but:

- Expert-level offloading (ktransformers-style CPU/GPU hybrid) has been **requested but not implemented** as of the available information.
- A feature request (Issue #706) discussed offloading a customizable number of experts to RAM for DeepSeek V3 685B.
- General CPU offloading (Issue #225) has also been discussed.
- ExLlamaV2 excels at **fully GPU-resident** inference with advanced quantization (EXL2 format) but lacks the MoE-specific offloading capabilities of ktransformers or llama.cpp's `-ot`.

**Verdict:** Not suitable for MoE offloading at this time.

### 3.5 PowerInfer

[PowerInfer](https://arxiv.org/abs/2312.12456) (SJTU) pioneered the concept of **neuron-level hot/cold partitioning** for LLM inference on consumer GPUs:

- Exploits power-law distribution in neuron activation: "hot" neurons stay on GPU, "cold" neurons computed on CPU.
- Achieved **13.20 tok/s average, 29.08 tok/s peak** on OPT-175B with a single RTX 4090.
- Up to **11.69x speedup** over llama.cpp.

**Limitations for MoE:**
- Originally designed for **dense models**, not MoE architectures.
- Fails to account for MoE-specific characteristics (routing, expert-level sparsity).
- Subsequent work (MoBiLE, MoE-Gen) addresses MoE-specific gaps.

### 3.6 HOBBIT

[HOBBIT](https://arxiv.org/abs/2411.01433) is a **mixed-precision expert offloading system** built on top of llama.cpp:

**Three-level optimization:**
1. **Token-level dynamic expert loading**: Uses gating output magnitude to score expert importance. Less critical cache-miss experts are loaded in low precision (INT4 instead of FP16, or INT2 instead of INT8). For Mixtral-8x7B: 67% high precision, 30% low precision, 3% skipped.
2. **Layer-level adaptive prefetching**: Achieves **96% prediction accuracy** for next-layer expert selection using a "stacking computer" that predicts multiple layers at once.
3. **Sequence-level multidimensional caching**: Combines LRU, LFU, LHU (Least Hot Used), and FLD (Frequency-Layered Decay) strategies.

**Benchmarks:**
| Platform | Model | Speedup vs llama.cpp | Speedup vs MoE-Infinity |
|----------|-------|---------------------|------------------------|
| RTX 4090 | Mixtral-8x7B | 3.21x | 2.30x |
| RTX 4090 | Phi-MoE | 3.29x | 3.92x |
| Jetson Orin | Mixtral-8x7B | 13.0x | 3.64x |
| Jetson Orin | Phi-MoE | 18.9x | 9.93x |

Accuracy impact: **< 1% degradation** on GSM8K and TruthfulQA.

Implementation: ~8,000 lines of C++/C on top of llama.cpp.

### 3.7 MoE-Gen

[MoE-Gen](https://arxiv.org/abs/2503.09716) targets **high-throughput batch MoE inference** on a single GPU:

**Key innovation: Module-based batching.** Instead of uniform batch sizes, different modules (attention vs. experts) get different batch sizes. Expert modules accumulate larger batches to amortize transfer costs.

**Performance on A5000 GPU:**
| Model | Baseline (tok/s) | MoE-Gen (tok/s) | Speedup |
|-------|------------------|-----------------|---------|
| Mixtral-8x7B | 33 | 469 | 14.2x |
| Mixtral-8x22B | 5 | 91 | 18.2x |
| DeepSeek-V2 | 1 | 31 | 31x |

These are **throughput numbers** (batch processing), not per-user latency. MoE-Gen is designed for dataset processing, not interactive chat.

### 3.8 MoE-Infinity

[MoE-Infinity](https://arxiv.org/abs/2401.14361) focuses on **single-user inference on personal machines**:

- Sparsity-aware expert cache that traces activation patterns.
- **3.1-16.7x per-token latency improvement** over vLLM, Ollama, DeepSpeed, and BrainStorm.
- Designed for batch-size-1 inference (typical for personal use).

---

## 4. Optimization Techniques

### 4.1 Expert Prediction and Prefetching

This is arguably the most impactful optimization for MoE offloading. Several approaches exist:

**Pre-attention expert prediction:**
- Lightweight routers trained to predict expert selection BEFORE the attention computation.
- Achieves **93-97% prediction accuracy** (93% on DeepSeek V2 Lite, 97% on Phi-mini-MoE).
- Enables prefetching experts for the current layer while attention is computing.

**Cross-layer prediction (HOBBIT):**
- Exploits high cosine similarity between consecutive layer inputs.
- **96% next-layer prediction accuracy**.
- A "stacking computer" predicts multiple layers simultaneously.

**Score-based prefetching (SpecMD):**
- Counterintuitively, **score-based prefetching outperforms fixed top-k** despite lower prediction accuracy.
- Adapts prefetch count per layer based on gating scores.
- Uses available bandwidth more effectively.

**Speculative prefetching (MoE-SpeQ):**
- Uses a small draft model to predict expert sequences for future tokens.
- Overlaps I/O with computation, hiding latency from critical path.
- **2.34x speedup** over state-of-the-art offloading for Phi-MoE.

### 4.2 Expert Caching Strategies

Traditional LRU caching is **suboptimal for MoE workloads** because MoE expert access follows deterministic sequential layer patterns, not temporal locality patterns.

**Least-Stale eviction (SpecMD, Feb 2026):**
- Reduces collision misses by up to **85x over LRU** at 1% cache capacity.
- Achieves **88-92% hit rates** with just 5% cache capacity.
- **10.7-34.7% TTFT reduction** on OLMoE with 0.6GB VRAM cache.

**Multidimensional caching (HOBBIT):**
- Combines LRU + LFU + LHU + FLD with weighted priority scores.
- **4.69-8.68% penalty reduction** over LRU alone.

**Frequency-based caching (LFU):**
- In empirical tests, LFU was **84.6% faster** than LRU baseline on A6000.
- Expert activation distributions are highly skewed (some experts activated 100x more than others).

**Practical implication for 16GB VRAM:** Even allocating 1-2 GB of VRAM as an expert cache (while keeping attention/shared experts in the remaining 14-15 GB) can achieve 85%+ cache hit rates, dramatically reducing PCIe transfer overhead.

### 4.3 KV Cache Management with Limited VRAM

With 16GB VRAM, KV cache competes with model weights and expert caches:

- **Full KV offloading to CPU** (MoE-Gen approach): Frees GPU VRAM for model weights and expert caches. CPU-based attention using AVX intrinsics handles KV operations.
- **Quantized KV cache**: Q8 or Q4 KV cache reduces VRAM usage by 2-4x.
- **Sliding window attention**: Models like Mixtral use sliding window, naturally limiting KV cache size.
- **Flash attention** (`-fa on`): Reduces peak VRAM during attention computation.

For the RTX 5080 with 16GB, a practical allocation might be:
- ~10-12 GB: Attention layers + shared experts + router weights
- ~2-3 GB: Expert VRAM cache (LRU/Least-Stale)
- ~1-2 GB: KV cache (quantized) + overhead

### 4.4 Quantization Strategies for MoE

MoE models present unique quantization challenges:

**Inter-expert imbalance:** Different experts see vastly different amounts of training data, leading to varying quantization sensitivity. Rarely-activated experts may have less robust weight distributions.

**Recommended strategy (mixed precision):**
- **Attention layers**: Higher precision (Q8 or even FP16) -- always active, critical for quality.
- **Shared experts**: Higher precision (Q8) -- always active.
- **Frequently-activated routed experts**: Medium precision (Q6_K or Q8).
- **Rarely-activated routed experts**: Lower precision (Q4_K_M or lower) -- less impact on quality.

**HOBBIT's approach:** Dynamically uses lower precision for cache-miss experts at inference time. High-precision and low-precision copies of experts are stored in CPU RAM. When a cache miss occurs and the expert has low importance (based on gating score), the smaller low-precision version is transferred, saving bandwidth with < 1% accuracy loss.

**llama.cpp `--override-tensor` for mixed quant:**
llama.cpp processes pre-quantized GGUF files, so mixed quantization is applied at the model preparation stage (using `llama-quantize` or during GGUF conversion). The `-ot` flag controls placement, not quantization level. However, Unsloth's "UD" (Ultra Dynamic) quants apply different quantization levels to different tensor types within a single GGUF file.

### 4.5 Pinned Memory and CUDA Async Transfers

**Pinned (page-locked) memory:**
- Prevents the OS from swapping expert weights to disk.
- Enables DMA transfers that bypass CPU, achieving full PCIe bandwidth.
- A single thread with pinned memory can saturate PCIe 4.0 (32 GB/s); multiple threads for PCIe 5.0.
- `--no-mmap` in llama.cpp helps ensure weights are in regular (potentially pinnable) memory.

**CUDA multi-stream scheduling:**
- Computation stream: runs current layer's inference.
- Communication stream: prefetches next layer's experts.
- CUDA events handle synchronization.
- This is the approach used by DuoServe-MoE and MoE-SpeQ.

**Implementation in practice:**
```
Layer N computation (GPU) ------>
                    Layer N+1 expert prefetch (PCIe) ------>
                                    Layer N+1 computation (GPU) ------>
```

If prefetch time < computation time, the transfer is fully hidden. With PCIe 5.0 and quantized experts, this is increasingly feasible.

---

## 5. Benchmarks and Real-World Results

### 5.1 Large MoE Models on Consumer Hardware

#### DeepSeek-R1/V3 671B

| Setup | Quantization | Prefill | Decode | Tool |
|-------|-------------|---------|--------|------|
| RTX 4090D 24GB + 382GB DRAM | FP8 hybrid | 286 tok/s | 14 tok/s | ktransformers v0.3 |
| 3090Ti 24GB + 96GB DDR5 | Q4_K_M | - | ~3.2 tok/s | ktransformers |
| M3 Ultra 512GB | Q4_K_M | ~0.9 tok/s (8K ctx) | ~6.2 tok/s | llama.cpp |
| CPU-only (high-mem) | IQ1_S 1.58bit | - | ~1-2 tok/s | llama.cpp |

**Note:** DeepSeek 671B at Q4_K_M requires ~405GB RAM. It does **not fit in 128GB**. For 128GB systems, you need IQ2 or lower quantization (~170-200GB), with significant quality loss. Or target smaller MoE models.

#### Qwen3-235B-A22B

| Setup | Quantization | Prefill | Decode | Tool |
|-------|-------------|---------|--------|------|
| 3090 24GB + 128GB DDR4-2933 | IQ3_K (ik_llama.cpp) | 24 tok/s | 7.4 tok/s | ik_llama.cpp |
| 3090 24GB + 128GB DDR4-2933 | Q3_XL (Unsloth) | 12 tok/s | 7.4 tok/s | llama.cpp |
| Ryzen 9950X + RTX 4000 SFF + RTX 2000 ADA + 192GB RAM | Q4_K_M | - | ~10 tok/s | llama.cpp |

**At Q4_K_M, Qwen3-235B-A22B requires ~135GB.** This is tight for 128GB RAM but feasible with swap or at IQ3/Q3 quantization (~90-100GB), leaving room for KV cache and OS overhead.

#### Mixtral-8x7B (45B total, 14B active)

| Setup | Configuration | Decode | Tool |
|-------|-------------|--------|------|
| RTX 4090 24GB | Expert offloading | Baseline: ~33 tok/s | llama.cpp |
| RTX 4090 24GB | HOBBIT | ~106 tok/s (3.21x) | HOBBIT |
| A5000 | MoE-Gen batch=256 | 469 tok/s (throughput) | MoE-Gen |
| 16GB GPU + RAM | Q4 + expert offload | ~7-15 tok/s | llama.cpp |

#### Mixtral-8x22B (141B total, 39B active)

| Setup | Configuration | Decode | Tool |
|-------|-------------|--------|------|
| A5000 + CPU RAM | MoE-Gen batch=256 | 91 tok/s (throughput) | MoE-Gen |
| 16GB GPU + 128GB RAM | Q4 expert offload | ~3-6 tok/s | llama.cpp (estimated) |

### 5.2 Projected Performance for RTX 5080 + 128GB RAM

Based on available benchmarks and scaling factors:

**Advantages of the RTX 5080 setup:**
- PCIe 5.0 provides ~2x bandwidth over PCIe 4.0 used in most benchmarks.
- 960 GB/s GDDR7 internal bandwidth (vs 1008 GB/s for 4090).
- Blackwell architecture improvements for inference.
- 32 CPU cores for CPU-side expert computation.

**Estimated performance (single user, interactive):**

| Model | Quant | Estimated Decode | Notes |
|-------|-------|-----------------|-------|
| Mixtral-8x7B | Q4_K_M | 15-25 tok/s | Fits mostly in VRAM |
| Mixtral-8x22B | Q4_K_M | 5-10 tok/s | Heavy CPU offload |
| Qwen3-235B-A22B | IQ3_XXS | 5-8 tok/s | ~90GB, fits in 128GB RAM |
| Qwen3-30B-A3B | Q8_0 | 20-40 tok/s | Small active params, fits in 16GB |
| DeepSeek-V3 671B | IQ2_XXS | 1-3 tok/s | ~170GB, borderline with 128GB |

These are rough estimates. Actual performance depends heavily on RAM speed, CPU architecture, and specific optimizations used.

---

## 6. Practical Recommendations for RTX 5080 + 128GB RAM

### 6.1 Best Tool Choices

1. **llama.cpp with `-ot`** -- Best general-purpose option. Mature, well-documented MoE offloading with tensor-level control. Start here.

2. **ik_llama.cpp** -- If you need better prefill performance, try the ikawrakow fork. Smarter MoE batch thresholds and additional optimizations.

3. **ktransformers** -- Best option if your CPU supports AVX-512 or AMX. Purpose-built for MoE hybrid inference. Notably faster decode than llama.cpp for large MoE models.

4. **HOBBIT** -- If it gets released/maintained, the mixed-precision approach could significantly speed up cache-miss scenarios.

### 6.2 Recommended Configuration Template

```bash
# For Qwen3-235B-A22B IQ3_XXS on RTX 5080 16GB + 128GB RAM
./llama-server \
  -m ./Qwen3-235B-A22B-IQ3_XXS.gguf \
  -ngl 999 \
  -ot "exps=CPU" \
  -c 8192 \
  -fa on \
  -b 4096 \
  -ub 4096 \
  -t 24 \
  --no-mmap

# For Mixtral-8x22B Q4_K_M
./llama-server \
  -m ./Mixtral-8x22B-Q4_K_M.gguf \
  -ngl 999 \
  -ot "exps=CPU" \
  -c 16384 \
  -fa on \
  -t 24 \
  --no-mmap
```

### 6.3 Model Selection Guide for 128GB RAM

| Model | Total Params | Active Params | Q4_K_M Size | IQ3 Size | Fits 128GB? | Recommended? |
|-------|-------------|---------------|-------------|----------|-------------|-------------|
| Qwen3-30B-A3B | 30B | 3B | ~17GB | ~12GB | Easily | Best "small MoE" |
| Mixtral-8x7B | 45B | 14B | ~26GB | ~18GB | Easily | Good baseline |
| Mixtral-8x22B | 141B | 39B | ~80GB | ~55GB | Yes | Good mid-range |
| Qwen3-235B-A22B | 235B | 22B | ~135GB | ~90GB | Tight at Q4, yes at IQ3 | Best large MoE for 128GB |
| DeepSeek-V3 671B | 671B | 37B | ~405GB | ~170GB | No at Q4, borderline at IQ2 | Too large |

### 6.4 RAM Speed Matters

For MoE offloading, CPU memory bandwidth is often the bottleneck during decode. With experts on CPU:

- DDR4-3200: ~50 GB/s (dual channel)
- DDR5-4800: ~77 GB/s (dual channel)
- DDR5-5600: ~90 GB/s (dual channel)
- DDR5-6000+: ~96+ GB/s (dual channel)

Higher RAM speeds directly translate to faster expert computation on CPU. If your 128GB is DDR5, you have a significant advantage over DDR4 systems.

---

## 7. References

### Guides and Documentation

- [Performant Local MoE CPU Inference with GPU Acceleration in llama.cpp](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide) -- Comprehensive guide covering `-ot` syntax, op offload, and optimization strategies.
- [How to Run Big MoE Models Like Qwen-3-235B-A22B in Llama.cpp](https://medium.com/@david.sanftenberg/gpu-poor-how-to-configure-offloading-for-the-qwen-3-235b-a22b-moe-model-using-llama-cpp-13dc15287bed) -- Practical walkthrough of override-tensor for MoE.
- [llama.cpp Discussion #18049: Automation for GPU Layers with MoE Optimizations](https://github.com/ggml-org/llama.cpp/discussions/18049) -- Auto-fitting feature details.
- [llama.cpp Discussion #13154: Documentation for -ot Flag](https://github.com/ggml-org/llama.cpp/discussions/13154) -- Syntax reference.

### Tools and Frameworks

- [ktransformers](https://github.com/kvcache-ai/ktransformers) -- CPU/GPU hybrid inference framework, SOSP 2025. [Paper (PDF)](https://madsys.cs.tsinghua.edu.cn/publication/ktransformers-unleashing-the-full-potential-of-cpu/gpu-hybrid-inference-for-moe-models/SOSP25-chen.pdf)
- [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) -- Fork with MoE-specific improvements. [PR #520: Better GPU Offload Strategy](https://github.com/ikawrakow/ik_llama.cpp/pull/520)
- [llama-swap](https://github.com/mostlygeek/llama-swap) -- Model management proxy for llama.cpp.
- [ExLlamaV2](https://github.com/turboderp-org/exllamav2) -- Fast inference library; MoE offloading requested but not yet implemented. [Issue #706](https://github.com/turboderp-org/exllamav2/issues/706)
- [MoE-Inf/awesome-moe-inference](https://github.com/MoE-Inf/awesome-moe-inference) -- Curated paper collection.

### Research Papers

- [HOBBIT: A Mixed Precision Expert Offloading System for Fast MoE Inference](https://arxiv.org/abs/2411.01433) -- Up to 9.93x speedup, mixed-precision loading, < 1% accuracy loss.
- [MoE-Gen: High-Throughput MoE Inference on a Single GPU](https://arxiv.org/abs/2503.09716) -- Module-based batching, up to 31x throughput improvement.
- [MoE-Infinity: Offloading-Efficient MoE Model Serving](https://arxiv.org/abs/2401.14361) -- Sparsity-aware caching for personal machines.
- [SpecMD: A Comprehensive Study on Speculative Expert Prefetching](https://arxiv.org/abs/2602.03921) -- Least-Stale eviction policy, 85x fewer collision misses than LRU.
- [MoE-SpeQ: Speculative Quantized Decoding](https://arxiv.org/abs/2511.14102) -- Draft-model-based expert prediction, 2.34x speedup.
- [In-depth Analysis on Caching and Pre-fetching in MoE Offloading](https://arxiv.org/abs/2511.05814) -- LFU 84.6% faster than LRU; speculative prefetch achieves 84.6% precision.
- [PowerInfer: Fast LLM Serving with Consumer-grade GPU](https://arxiv.org/abs/2312.12456) -- Neuron-level hot/cold partitioning, up to 11.69x speedup.
- [DuoServe-MoE: Dual-Phase Expert Prefetch and Cache Scheduling](https://arxiv.org/abs/2509.07379) -- Dual CUDA stream prefetching with pinned memory.
- [SpecMoEOff: Hiding Offloading Latency with Speculative Decoding](https://arxiv.org/abs/2508.21706) -- Combines speculative decoding with MoE offloading.
- [MoEQuant: Enhancing Quantization for MoE LLMs](https://arxiv.org/abs/2505.03804) -- Expert-balanced sampling for quantization.
- [MoBiLE: Efficient MoE Inference with Mixture of Big Little Experts](https://arxiv.org/abs/2510.12357) -- Addresses MoE-specific gaps in PowerInfer.
- [ExpertFlow: Adaptive Expert Scheduling and Memory Management](https://arxiv.org/abs/2510.26730) -- Adaptive prefetch horizons and cache-aware routing.

### Community Discussions

- [llama.cpp Issue #4667: MoE Offloading Real Layers](https://github.com/ggml-org/llama.cpp/issues/4667) -- Feature request for finer-grained MoE offloading.
- [llama.cpp PR #11397: Override Model Tensor Buffers](https://github.com/ggml-org/llama.cpp/pull/11397) -- The PR that introduced `-ot`.
- [llama.cpp Discussion #8721: Ideas from ktransformers Report](https://github.com/ggml-org/llama.cpp/discussions/8721) -- Cross-pollination between ktransformers and llama.cpp.
- [vLLM Forums: Expert Offloading](https://discuss.vllm.ai/t/expert-offloading/1880) -- Status of vLLM expert offloading.
- [Ollama Issue #11772: CPU Offload for MoE Weights](https://github.com/ollama/ollama/issues/11772) -- Community request for Ollama MoE support.
- [KTransformers DeepSeek R1/V3 Tutorial](https://kvcache-ai.github.io/ktransformers/en/DeepseekR1_V3_tutorial.html) -- Official setup guide.
