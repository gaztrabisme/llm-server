# Expert Caching Research for MoE Offloading

**Date**: 2026-02-25
**Model**: Qwen3-Next-80B-A3B-Instruct (Q8_0 GGUF)
**Engine**: llama.cpp
**Hardware**: RTX 5080 16GB, 128GB DDR5, PCIe 5.0 x16

---

## Executive Summary

llama.cpp does **not** currently implement GPU-side expert caching or LRU/LFU expert eviction. All routed experts assigned to CPU stay on CPU — there is no dynamic caching of hot experts into VRAM. However, llama.cpp's `-ot` flag allows **static partial offloading**: keeping some MoE layers fully on GPU (experts included) while offloading others to CPU. This is the most practical speedup path today.

Academic research (HOBBIT, MoE-Infinity, ProMoE) demonstrates 2-10x speedups from expert caching with LRU/LFU policies and prefetching, but none are integrated into mainline llama.cpp. An open PR (#17044) proposes expert caching but has no implementation yet.

**Recommended approach**: Use `-ot` to keep 6-12 MoE layers fully on GPU within our ~10GB VRAM budget, reducing PCIe transfers by 12-25%. This requires no code changes and can be tested immediately.

---

## 1. Current llama.cpp MoE Offloading Architecture

### How It Works Today

When using `-ot "exps=CPU"` with `-ngl 999`:
1. All layer tensors (attention, norms, shared experts, embeddings) load to GPU VRAM
2. Routed expert FFN weights (gate, up, down projections) stay in system RAM
3. Per token, the router selects 10 of 512 experts per layer
4. The token's **activation vector** (2048 floats = 8KB) is sent CPU-side over PCIe
5. CPU performs the expert FFN computation using weights already in RAM
6. Result is sent back to GPU

**Critical insight**: llama.cpp does NOT transfer expert weights to GPU. It transfers the small activation vector to CPU and computes there. This means "expert caching on GPU" would require a fundamentally different execution model — the expert computation would need to happen on GPU using cached weights.

### Current `-ot` Granularity

The `-ot` (override tensor) flag uses **regex matching on tensor names**:

```
blk.{N}.ffn_gate_exps.weight   — routed expert gate projections for layer N
blk.{N}.ffn_up_exps.weight     — routed expert up projections for layer N
blk.{N}.ffn_down_exps.weight   — routed expert down projections for layer N
```

**Supported patterns**:
- `exps=CPU` — all routed experts to CPU (shorthand)
- `blk\.(0|1|2)\..*exps.*=CPU` — specific layers' experts to CPU
- `blk\.([0-9])\.=CUDA0` — layers 0-9 to GPU 0
- Layer-range regex for selective placement

**Limitation**: Cannot target individual expert indices within a layer. The 512 experts per layer are stored as a single merged tensor (`ffn_gate_exps` is shape `[512, 2048, 512]`). You can only move ALL experts in a layer to GPU or ALL to CPU.

### `--n-cpu-moe` Flag

Alternative to `-ot` for simpler configs:
- `--n-cpu-moe N` — offload N MoE layers' experts to CPU (counting from highest layer)
- Inverse: if model has 48 MoE layers and you set `--n-cpu-moe 42`, then 6 layers keep experts on GPU

---

## 2. VRAM Budget Analysis

### Current Usage (A2 config, all experts on CPU)

| Component | VRAM |
|-----------|------|
| Attention layers (48 layers) | ~1.5 GB |
| Shared experts (48 layers) | ~0.3 GB |
| Embeddings + output head | ~0.6 GB |
| Layer norms | ~0.1 GB |
| KV cache (32k ctx, q8_0) | ~1.5 GB |
| Compute buffers | ~2.2 GB |
| **Total** | **~6.2 GB** |
| **Available** | **~9.8 GB** |

### Expert Weight Sizes (Q8_0)

| Scope | Size |
|-------|------|
| 1 expert (gate+up+down) | ~3.15 MB |
| 10 active experts (1 layer) | ~31.5 MB |
| All 512 experts (1 layer) | ~1,612 MB (~1.57 GB) |
| All experts (48 layers) | ~77.4 GB |

### How Many Full Layers Fit in 9.8 GB?

Each full MoE layer's routed experts = ~1.57 GB (Q8_0).

- 6 layers: 6 x 1.57 = **9.4 GB** (fits, with ~400 MB margin)
- 7 layers: 7 x 1.57 = **11.0 GB** (does NOT fit)
- **Maximum: 6 full MoE layers on GPU** within current VRAM budget

### What If We Used KV Cache q4_0?

Switching KV cache from q8_0 to q4_0 would save ~750 MB:
- Available VRAM: ~10.55 GB
- Could fit 6 layers with comfortable margin, still not 7

### Individual Expert Caching (Hypothetical)

If llama.cpp supported per-expert caching instead of per-layer:
- 9.8 GB / 3.15 MB per expert = **~3,111 expert slots**
- That's 6.1x the total 512 experts, or ~64 experts per layer across 48 layers
- With top-10 routing and locality, a cache of ~64 experts per layer could achieve high hit rates
- But this requires code changes llama.cpp doesn't have yet

---

## 3. Static Partial Offloading Strategy

### Approach: Keep N Layers Fully on GPU

The most practical approach today is using `-ot` to keep 6 MoE layers with experts on GPU while offloading the remaining 42 layers' experts to CPU.

**Proposed config** (keeping first 6 layers on GPU):
```bash
./llama-server \
  -m ./Qwen3-Next-80B-A3B-Instruct-Q8_0.gguf \
  -c 32768 \
  -ngl 999 \
  -ot "blk\.([6-9]|[1-3][0-9]|4[0-7])\.ffn_(gate|up|down)_exps\.weight=CPU" \
  -fa on \
  -t 16 \
  -b 4096 \
  -ub 4096 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

Or equivalently, using the simpler approach:
```bash
-ot "exps=CPU" -ot "blk\.(0|1|2|3|4|5)\.ffn_(gate|up|down)_exps\.weight=CUDA0"
```

**Note**: The exact `-ot` override precedence needs testing. An alternative is `--n-cpu-moe 42` (offload 42 of 48 MoE layers' experts to CPU, keeping 6 on GPU).

### Expected Impact

- 6 of 48 MoE layers (12.5%) skip PCIe transfers entirely
- PCIe transfer reduction: ~12.5% fewer expert round-trips per token
- Those 6 layers run at full GPU speed instead of CPU+PCIe speed
- **Estimated speedup**: 5-15% (modest, since 87.5% of layers still use CPU path)

### Which Layers to Keep on GPU?

Options to consider:
1. **First layers (0-5)**: Process input representations; may have less expert diversity
2. **Last layers (42-47)**: Process final representations; may have more specialized experts
3. **GQA layers only**: The 12 GQA attention layers (every 4th: 3,7,11,...,47) are already more compute-intensive; keeping their adjacent MoE on GPU could help pipeline efficiency
4. **Empirical**: Benchmark each configuration

Research suggests later layers tend to have more expert specialization, so keeping the last 6 layers on GPU may provide marginal quality benefits, but performance-wise the impact should be similar regardless of which layers are chosen.

---

## 4. Dynamic Expert Caching (Not Yet in llama.cpp)

### What Would Be Needed

True expert caching would require:
1. **GPU-side expert weight cache**: Pre-allocated VRAM buffer for N expert slots
2. **Cache management**: LRU/LFU eviction policy tracking which experts are hot
3. **Execution path change**: When an expert is cached on GPU, compute there instead of sending activation to CPU
4. **Async prefetch**: Overlap expert weight transfers with computation of previous layers

### PR #17044: "Add MoE dynamic routing with expert caching"

- **Status**: OPEN (since 2025-11-06)
- **Description**: Empty body, no implementation, no comments
- **Labels**: documentation, build, examples
- **Assessment**: Appears to be a placeholder/proposal, not an active implementation

### Academic Systems

#### HOBBIT (Nov 2024)
- Built on llama.cpp (~8,000 lines of C++/C code modifications)
- Implements LRU, LFU, and LHU (Least High-precision Usage) cache policies
- Uses mixed precision: hot experts at high precision, cold experts at low precision
- Achieves up to **9.93x speedup** on Jetson Orin (edge device)
- Key finding: Experts selected for current token have elevated probability of reuse in next token
- **Not merged into mainline llama.cpp**

#### MoE-Infinity (Jan 2024)
- Activation-aware expert offloading with sequence-level tracing
- Uses K-Means clustering to build Expert Activation Matrices
- Achieves **4-20x latency reduction** vs baseline offloading
- Key finding: Only 3-20% of experts activated per sequence, 30-46% reused more than once
- **Separate system, not based on llama.cpp**

#### ProMoE (Oct 2024)
- Learned predictor for proactive expert prefetching
- Achieves **2.84x decode speedup** vs naive offloading
- Prediction accuracy: maintains high accuracy even 8 layers ahead
- Key finding: Modern MoE models have "low skewness" — experts accessed relatively uniformly
- Challenge: incorrect predictions waste bandwidth
- **Separate system**

#### Caching & Prefetching Analysis (Nov 2025, arXiv:2511.05814)
- LFU achieves **84.6% faster inference** than LRU on A6000
- Expert imbalance is stark: some experts activated only once during entire decoding
- Cache of 4-6 expert slots per layer is sufficient for good hit rates
- ~2 GB VRAM savings per additional offloaded expert layer
- Frequency-based selection substantially outperforms recency-based
- **Research paper only, no released system**

---

## 5. Expert Activation Locality

### General MoE Findings

Research consistently shows that MoE models exhibit **temporal locality** in expert activation:
- Experts activated for token N have elevated probability of activation for token N+1
- Within a single sequence, 30-46% of experts are reused more than once
- Some experts are activated far more frequently than others (imbalanced distribution)
- Locality is stronger within a single sequence than across sequences

### Qwen3-Next-80B Specifics

- 512 routed experts with top-10 routing per token per layer
- 10/512 = ~2% activation rate per expert per token
- With 48 MoE layers, each token activates 480 expert slots total (10 x 48)
- The model uses global-batch load balancing (encourages expert diversity during training)
- This may reduce locality compared to models without load balancing

### Implications for Caching

- With 512 experts and top-10 routing, the theoretical random hit rate for a cache of size C is: `1 - (502/512)^10` for C=10 experts cached
- For C=64 per layer: hit rate = `1 - (448/512)^10` = ~73%
- For C=128 per layer: hit rate = `1 - (384/512)^10` = ~95%
- Real-world hit rates would be **higher** due to temporal locality (same experts reused)
- But Qwen3-Next's load balancing may reduce the skew compared to DeepSeek-style models

---

## 6. Alternative Approaches

### A. GGML_OP_OFFLOAD_MIN_BATCH Tuning

llama.cpp has an environment variable `GGML_OP_OFFLOAD_MIN_BATCH` that controls when GPU handles prompt processing vs CPU. Default is 32 tokens. Tuning this could improve prompt processing speed but doesn't affect token generation.

### B. Expert Quantization

Instead of caching, quantize experts more aggressively to reduce transfer sizes:
- Q4_0 experts: ~1.58 MB each (vs 3.15 MB at Q8_0)
- Transfer per token per layer: ~15.8 MB (vs ~31.5 MB)
- Trades quality for speed — may be acceptable for some use cases
- Would require re-quantizing the GGUF

### C. NUMA-Aware Expert Placement

For multi-socket systems (not applicable to our single-socket setup), but worth noting:
- GitHub issue #11333 requests NUMA-aware expert allocation
- Could improve memory bandwidth if experts are on the same NUMA node as the processing core

### D. Custom Expert Cache Implementation

Building on HOBBIT's approach, one could:
1. Fork llama.cpp
2. Add a GPU-side expert weight buffer (~10 GB)
3. Implement LFU cache management per layer
4. Modify the expert dispatch to check cache before CPU fallback
5. Estimated effort: significant (~8,000 lines per HOBBIT)

---

## 7. Recommendations

### Immediate (No Code Changes)

1. **Test partial offloading**: Keep 6 MoE layers fully on GPU using `-ot` or `--n-cpu-moe 42`
   - Expected: 5-15% speedup (22 -> 23-25 tok/s)
   - Risk: minimal, easily reversible
   - Effort: one config change + benchmark

2. **Try different layer selections**: Test first-6 vs last-6 vs GQA-adjacent layers

### Medium-Term

3. **Monitor PR #17044**: If expert caching lands in llama.cpp, it could provide 2-5x speedup
4. **Evaluate HOBBIT**: If the project is released as a usable fork, test it
5. **Expert quantization**: If quality allows, re-quantize experts to Q4_0 for 2x less PCIe transfer

### Long-Term

6. **Custom expert caching**: Fork llama.cpp and implement GPU-side LFU cache
   - Potential: 2-10x speedup based on academic results
   - Effort: significant (weeks of C++/CUDA development)
   - VRAM budget allows ~3,100 expert slots — more than enough for high hit rates

---

## 8. Proposed Benchmark Plan

Test partial offloading configurations against current A2 baseline:

| Config | Description | Expected VRAM |
|--------|-------------|---------------|
| A2 (baseline) | All experts on CPU | ~6.2 GB |
| P1 | First 6 layers' experts on GPU | ~15.6 GB |
| P2 | Last 6 layers' experts on GPU | ~15.6 GB |
| P3 | GQA-adjacent layers (3,7,11,15,19,23) on GPU | ~15.6 GB |
| P4 | First 4 layers on GPU (safer margin) | ~12.5 GB |

Run each with the standard benchmark suite (`bash scripts/bench.sh`).

---

## References

- [llama.cpp MoE offloading guide (HuggingFace)](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [Understanding MoE Offloading (DEV Community)](https://dev.to/someoddcodeguy/understanding-moe-offloading-5co6)
- [llama.cpp GitHub Issue #11532: MoE expert loading feature request](https://github.com/ggml-org/llama.cpp/issues/11532)
- [llama.cpp Discussion #13154: -ot flag documentation](https://github.com/ggml-org/llama.cpp/discussions/13154)
- [llama.cpp PR #17044: MoE dynamic routing with expert caching](https://github.com/ggml-org/llama.cpp/pull/17044)
- [HOBBIT: Mixed Precision Expert Offloading (arXiv:2411.01433)](https://arxiv.org/html/2411.01433v2)
- [MoE-Infinity: Activation-Aware Expert Offloading (arXiv:2401.14361)](https://ar5iv.labs.arxiv.org/html/2401.14361)
- [ProMoE: Fast MoE Serving with Proactive Caching (arXiv:2410.22134)](https://arxiv.org/html/2410.22134v1)
- [Caching and Pre-Fetching in MoE Offloading (arXiv:2511.05814)](https://arxiv.org/pdf/2511.05814)
- [Qwen3-Next-80B-A3B-Instruct (HuggingFace)](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- [NVIDIA Blog: Qwen3-Next Hybrid MoE](https://developer.nvidia.com/blog/new-open-source-qwen3-next-models-preview-hybrid-moe-architecture-delivering-improved-accuracy-and-accelerated-parallel-processing-across-nvidia-platform/)
