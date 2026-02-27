# Community Follow-up Research: Ollama, Pre-built Binaries, Bartowski Quants

**Date**: 2026-02-26
**Session**: 006 — Community Follow-up

---

## Topic 1: Why Ollama Is Much Slower Than llama.cpp for MoE Offloading

### Context

A Reddit user (u/InternationalNebula7) reported Ollama gets only 21.6 tok/s on Qwen3.5-35B-A3B Q4_K_M on an RTX 5080, while our llama.cpp config gets ~70 tok/s on the same hardware for a similar-class MoE model. Why the 3x gap?

### Root Cause: Ollama Does Not Support MoE Expert Offloading

The single biggest factor is that **Ollama has no equivalent of `-ot "exps=CPU"`**. When you run a MoE model in Ollama, it treats it like a dense model for layer-splitting purposes. If the full model doesn't fit in VRAM, Ollama splits at the layer level — some layers go to GPU, some to CPU. This is catastrophic for MoE models because:

- Dense layer splitting means **entire transformer blocks** (attention + FFN + experts) get pushed to CPU
- With expert-only offloading (`-ot "exps=CPU"`), only the routed expert FFN weights live in RAM while attention, shared experts, embeddings, and norms stay on GPU
- The active parameter count per token in a MoE model is tiny (~3-4B), so the actual compute on the expert weights is small — the bottleneck is PCIe transfer, not compute
- Expert-only offloading keeps the GPU busy on attention/norms while PCIe transfers happen, vs. full-layer offloading where the GPU sits idle waiting for CPU layers

**Evidence**: A user on GitHub issue [ollama/ollama#11772](https://github.com/ollama/ollama/issues/11772) demonstrated that running gpt-oss:120b with `--n-cpu-moe 24` in llama.cpp achieved **3.5x the performance** compared to Ollama's approach (29.4 tok/s vs 8.5 tok/s).

### Does Ollama Support `-ot "exps=CPU"` or `--n-cpu-moe`?

**No.** Neither flag is available in Ollama. There is an open PR ([ollama/ollama#12333](https://github.com/ollama/ollama/pull/12333)) that would add a `num_moe_offload` parameter to Modelfiles, but it has **not been merged** as of February 2026. The Ollama maintainers have reservations:
- They prefer automatic memory-based optimization over manual user configuration
- Multi-GPU support in the PR is inadequate
- They generally avoid adding features to the legacy llama engine (Ollama is moving toward its own inference backend)

### Can You Pass Custom llama.cpp Flags Through Ollama?

**No.** Ollama is not a thin wrapper around llama.cpp. It embeds llama.cpp as a library via CGO, and newer models (e.g., Gemma 3) are implemented directly in Ollama's own engine. There is no `OLLAMA_RUNNER_OPTS` or similar mechanism to pass arbitrary llama.cpp CLI flags.

### Ollama's Suboptimal Defaults

Even ignoring MoE offloading, Ollama's defaults leave performance on the table:

| Setting | Ollama Default | Our llama.cpp Config | Impact |
|---------|---------------|---------------------|--------|
| Context length | 4096 | 32768 | Not a speed factor, but affects capability |
| KV cache type | f16 | q8_0 | **~20% throughput penalty + more VRAM** |
| Flash attention | Off by default (recently enabled for some models) | `-fa on` | Slower attention, more VRAM for KV cache |
| Thread count | Auto (all cores) | 20 (tuned) | Sub-optimal; t=all-cores can be worse than tuned value |
| Batch size | Not configurable | 4096 | Unknown impact, but Ollama provides no control |
| Expert offloading | Not supported | `-ot "exps=CPU"` | **3x throughput difference for MoE models** |

### Key Takeaway

The 3x speed gap is almost entirely explained by Ollama's lack of MoE expert offloading. KV cache quantization (f16 vs q8_0) and flash attention account for another ~20%. For MoE models on consumer GPUs, llama.cpp with expert offloading is currently the only way to get good performance.

### Sources

- [Ollama MoE offloading feature request — ollama/ollama#11772](https://github.com/ollama/ollama/issues/11772)
- [Ollama MoE offloading PR (open, not merged) — ollama/ollama#12333](https://github.com/ollama/ollama/pull/12333)
- [Ollama FAQ — default settings](https://docs.ollama.com/faq)
- [Bringing K/V Context Quantisation to Ollama](https://smcleod.net/2024/12/bringing-k/v-context-quantisation-to-ollama/)
- [Ollama vs llama.cpp speed comparison (26.8% overhead measured)](https://huggingface.co/posts/mitkox/389008233017077)
- [Ollama vs llama.cpp speed discussion — ollama/ollama#11259](https://github.com/ollama/ollama/issues/11259)
- [HuggingFace MoE offloading guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)
- [Switching from Ollama to llama.cpp](https://www.nijho.lt/post/llama-nixos/)

---

## Topic 2: Pre-built vs Source-built llama.cpp Performance

### Context

A Reddit user (u/wisepal_app) asks whether pre-built Windows binaries (`llama-b8149-bin-win-cuda-12.4-x64.zip`) perform the same as building from source.

### What CUDA Architectures Do the Release Binaries Include?

The llama.cpp GitHub Actions release workflow builds Windows CUDA binaries with:

```
cmake -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_CPU=OFF
```

**No explicit `CMAKE_CUDA_ARCHITECTURES` is set.** When `GGML_NATIVE=OFF` and no architecture is specified, llama.cpp's CMake logic falls back to a tiered list based on CUDA toolkit version:

| CUDA Version | Architectures Included |
|-------------|----------------------|
| < 13.0 | `50-virtual, 61-virtual, 70-virtual` |
| All versions | `75-virtual, 80-virtual, 86-real` |
| >= 11.8 | `+ 89-real` |
| >= 12.8 | `+ 120a-real` (Blackwell) |
| >= 12.9 | `+ 121a-real` |

**Key finding**: The release binaries are built with **CUDA 12.4**, which means:
- Included: `50-virtual, 61-virtual, 70-virtual, 75-virtual, 80-virtual, 86-real, 89-real`
- **NOT included: `120a-real` (Blackwell/sm_120)** — requires CUDA >= 12.8

### Does Building with `-DCMAKE_CUDA_ARCHITECTURES=<your_gpu>` Give Measurable Speedups?

**Yes, but the reasons are nuanced:**

1. **`-real` vs `-virtual` architecture matters**: A `-virtual` architecture means PTX code is included and JIT-compiled at runtime. A `-real` architecture means native SASS code is pre-compiled. JIT compilation adds startup latency (first load) but should produce equivalent runtime performance after compilation. However, the JIT compiler may not optimize as aggressively as build-time compilation.

2. **For Blackwell GPUs (sm_120) with CUDA 12.4 binaries**: The pre-built binaries **do not include sm_120** at all. CUDA's forward compatibility means PTX from sm_89 can be JIT-compiled for sm_120, but:
   - First launch will be slow (JIT compilation)
   - The JIT may not use Blackwell-specific instructions (e.g., native MXFP4 Tensor Core ops)
   - Any Blackwell-specific CUDA kernels compiled with `#if __CUDA_ARCH__ >= 1200` guards will be absent

3. **Measured overhead**: General benchmarks suggest building from source with the exact architecture gives **0-5% speedup in steady-state token generation** but can significantly reduce first-load latency. The bigger win comes from:
   - Using a newer CUDA toolkit (12.8+) that has Blackwell-optimized libraries (cuBLAS, etc.)
   - Enabling architecture-specific features at compile time (flash attention all quants, etc.)

### Compile-Time Optimizations That Differ

| Optimization | Release Binary | Custom Build | Impact |
|-------------|---------------|-------------|--------|
| CUDA architecture | Multi-arch (virtual PTX) | Native SASS for your GPU | 0-5% steady-state, faster first load |
| CUDA toolkit version | 12.4 | 12.8+ available | Newer cuBLAS, Blackwell support |
| `GGML_CUDA_FA_ALL_QUANTS` | Unknown (likely OFF) | Can enable | Flash attention for all quant types |
| CPU features (`GGML_NATIVE`) | OFF (generic x86-64) | ON (uses AVX-512, etc.) | 10-30% for CPU-bound operations |
| Link-time optimization | Likely OFF | Can enable `-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON` | Minor (~2-5%) |

### For a Blackwell GPU Specifically

**Building from source is strongly recommended:**
1. Use CUDA 12.8+ to get `120a-real` architecture
2. Enable `GGML_CUDA_FA_ALL_QUANTS` for flash attention on all quant types
3. Set `CMAKE_CUDA_ARCHITECTURES=120` for native SASS code
4. Future Blackwell-specific optimizations (native MXFP4, grouped GEMM) will only be available with CUDA 13.x builds

### Key Takeaway

For most users on RTX 30xx/40xx GPUs, pre-built binaries are **fine** — the performance difference is small (0-5%). For Blackwell GPU users (RTX 50xx), building from source with CUDA 12.8+ is **important** because the CUDA 12.4 binaries don't include sm_120 native code at all. The biggest win isn't the architecture flag itself but using a newer CUDA toolkit with Blackwell-optimized libraries.

### Sources

- [llama.cpp build documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [llama.cpp CUDA CMakeLists.txt — architecture defaults](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-cuda/CMakeLists.txt)
- [llama.cpp release workflow](https://github.com/ggml-org/llama.cpp/blob/master/.github/workflows/release.yml)
- [NVIDIA Blackwell migration guide for llama.cpp](https://forums.developer.nvidia.com/t/software-migration-guide-for-nvidia-blackwell-rtx-gpus-a-guide-to-cuda-12-8-pytorch-tensorrt-and-llama-cpp/321330)
- [Building llama-cpp-python for Blackwell](https://github.com/abetlen/llama-cpp-python/issues/2028)
- [llama.cpp CUDA performance discussion](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [Pre-built llama.cpp CUDA binaries (community)](https://github.com/ai-dock/llama.cpp-cuda)

---

## Topic 3: Bartowski Q4_K_L Quant and Unsloth UD-Q4_K_XL vs MXFP4_MOE

### Context

A user (u/bettertoknow) suggested Bartowski's Q4_K_L has better KLD/PPL than Q4_K_M and showed tensor breakdowns. They also revealed that UD-Q4_K_XL has 275 mxfp4 tensors — more than the dedicated MXFP4_MOE quant (120 mxfp4 tensors).

### Does Bartowski Q4_K_L Exist for Qwen3.5-35B-A3B?

**Yes.** Available at [bartowski/Qwen_Qwen3.5-35B-A3B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF).

| Quant | File Size | Description |
|-------|-----------|-------------|
| Q4_K_L | 21.60 GB | Uses Q8_0 for embed and output weights |
| Q4_K_M | 21.23 GB | Standard good quality, recommended default |
| Q4_K_S | 20.44 GB | Slightly lower quality, more space savings |

### Why Is Q4_K_L Better Than Q4_K_M?

The "L" (Large) suffix in llama.cpp K-quants means **less aggressive quantization** — more tensors are kept at higher precision:

- **Q4_K_M**: Most tensors at Q4_K, with "important" matrices (attention value/output projections, FFN down projections) upcasted to Q6_K. Embeddings and output head use default precision.
- **Q4_K_L**: Same as Q4_K_M but with **embeddings (`token_embd.weight`) and output head (`output.weight`) quantized at Q8_0** instead of a lower default.

The embeddings and output head are the model's "interface" with the token vocabulary. Quantizing them aggressively (Q4/Q5) introduces error that propagates through every token. Using Q8_0 for these tensors is essentially free — they're small relative to the total model — but meaningfully improves output quality, especially for:
- Token selection accuracy (the output head directly produces logits)
- Embedding representation fidelity (affects every layer downstream)

The size difference is small: Q4_K_L is only 370 MB larger than Q4_K_M (21.60 vs 21.23 GB), a ~1.7% increase for better quality on the most sensitive tensors. **There is no speed difference** — Q8_0 embeddings don't affect generation speed because they're only accessed once per token.

### Published KLD/PPL Data

Bartowski's HuggingFace model card for Qwen3.5-35B-A3B-GGUF **does not include KLD or PPL data**. It references [Artefact2's quantization overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9) for general quality comparisons, but model-specific measurements are not published.

For the Llama-3-8B reference model (from llama.cpp's quantize README):
- Q4_K_M: +0.1754 PPL degradation
- Q4_K_L: Not listed in the official README (it's a newer addition)

The user's claim about "better KLD/PPL" is likely based on either: (a) their own measurements, (b) general K-quant behavior where L variants consistently beat M variants, or (c) tensor-level analysis showing the Q8_0 embed/output tensors reduce KL divergence.

### UD-Q4_K_XL (275 mxfp4 tensors) vs MXFP4_MOE (120 mxfp4 tensors)

This is counterintuitive but **intentional**, not a bug. Here's why:

**MXFP4_MOE** is a specialized format that:
- Uses native MXFP4 format **only for MoE expert FFN weights** (gate/up/down projections in routed experts)
- Keeps everything else (attention, shared experts, embeddings, norms) at higher precision (Q8_0 or f16)
- The "120 mxfp4 tensors" corresponds to the expert FFN layers only
- Optimized for Blackwell GPUs with native FP4 Tensor Core hardware

**UD-Q4_K_XL** (Unsloth Dynamic) is a calibration-based format that:
- Uses Unsloth's proprietary calibration dataset (300K-1.5M tokens) to determine which tensors can safely use lower precision
- Dynamically assigns each tensor to the best quantization type (Q4_K, Q5_K, Q8_0, or MXFP4)
- The "275 mxfp4 tensors" means Unsloth's calibration determined that **275 tensors** (including non-expert tensors like some attention projections) can safely use MXFP4 without quality loss
- It applies MXFP4 more broadly because calibration proved those specific tensors tolerate it

**Why more MXFP4 in UD-Q4_K_XL?**

MXFP4_MOE is conservative — it only applies MXFP4 to expert weights because that's the safe, architecture-driven choice. UD-Q4_K_XL is empirical — it tested every tensor and found that many non-expert tensors also tolerate MXFP4. The dynamic calibration process can discover that, e.g., certain attention Q/K projections or shared expert weights can use MXFP4 without measurable quality loss.

The tradeoff:
- **MXFP4_MOE**: Conservative, predictable, slightly faster on Blackwell (optimized memory layout)
- **UD-Q4_K_XL**: Empirically optimized, potentially better quality-per-bit, uses Q5_K for "important" matrices where MXFP4 would hurt

From the [Unsloth discussion](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/6), the Unsloth team confirms: "Q4_K_XL actually has some tensors in MXFP4 as well — MXFP4 is partially dynamic as well — so both are fine." They recommend **always using the XL quants for better results**.

### Practical Recommendation

For Qwen3.5-35B-A3B on an RTX 5080:
1. **Bartowski Q4_K_L** (21.6 GB) — best standard K-quant choice. Free quality improvement over Q4_K_M for negligible size increase.
2. **Unsloth UD-Q4_K_XL** — if you want the best possible 4-bit quality via dynamic calibration. Slightly different tensor allocation than standard K-quants.
3. **MXFP4_MOE** — only worthwhile if running on Blackwell with native MXFP4 acceleration. Otherwise UD-Q4_K_XL is better.

### Sources

- [bartowski/Qwen_Qwen3.5-35B-A3B-GGUF on HuggingFace](https://huggingface.co/bartowski/Qwen_Qwen3.5-35B-A3B-GGUF)
- [Unsloth Dynamic 2.0 GGUFs documentation](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs)
- [Unsloth Dynamic v2.0 blog post](https://unsloth.ai/blog/dynamic-v2)
- [MXFP4 vs Q4_K_XL discussion — Qwen3.5-397B](https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF/discussions/1)
- [UD-Q4_K_XL vs Q4_K_M discussion — Qwen3-30B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/discussions/6)
- [Artefact2 GGUF quantization overview](https://gist.github.com/Artefact2/b5f810600771265fc1e39442288e8ec9)
- [llama.cpp quantize tool README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [Qwen3 MoE quant benchmarking roundup](https://gist.github.com/ubergarm/0f9663fd56fc181a00ec9f634635eb38)
