# NVIDIA GeForce RTX 5080 -- Specifications, CUDA, and LLM Inference Analysis

**Date**: February 17, 2026

---

## 1. Hardware Specifications

### Core Configuration

| Specification | Value |
|---|---|
| **Architecture** | NVIDIA Blackwell (RTX variant) |
| **GPU Die** | GB203 |
| **Process Node** | TSMC 4N |
| **Die Size** | 378 mm^2 |
| **Transistor Count** | 45.6 billion |
| **GPCs** | 11 |
| **TPCs** | 42 |
| **Streaming Multiprocessors (SMs)** | 84 |
| **CUDA Cores** | 10,752 (128 per SM) |
| **Tensor Cores** | 336 (5th Generation, 4 per SM) |
| **RT Cores** | 84 (4th Generation, 1 per SM) |
| **Texture Units** | 336 |
| **ROPs** | 112 |
| **Compute Capability** | **sm_120** (12.0) |

### Memory Subsystem

| Specification | Value |
|---|---|
| **VRAM** | 16 GB GDDR7 |
| **Memory Bus Width** | 256-bit |
| **Memory Speed** | 30 Gbps effective |
| **Memory Bandwidth** | **960 GB/s** |
| **L2 Cache** | 64 MB |
| **L1 Cache / Shared Memory** | 128 KB per SM (configurable carveout) |
| **Register File** | 64K 32-bit registers per SM (256 KB) |

### Interface and Power

| Specification | Value |
|---|---|
| **PCIe Interface** | PCIe 5.0 x16 |
| **TDP** | 360W |
| **Base Clock** | 2,235 MHz |
| **Boost Clock** | 2,620 MHz |
| **Power Connector** | 16-pin (12V-2x6) |
| **Recommended PSU** | 700W+ |

### Compute Throughput

| Precision | Dense | With Sparsity (2:4) |
|---|---|---|
| **FP32 (CUDA cores)** | ~56.3 TFLOPS | N/A |
| **FP16 Tensor** | 112.5 TFLOPS | 225.1 TFLOPS |
| **BF16 Tensor** | 112.5 TFLOPS | 225.1 TFLOPS |
| **FP8 Tensor** | 225.1 TFLOPS | 450.2 TFLOPS |
| **INT8 Tensor** | ~900 TOPS | ~1,801 TOPS |
| **FP4 Tensor** | ~450.2 TFLOPS | ~900.4 TFLOPS |

The "1,801 AI TOPS" marketing figure = INT8 with sparsity.

---

## 2. Blackwell Architecture: What's New for LLM Inference

### 5th-Generation Tensor Cores

- **FP4 (4-bit floating point)**: Blackwell-exclusive. 2x density of FP8, doubles theoretical throughput for compatible workloads.
- **FP6 (6-bit floating point)**: New intermediate precision between FP4 and FP8.
- **2nd-Gen FP8 Transformer Engine**: Improved FP8 (E4M3/E5M2) support.
- **Block-scaled quantization**: Hardware support for block-scaled FP4 and FP8 matmul.

### Supported Tensor Core Data Types

FP16, BF16, TF32, FP8 (E4M3, E5M2), FP6, FP4, INT8, INT4.

Ada Lovelace (RTX 40-series) lacked FP4 and FP6.

### Memory Improvements

- **GDDR7**: 960 GB/s on 256-bit bus (PAM3 signaling)
- **L2 Cache**: 64 MB
- **256-bit loads/stores**: New aligned vector types for wider memory transactions

### Architecture Differences (sm_120 vs sm_100)

- **Shared Memory per SM**: 128 KB (sm_120) vs 228 KB (sm_100 data center)
- **Max warps per SM**: 48 (sm_120) vs 64 (sm_100)
- **No NVLink** on consumer GeForce (multi-GPU via PCIe 5.0 only)

---

## 3. CUDA 13.1 Details

### CUDA Toolkit Version History for Blackwell

| CUDA Version | Blackwell Support | Notes |
|---|---|---|
| **12.8** | First native Blackwell | sm_100, sm_101, sm_120 targets |
| **12.9** | Improved | Block-scaled FP4/FP8 matmul |
| **13.0** | Full optimization | Auto-tuning cuBLAS, 256-bit vector types |
| **13.1** | Current release | CUDA Tile, Grouped GEMM for MoE |

### Driver Requirements

| CUDA Toolkit | Minimum Linux Driver |
|---|---|
| CUDA 13.1 Update 1 | >= 590.48.01 |
| CUDA 13.1 GA | >= 590.44.01 |
| CUDA 13.0 Update 2 | >= 580.95.05 |

**Current system: driver 590.48.01 = exact minimum for CUDA 13.1 Update 1.**

### Key CUDA 13.x Features for LLM Inference

**CUDA 13.0:**
- **cuBLAS Auto-tuning**: `CUBLAS_GEMM_AUTOTUNE` flag -- benchmarks and caches optimal GEMM algorithms
- **256-bit aligned vector types**: Wider load/store paths for memory-bound kernels
- **Fatbin compression**: LZ4 to Zstandard, 17% smaller binaries

**CUDA 13.1:**
- **CUDA Tile**: Tile-based programming model with Python DSL (`cuTile`), Blackwell-only
- **cuBLAS Grouped GEMM**: FP8 and BF16/FP16 for Blackwell, **up to 4x speedup** for MoE models
- **cuDNN**: Improved fused scaled dot-product attention for inference decoder

---

## 4. Docker + NVIDIA Container Toolkit

### Recommended Base Images

```
# Best: CUDA 13.1 devel (includes nvcc, headers, libraries)
nvidia/cuda:13.1.1-devel-ubuntu24.04

# Alternative: CUDA 12.8 devel (minimum for sm_120)
nvidia/cuda:12.8.0-devel-ubuntu24.04
```

Image variants:
- **base**: Minimal CUDA runtime
- **runtime**: Math libraries (cuBLAS, cuFFT, cuSPARSE, NCCL) + cuDNN
- **devel**: Everything + nvcc compiler (needed for llama.cpp)

### NVIDIA Container Toolkit Notes (v1.18.2)

- **CDI mode**: v1.18+ uses CDI (Container Device Interface) by default. May cause library-not-found errors.
- **Workaround**: Add CUDA library paths to `LD_LIBRARY_PATH` or use `--runtime=nvidia` for legacy mode.
- **Driver compatibility**: Host driver (590.48.01) is what matters. Any container CUDA <= 13.1 works.

### Recommended Docker Build Command

```bash
docker run --gpus all \
  -v /path/to/llama.cpp:/workspace \
  nvidia/cuda:13.1.1-devel-ubuntu24.04 \
  bash -c "cd /workspace && make GGML_CUDA=1 CUDA_DOCKER_ARCH=sm_120 -j$(nproc)"
```

---

## 5. Memory Bandwidth Comparison

| GPU | Memory Type | Bus Width | Bandwidth | VRAM |
|---|---|---|---|---|
| **RTX 5080** | GDDR7 | 256-bit | **960 GB/s** | 16 GB |
| RTX 4080 | GDDR6X | 256-bit | 716.8 GB/s | 16 GB |
| RTX 4090 | GDDR6X | 384-bit | 1,008 GB/s | 24 GB |
| RTX 5090 | GDDR7 | 512-bit | 1,792 GB/s | 32 GB |

- RTX 5080 vs RTX 4080: **+34% bandwidth**
- RTX 5080 vs RTX 4090: **-4.8% bandwidth** (but 4090 has wider 384-bit bus + 8GB more VRAM)

### Theoretical Token Generation (Models Fitting in VRAM)

Using ~768 GB/s effective (80% of 960 GB/s):

| Model | Quant | Size | Theoretical t/s |
|---|---|---|---|
| Llama 3.1 8B | Q4_K_M | ~4.9 GB | ~157 |
| Llama 3.1 8B | Q8_0 | ~8.5 GB | ~90 |
| Mistral 7B | Q4_K_M | ~4.4 GB | ~175 |
| DeepSeek-R1 14B | Q4_K_M | ~8.7 GB | ~88 |

### Measured Benchmarks

**llama.cpp on RTX 5080:**

| Model | Quant | Prompt (pp512) | Token Gen (tg128) |
|---|---|---|---|
| Llama 2 7B | Q4_0 | 9,488 t/s (FA=1) | 185 t/s (FA=1) |

**UL Procyon AI benchmark:**

| Model | RTX 5080 | RTX 4090 | RTX 5090 |
|---|---|---|---|
| Phi | 209.5 | 244.3 | 314.4 |
| Mistral | 163.6 | 183.3 | 255.9 |
| Llama 3 | 136.2 | 150.0 | 214.3 |
| Llama 2 | 83.7 | 92.9 | 134.5 |

---

## 6. PCIe 5.0 x16 for MoE Offloading

| Metric | Value |
|---|---|
| **Theoretical bandwidth** | ~64 GB/s bidirectional |
| **Practical bandwidth** | ~50-55 GB/s |
| **Expert transfer per token (Q8_0)** | ~1.51 GB |
| **Transfer time per token** | ~23.6 ms (at 64 GB/s) |
| **Theoretical ceiling** | ~42 t/s (transfer-limited) |
| **Realistic estimate** | 15-30 t/s (with overhead) |

PCIe 5.0 is a significant advantage:
- PCIe 4.0 x16 (~32 GB/s) would halve to ~8-15 t/s
- PCIe 3.0 x16 (~16 GB/s) would be ~4-8 t/s

---

## Sources

- [NVIDIA GeForce RTX 5080 Product Page](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/)
- [GeForce RTX 50 Series - Wikipedia](https://en.wikipedia.org/wiki/GeForce_RTX_50_series)
- [NVIDIA CUDA GPU Compute Capability List](https://developer.nvidia.com/cuda/gpus)
- [CUDA Toolkit 13.1 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
- [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [What's New in CUDA 13.0](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/)
- [CUDA 13.1 Announcement](https://developer.nvidia.com/blog/nvidia-cuda-13-1-powers-next-gen-gpu-programming-with-nvidia-cuda-tile-and-performance-gains/)
- [llama.cpp CUDA Performance Discussion](https://github.com/ggml-org/llama.cpp/discussions/15013)
- [StorageReview RTX 5080 Review](https://www.storagereview.com/review/nvidia-geforce-rtx-5080-review-the-sweet-spot-for-ai-workloads)
- [RTX Blackwell GPU Architecture Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf)
