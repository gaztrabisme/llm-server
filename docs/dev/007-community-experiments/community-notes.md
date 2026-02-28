# Community Hardware Notes: Qwen3.5-35B-A3B on Non-5080 Hardware

Model: **Qwen3.5-35B-A3B** (Mixture-of-Experts, 256 experts per layer, top-8 routing + 1 shared expert, 40 MoE layers, ~3B active params per token)

Production quant: **Q4_K_M** (~20 GB) | Reference quant: **Q8_0** (~37 GB)

Our benchmarks were run on an NVIDIA RTX 5080 16GB with CUDA. This document provides guidance for users on different hardware.

---

## Key Principle: Why MoE Offloading Is the Main Lever

Mixture-of-Experts models are fundamentally different from dense models when it comes to GPU offloading. In a dense model, every parameter is used for every token, so the only split is at the layer level -- entire transformer blocks go to GPU or CPU. In an MoE model, each layer has hundreds of expert sub-networks, but only a handful are activated per token (Qwen3.5-35B-A3B activates 8 of 256 routed experts + 1 shared expert per layer).

This means the **expert weights** are by far the largest part of the model, but most of them sit idle on any given token. The critical optimization is to keep the **attention layers, norms, and shared experts on GPU** (these are always active) while letting the routed expert weights live on CPU and transfer over PCIe only when selected by the router.

The flags that control this:

- `--fit on` -- llama.cpp automatically determines the optimal GPU/CPU split for expert layers based on available VRAM (CUDA only, see caveats below)
- `--n-cpu-moe N` -- manually specify how many expert layers run on CPU (works on all backends)
- `-ot "exps=CPU"` -- force all expert weights to CPU, keeping only attention/norms/shared on GPU (minimal VRAM, still faster than full CPU)
- `-ngl N` -- offload N full transformer layers to GPU (coarse-grained, not MoE-aware)

The difference is massive. On our RTX 5080 16GB with Q4_K_M:

| Strategy | tok/s | VRAM |
|----------|-------|------|
| Full CPU (`-ngl 0`) | ~35 | minimal |
| Expert offload (`-ot "exps=CPU"`) | ~51 | ~7.2 GB |
| Manual split (`--n-cpu-moe 24`) | ~70 | ~14.9 GB |
| Auto split (`--fit on`, no -b/-ub) | ~74 | ~14.6 GB |

The lesson: **every MoE expert layer you keep on GPU adds throughput**. The goal on any hardware is to maximize the number of expert layers that fit in your VRAM.

---

## AMD/ROCm Guidance

**Source**: u/Psyko38, RX 7900 XTX 24GB

`--fit on` significantly underperforms on ROCm. Psyko38 reports **15.82 tok/s with `--fit on`** compared to **37.74 tok/s with manual offloading** -- a 2.4x difference. The `--fit` auto-detection algorithm is CUDA-specific and does not produce good splits on AMD hardware.

**Recommendation**: Use manual offloading with `-ngl 999 --n-cpu-moe 24`. This is the same approach as our Session 005 C7 config, which was the best manual config on NVIDIA as well before `--fit on` was available.

**Recommended launch command** (RX 7900 XTX 24GB, Q4_K_M):

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 \
  -ngl 999 \
  --n-cpu-moe 24 \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Expected speed**: ~38 tok/s on RX 7900 XTX 24GB (from Psyko38's data).

**Notes**:
- `-ngl 999` offloads all transformer layers to GPU. With 24GB VRAM, most or all layers should fit.
- `--n-cpu-moe 24` keeps 24 of 40 MoE layers' experts on CPU. Tune this value: try lowering it (e.g., 20, 16) if you have VRAM headroom, or raising it if you hit OOM.
- ROCm builds of llama.cpp may not support all flags. Check your build's `--help` output.
- Flash attention (`-fa on`) and KV cache quantization (`-ctk q8_0 -ctv q8_0`) should work on ROCm but verify with your build.
- Thread count (`-t`) is hardware-dependent. Start with `physical_cores / 1.5` and sweep a few values.

---

## Vulkan Guidance

**Source**: u/Corosus, RTX 3090 24GB via Vulkan backend

`--fit on` is broken on Vulkan. Corosus reports **13 tok/s with `--fit on`** compared to **33 tok/s with manual offloading** -- a 2.5x difference. The Vulkan backend does not support the auto-detection logic that `--fit` relies on.

**Recommendation**: Use manual layer offloading with `-ngl` and `--n-cpu-moe`. Do not use `--fit on`.

**Recommended launch command** (RTX 3090 24GB, Vulkan, Q4_K_M):

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 \
  -ngl 999 \
  --n-cpu-moe 24 \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Expected speed**: ~33 tok/s on RTX 3090 24GB via Vulkan (from Corosus's data).

**Notes**:
- If you have an NVIDIA GPU and are using Vulkan instead of CUDA, consider switching to a CUDA build. CUDA is significantly better optimized for llama.cpp, especially for MoE expert offloading.
- Vulkan is the right choice if you do not have CUDA support (e.g., Intel GPUs, or NVIDIA GPUs on non-CUDA-capable systems).
- Start with `--n-cpu-moe 24` and tune: lower means more experts on GPU (faster but more VRAM), higher means more on CPU (slower but fits).
- Flash attention and KV cache quantization support on Vulkan may vary. If you encounter errors, try removing `-fa on` and/or the `-ctk`/`-ctv` flags.

---

## 8GB VRAM Guidance

With only 8GB of VRAM, Q4_K_M (~20 GB) cannot fit enough expert layers on the GPU for partial offload to be effective. The best strategy is full expert offload, which keeps attention, norms, and shared experts on GPU (~7.2 GB) while running all routed experts on CPU.

**Recommendation**: Full expert offload with `-ot "exps=CPU"`.

**Recommended launch command** (8GB VRAM, Q4_K_M):

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 8192 \
  -ot "exps=CPU" \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Expected speed**: ~35-50 tok/s depending on RAM bandwidth.
- DDR5-5600+: upper end (~45-50 tok/s)
- DDR4-3200: lower end (~35-40 tok/s)
- RAM bandwidth is the bottleneck here since all expert computation happens on CPU.

**Notes**:
- Context length is reduced to 8192. At 65k context with KV q8_0, the KV cache alone would consume several GB of VRAM, leaving no room for model layers. Use `-c 8192` or `-c 16384` based on your needs; 65k is not practical at 8GB.
- `-ot "exps=CPU"` is more granular than `-ngl 0`. It keeps the attention and norm weights on GPU (which are small but accessed every token), only moving the large expert FFN weights to CPU.
- If even `-ot "exps=CPU"` does not fit (due to long context KV cache), fall back to `-ngl 0` for full CPU inference.
- **Alternative quants**: If you want a smaller model to enable some partial GPU offload, consider Q4_0 (~17.5 GB) or IQ4_XS (~16.5 GB). These trade quality for size and may allow a few expert layers on GPU within 8GB.
- Thread count matters more here since the CPU is doing most of the work. Sweep `-t` values: try 8, 12, 16, 20, and your physical core count.

---

## 24GB VRAM Guidance

With 24GB of VRAM, Q4_K_M (~20 GB) should fit most or all expert layers on the GPU. This is the sweet spot for this model -- you have enough VRAM for aggressive offloading without the constraints of 16GB.

**Recommendation**: Use `--fit on` (CUDA only) or manual offload with a low `--n-cpu-moe` value.

**Recommended launch command** (24GB VRAM, CUDA, Q4_K_M):

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 65536 \
  --fit on \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

**Expected speed**: ~90-110 tok/s (extrapolation).

This estimate is based on our 16GB data: `--fit on` places approximately 16 of 40 MoE layers on GPU at 14.6 GB VRAM, yielding ~74 tok/s. With 24GB, `--fit` should place 30-40 of 40 layers on GPU, significantly reducing PCIe transfers. The relationship is not perfectly linear (diminishing returns as more layers are on GPU), but 90-110 tok/s is a reasonable range.

**Alternative -- Q8_0 with partial offload** (24GB VRAM, CUDA):

```bash
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q8_0.gguf \
  -c 65536 \
  --fit on \
  -fa on \
  -t 20 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

Q8_0 (~37 GB) will not fit entirely in 24GB, but `--fit on` should place roughly 10-15 of 40 MoE layers on GPU. Expected speed: ~50-65 tok/s (similar to our 16GB Q4_K_M expert offload configs). The quality is measurably better (PPL 6.5342 vs 6.6688), so if you value quality over speed, this is a viable option.

**Notes**:
- If you are on AMD/ROCm or Vulkan with 24GB, use `-ngl 999 --n-cpu-moe 8` instead of `--fit on`. Start with `--n-cpu-moe 8` and try lower values (4, 0) to see how many expert layers fit.
- Do not use `-b/-ub` batch flags with `--fit on` -- they consume VRAM that `--fit` needs for expert layers (this was our key Session 006 finding).
- With 24GB, you might also try `--fit on` without KV cache quantization (`-ctk f16 -ctv f16`) for maximum quality, since you have more VRAM headroom. However, KV q8_0 is nearly free in quality terms and still saves VRAM for expert layers, so we recommend keeping it.

---

## LM Studio Limitations

LM Studio uses llama.cpp as its inference backend but wraps it in a GUI that does not expose all command-line flags.

**Key limitations**:
- `--fit on` may not be available in the LM Studio interface. This is the single most impactful flag for MoE performance on NVIDIA GPUs.
- `--n-cpu-moe` may not be exposed. Without it, LM Studio falls back to layer-level offloading (`-ngl`), which is far less efficient for MoE models (see the "Key Principle" section above).
- `-ot "exps=CPU"` (per-tensor offloading) is unlikely to be exposed.
- KV cache quantization (`-ctk`/`-ctv`) may or may not be available depending on the LM Studio version.
- Batch size flags (`-b`/`-ub`) may be auto-configured by LM Studio in ways that conflict with `--fit on` optimization.

**Recommendation**: For MoE models on consumer GPUs, use `llama-server` directly for full control over offloading flags. The performance difference can be 2-3x compared to default LM Studio settings.

If you prefer using LM Studio:
- Set GPU layers (`-ngl`) to the maximum your VRAM allows.
- If `--n-cpu-moe` is available in advanced settings, use it (start with 24 and tune down).
- Enable flash attention if available.
- Reduce context length if you are running out of VRAM.

**Recommended launch command** (direct llama-server, any hardware):

```bash
# Generic starting point -- adjust -t, --n-cpu-moe, and -c for your hardware
./llama-server \
  -m ./Qwen3.5-35B-A3B-Q4_K_M.gguf \
  -c 32768 \
  -ngl 999 \
  --n-cpu-moe 24 \
  -fa on \
  -t 16 \
  --no-mmap \
  --jinja \
  -ctk q8_0 \
  -ctv q8_0
```

Tune `--n-cpu-moe` down (fewer on CPU = faster) until you hit VRAM limits. Tune `-t` by sweeping a few values around `physical_cores / 1.5`.

---

## Quick Reference: Choosing Your Config

| Your Hardware | Backend | Recommended Strategy | Key Flags | Expected tok/s |
|---------------|---------|---------------------|-----------|---------------|
| NVIDIA 8GB | CUDA | Full expert offload | `-ot "exps=CPU"` `-c 8192` | 35-50 |
| NVIDIA 16GB | CUDA | Auto split | `--fit on` (no -b/-ub) | ~74 |
| NVIDIA 24GB | CUDA | Auto split | `--fit on` (no -b/-ub) | 90-110 |
| NVIDIA 24GB | CUDA | Q8_0 partial offload | `--fit on` (no -b/-ub) | 50-65 |
| AMD 24GB | ROCm | Manual split | `-ngl 999 --n-cpu-moe 24` | ~38 |
| Any GPU | Vulkan | Manual split | `-ngl 999 --n-cpu-moe 24` | ~33 |
| CPU only | N/A | Full CPU | `-ngl 0` | 20-35 |

All configs should include: `-fa on --no-mmap --jinja -ctk q8_0 -ctv q8_0`

Speed estimates for NVIDIA 24GB and CPU-only are extrapolations. AMD and Vulkan numbers are from community reports. Your mileage will vary based on RAM bandwidth, CPU core count, and specific GPU model.

---

## Community Attribution

- **u/Psyko38**: AMD RX 7900 XTX 24GB ROCm benchmarks (15.82 vs 37.74 tok/s with/without `--fit`)
- **u/Corosus**: RTX 3090 24GB Vulkan benchmarks (13 vs 33 tok/s with/without `--fit`)
- **u/KierkegaardsSisyphus**: MXFP4 `--fit-target 1500` config (claims 77 tok/s on 5080)
- **u/Chromix_**: Insight that `-b/-ub` batch flags consume VRAM and conflict with `--fit`
- **u/Qxz3**: 8GB VRAM use case prompting the full-offload guidance
- **u/danielhanchen** (Unsloth): Confirmed UD-Q4_K_XL bug and provided fixed quant data
