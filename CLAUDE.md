# llm-server

Local LLM inference server using llama.cpp, optimized for MoE model offloading on a single-GPU consumer workstation.

## Goal

Run large Mixture-of-Experts models locally and serve them via API for other projects: classification, synthetic data generation, chatbot, agentic workflows.

## Hardware

- **GPU**: NVIDIA GeForce RTX 5080 16GB GDDR7 (Blackwell, sm_120, 960 GB/s, PCIe 5.0 x16)
- **CPU**: 32 cores
- **RAM**: 128 GB DDR5
- **CUDA**: 13.1, driver 590.48.01

## Primary Model

**Qwen3.5-35B-A3B** — production quant: **Q4_K_M** (~20 GB), reference quant: Q8_0 (36.9 GB)
- MoE: 256 experts per layer, top-8 routing + 1 shared expert (9 active), 40 MoE layers, ~3B active params per token
- 65536 context length, thinking mode enabled by default
- WikiText-2 PPL: Q8_0 = 6.5342, Q4_K_M = 6.6688 (+2.1% — negligible quality loss)
- UD-Q4_K_XL NOT recommended: PPL 7.1702 (+9.7%, worse than standard Q4_K_M)

Previous model: Qwen3-Next-80B-A3B-Instruct (Q8_0, 84.8 GB, ~22 tok/s) — replaced in Session 005

## Offloading Strategy

Auto VRAM management via `--fit on`: llama.cpp automatically determines the optimal GPU/CPU split based on available VRAM. Key insight (Session 006): removing `-b/-ub` batch flags frees VRAM for more expert layers on GPU, yielding better throughput than manual `--n-cpu-moe` tuning. VRAM usage: ~14.6 GB.

**Production performance**: ~74 tok/s token generation (Q4_K_M + `--fit on`, no batch flags).

## Reference Launch Command

Winning config (fit-nobatch: Q4_K_M + auto offload, ~74 tok/s, Session 006):

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

Alternative (quality-first): Q8_0 + `--fit on` at ~40 tok/s (config C4r, requires b8149+ build with `--fit` support).

## Docker

- Base image: `nvidia/cuda:12.8.1-devel-ubuntu24.04` (build) / `nvidia/cuda:12.8.1-runtime-ubuntu24.04` (runtime)
- Build flags: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA_ALL_QUANTS=ON`
- Build quirk: must symlink `libcuda.so.1` from CUDA stubs for Docker builds without GPU
- **`LLAMA_CPP_REF` build arg**: pin llama.cpp to a specific commit/tag (e.g., `docker build --build-arg LLAMA_CPP_REF=b8149 .`). Unpinned builds caused 30% regression in Session 003/004.
- Images: `llm-server/llama-cpp:latest` (HEAD), `llm-server/llama-cpp:latest-fit` (b8149, supports `--fit on`)
- Run with: `docker run --gpus all --ipc host`
- Docker Compose: `docker compose --profile llama-cpp up`
- NVIDIA Container Toolkit v1.18.2 injects `libcuda.so.1` at runtime via `--gpus all`

## API

llama-server provides:
- OpenAI-compatible: `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`
- Anthropic Messages API: `/v1/messages` (merged Dec 2025)

## Key Constraints

- 16 GB VRAM: partial MoE offload required — `--fit on` auto-splits ~16/40 layers to GPU with Q4_K_M, ~14.6 GB VRAM
- PCIe 5.0 bandwidth (~64 GB/s) is the bottleneck, not GPU compute
- `--no-mmap` is important: loads entire model into RAM upfront for consistent offload performance
- Thread count (`-t`) is tuned: **20 is optimal** (Session 004 sweep). U-shaped curve — t16 is worst, t8/t20/t24 are best tier
- KV cache q8_0 is a free lunch: +12-38% throughput AND less VRAM with no quality impact (confirmed by PPL matrix, Session 006)
- Q4_K_M preferred over UD-Q4_K_XL for MoE models (3.9x worse KLD, confirmed by KL divergence, Session 006)
- Do NOT use `-b/-ub` batch flags with `--fit on` — they consume VRAM that `--fit` needs for expert layers (Session 006)

## Benchmark Results (Sessions 005-006 — Qwen3.5-35B-A3B)

Tested 2026-02-25 (S005), 2026-02-27 (S006). All configs: 20 threads, 65k ctx, `--no-mmap`, KV cache q8_0.

### Summary Table

| Config | Quant | Strategy | Short (tok/s) | Medium (tok/s) | Long (tok/s) | Multi-turn (tok/s) | VRAM (MB) |
|--------|-------|----------|--------------|----------------|-------------|-------------------|-----------|
| C1 | Q8_0 | full offload (`-ot "exps=CPU"`) | 35.7 | 32.8 | 33.2 | 35.2 | 8064 |
| C4r | Q8_0 | `--fit on` (auto, b8149) | 40.5 | 40.3 | 39.6 | 40.3 | 14660 |
| C5 | Q4_K_M | full offload | 51.0 | 49.8 | 49.4 | 50.5 | 7217 |
| C7 | Q4_K_M | `--n-cpu-moe 24` (16/40 on GPU) | 69.6 | 67.0 | 65.7 | 69.2 | 14874 |
| **fit-nobatch** | **Q4_K_M** | **`--fit on`, no -b/-ub** | **74.7** | **72.9** | **73.7** | **76.1** | **14559** |
| MXFP4_MOE | MXFP4 | `--fit on`, no -b/-ub | 49.5 | 47.8 | 46.9 | 43.0 | 14531 |
| Q4_K_L | Q4_K_L | `--fit on`, no -b/-ub | 41.4 | 41.4 | 40.8 | 41.8 | 14489 |
| 27B dense | Q4_K_M | `--fit on` (dense, 8k ctx) | 7.4 | 7.4 | 7.2 | 7.1 | 14075 |

### Winner: Q4_K_M + fit-nobatch (Session 006)

**~74 tok/s** with 14.6 GB VRAM — +7% over Session 005's C7 config, and simpler (no manual `--n-cpu-moe` tuning). Key insight: removing `-b/-ub` batch flags frees VRAM for `--fit` to allocate more expert layers on GPU.

### Key Findings

- **KV cache q8_0 is a free lunch**: < 0.4% PPL difference, +12-38% throughput (confirmed by full PPL matrix, S006 E1)
- **UD-Q4_K_XL is NOT recommended**: 3.9x worse KLD than Q4_K_M, only 86.2% same-top-1 vs 92.4% (S006 E2)
- **`--fit on` without -b/-ub beats manual tuning**: 74.7 vs 69.6 tok/s, simpler config (S006 E4)
- **Q4_K_L not worth the speed penalty**: -36% better KLD but 44% slower due to larger tensors (S006 E3)
- **MXFP4_MOE not recommended**: marginal quality gain, 34-42% slower, memory leak in dequant path (S006 E7)
- **Qwen3.5-27B dense is 10x slower**: all 27B params active per token, worse PPL too (S006 E6)
- **ngram self-speculation**: no benefit for conversational workloads, ngram-mod unstable (S006 E5)
- **Partial offload is the key lever**: keeping expert layers on GPU is 36% faster than full CPU offload
- **Q4_K_M halves model size** (20 GB vs 37 GB) enabling more layers on GPU within 16 GB VRAM

### Historical: Session 002 (Qwen3-Next-80B-A3B)

Winner was A2 (llama.cpp + KV q8_0) at ~22 tok/s, 6247 MB VRAM. Superseded by Session 005 model migration.

Run with: `bash scripts/bench.sh <engine> <config> [thread_count]`
Compare with: `python3 scripts/compare-results.py benchmarks/`

## Optimization Avenues

### Done
- Model migration to Qwen3.5-35B-A3B (Session 005) — 3.2x speedup over Qwen3-Next-80B
- **fit-nobatch config** (Session 006) — `--fit on` without `-b/-ub` batch flags, **74.7 tok/s** (+7% over manual offload)
- Partial MoE offload `--n-cpu-moe 24` (Session 005) — superseded by fit-nobatch
- KV cache q8_0 quality validation (Session 006) — full PPL matrix confirms < 0.4% impact
- KL divergence analysis (Session 006) — UD-Q4_K_XL 3.9x worse than Q4_K_M, 86.2% vs 92.4% same-top-1
- Bartowski Q4_K_L evaluation (Session 006) — better KLD but 44% slower, not worth it on 16GB
- MXFP4_MOE evaluation (Session 006) — marginal quality gain, 34-42% slower, memory leak bug
- Qwen3.5-27B dense comparison (Session 006) — 10x slower than MoE, worse quality
- Speculative decoding (Session 006) — no compatible draft model (248K vs 151K vocab), ngram methods no benefit
- Q4_K_M quant validation (Session 005) — only +2.1% PPL loss, enables 74 tok/s
- Pin Dockerfile to known-good commit (Session 005) — `LLAMA_CPP_REF` build arg added
- Thread sweep (Session 004) — **20 threads optimal**, +27% over t16
- ik_llama.cpp fork (dropped — unstable, no speed advantage)
- ktransformers (dropped — requires ~320GB RAM, exceeds 128GB)
- Unsloth Dynamic quants (Session 003/005/006 — slower or worse quality than standard quants for MoE)

### Ready to Test (Medium Impact)
- **Vision** — mmproj-BF16.gguf downloaded, needs smoke test with `--mmproj` flag
- **Thinking mode** — on by default in Qwen3.5, verify it works well with downstream apps
- **Chat template** — community reports GGUF embedded template may be incomplete, test with explicit `--chat-template`
- **Pin Docker image** to winning llama.cpp commit — prevent regressions on rebuild

### Future / Blocked
- Expert caching (locality-based, LFU) — not in mainline llama.cpp, HOBBIT paper shows 10x potential but no public code
- cuBLAS Grouped GEMM (CUDA 13.1 — up to 4x MoE speedup, needs llama.cpp support)
- FP4 quantization via Blackwell's native Tensor Core support
- KV cache q4_0 for longer contexts (frees VRAM for more experts on GPU)
- Draft-model speculative decoding — blocked until small Qwen3.5 model (1-3B) is released

## Status

**Production-ready.** Server achieves ~74 tok/s at Q4_K_M with 2.1% PPL loss. API is OpenAI-compatible. All quantization alternatives tested (Q4_K_L, MXFP4_MOE, UD-Q4_K_XL, 27B dense) — Q4_K_M remains best for single-GPU consumer hardware. Next step is downstream applications (classification, synthetic data, chatbot, agentic workflows).

## Research Documentation

See `docs/dev/001-research-and-setup/research/` for detailed findings:
- `qwen3-next-80b-research.md` — model architecture, memory analysis, benchmarks
- `rtx-5080-specs-and-cuda.md` — GPU specs, CUDA 13.1, Docker setup
- `llama-cpp-research-feb2026.md` — llama.cpp features and Blackwell support
- `moe-offloading-research.md` — offloading strategies and alternative tools
- `local-llm-serving-guide.md` — serving patterns and application integration

See `docs/dev/005-qwen35-migration/` for Qwen3.5-35B-A3B migration:
- `success-criteria.md` — benchmark matrix and quality results
- `handoff.md` — delivered files, decisions, and next steps

See `docs/dev/006-community-followup/` for community follow-up experiments:
- `success-criteria.md` — all 7 experiments with full results
- `reddit-followup-post.md` — Reddit follow-up post draft
- `reddit-replies.md` — individual reply drafts
