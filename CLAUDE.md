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

**Qwen3.5-35B-A3B** — recommended quant: **UD-Q4_K_XL** (~20 GB, Unsloth), reference quant: Q8_0 (36.9 GB)
- MoE: 256 experts per layer, top-8 routing + 1 shared expert (9 active), 40 MoE layers, ~3B active params per token
- 262,144 native context length (extensible to 1,010,000 via YaRN), production config uses `-c 65536` conservatively. Thinking mode enabled by default
- WikiText-2 PPL: Q8_0 = 6.5342, UD-Q4_K_XL = 6.5959 (+1.0%), Q4_K_M = 6.6688 (+2.1%)
- UD-Q4_K_XL (FIXED) is the best Q4-class quant: KLD 0.0145 vs Q4_K_M 0.0286, same-top-p 94.46% vs 92.46%. (S007 E4 — old UD quant had MXFP4 bug, now fixed)
- Note: production server currently runs Q4_K_M (~74 tok/s). UD-Q4_K_XL achieves ~48 tok/s with same config (similar speed, better quality)

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
- Images: `llm-server/llama-cpp:latest` (HEAD ecbcb7e, supports --mmproj vision + perplexity overflow fix), `llm-server/llama-cpp:latest-fit` (b8149, supports `--fit on`)
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
- **KV cache q8_0 is a confirmed free lunch across all context lengths** (4k-32k): PPL delta < 0.3%, KLD < 0.019, same-top-p > 95.1% at every tested context length. KLD actually decreases from 4k→16k, slight uptick at 32k but well within noise. KV cache savings: q8_0 47%, q4_0 72%, asym 59% vs f16. Only 10 KV cache layers due to hybrid SSM architecture. (S006 E1 + S007 E1 full tier)
- **UD-Q4_K_XL (FIXED) is the best Q4-class quant**: PPL 6.5959 (vs Q4_K_M 6.6688), KLD 0.0145 (vs 0.0286), same-top-p 94.46% (vs 92.46%). Better quality at same speed and size. Old UD quant had MXFP4 bug (S006 E2) — now fixed. (S007 E4)
- Do NOT use `-b/-ub` batch flags with `--fit on` — they hurt TG by ~35% and don't help PP at short prompts (512 tokens). At 1024+ tokens, asymmetric batch is only 8% faster PP, not worth the TG penalty. No-batch wins for all workloads. (S006 + S007 E2)
- `--fit on` is CUDA-specific: Vulkan users report 2.5x slower (Corosus, 5070 Ti), ROCm users report 2.4x slower (Psyko38, RX 9060 XT). AMD/Vulkan users should use manual `--n-cpu-moe` instead

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

- **KV cache q8_0 is a confirmed free lunch**: PPL delta < 0.3%, KLD < 0.019, holds across all context lengths 4k-32k (S006 E1, S007 E1)
- **UD-Q4_K_XL (FIXED) beats Q4_K_M**: KLD 0.0145 vs 0.0286, same-top-p 94.46% vs 92.46%, PPL 6.5959 vs 6.6688. Recommend as production quant. (S007 E4)
- **`--fit on` without -b/-ub beats all configs**: 74.7 vs 69.6 tok/s, and batch flags don't help PP at short prompts (S006 E4, S007 E2)
- **Q4_K_L not worth the speed penalty**: -36% better KLD but 44% slower due to larger tensors (S006 E3)
- **MXFP4_MOE max 52.4 tok/s** (not 77 tok/s claimed): fit-target 1500 reduces VRAM for experts, hurting TG. Better PP than Q4_K_M due to smaller model. (S006, S007 E3)
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
- KL divergence analysis (Session 006) — old UD-Q4_K_XL had MXFP4 bug (3.9x worse KLD). Fixed version is best Q4-class quant (S007 E4)
- Bartowski Q4_K_L evaluation (Session 006) — better KLD but 44% slower, not worth it on 16GB
- MXFP4_MOE evaluation (Session 006) — marginal quality gain, 34-42% slower, memory leak bug
- Qwen3.5-27B dense comparison (Session 006) — 10x slower than MoE, worse quality
- Speculative decoding (Session 006) — no compatible draft model (248K vs 151K vocab), ngram methods no benefit
- Q4_K_M quant validation (Session 005) — only +2.1% PPL loss, enables 74 tok/s
- Pin Dockerfile to known-good commit (Session 005) — `LLAMA_CPP_REF` build arg added
- Thread sweep (Session 004) — **20 threads optimal**, +27% over t16
- ik_llama.cpp fork (dropped — unstable, no speed advantage)
- ktransformers (dropped — requires ~320GB RAM, exceeds 128GB)
- Unsloth Dynamic quants (Session 003/005/006 — old UD quants had MXFP4 bug, caused bad results)
- **KV cache full context sweep** (Session 007 E1) — PPL+KLD across 4k/8k/16k/32k × 4 KV configs. Confirms free lunch holds at all context lengths. Found int32 overflow bug in llama-perplexity KLD mode for large-vocab models (ctx >= 8648 with n_vocab=248320) — patched in Dockerfile
- **PP vs TG batch tradeoff** (Session 007 E2) — batch flags hurt TG by 35%, don't help PP at short prompts. No-batch confirmed optimal
- **MXFP4 re-evaluation** (Session 007 E3) — max 52.4 tok/s (not 77 claimed), fit-target 1500 reduces VRAM for experts
- **Fixed UD-Q4_K_XL validation** (Session 007 E4) — KLD 0.0145 vs Q4_K_M 0.0286. Better quality, same speed. New best Q4-class quant
- **Vision mode evaluation** (Session 007 E5) — works with --mmproj, 49.5 tok/s TG (33% slower due to fit-target 2000), minimal VRAM overhead (~114 MB). lmms-eval quality benchmarks blocked by v0.6.1 task registration issue
- **Community notes** (Session 007 E6) — AMD/ROCm, Vulkan, 8GB/24GB VRAM, LM Studio guidance

New scripts: `bench-matrix.sh` (universal benchmark runner with PP+TG), `compare-matrix.py` (matrix comparison tool), `vision-eval.sh` (vision quality eval). Shared bench library: `lib-bench-common.sh` (env parsing, Docker lifecycle, VRAM capture).

### Ready to Test
- **Thinking mode** — on by default in Qwen3.5, verify it works well with downstream apps
- **Chat template** — community reports GGUF embedded template may be incomplete, test with explicit `--chat-template`
- **`--fuse-gate-up-exps`** — b8164 feature, ~12% PP speedup for MoE, requires re-quantizing the GGUF

### Future / Blocked
- Expert caching (locality-based, LFU) — not in mainline llama.cpp, HOBBIT paper shows 10x potential but no public code
- cuBLAS Grouped GEMM (CUDA 13.1 — up to 4x MoE speedup, needs llama.cpp support)
- FP4 quantization via Blackwell's native Tensor Core support
- Draft-model speculative decoding — blocked until small Qwen3.5 model (1-3B) is released

## Status

**Production-ready.** Server achieves ~74 tok/s at Q4_K_M with 2.1% PPL loss. API is OpenAI-compatible. Session 007 completed 6 community-requested experiments: confirmed KV q8_0 free lunch across all context lengths (4k-32k), validated fixed UD-Q4_K_XL as best Q4-class quant (KLD 0.0145 vs Q4_K_M 0.0286), debunked 77 tok/s MXFP4 claim (max 52.4), confirmed no-batch as optimal for all workloads, demonstrated vision mode (49.5 tok/s, works but 33% TG penalty), and documented guidance for AMD/ROCm/Vulkan/varying VRAM users. Vision quality benchmarks (lmms-eval) blocked by v0.6.1 packaging issue.

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

See `docs/dev/007-community-experiments/` for Session 007 (complete):
- `success-criteria.md` — 6 experiments with results: KV deep dive, PP/TG tradeoff, MXFP4 redemption, fixed UD-Q4_K_XL, vision, community notes
- `progress-checkpoint.md` — detailed results tables and findings
- `community-notes.md` — AMD/ROCm, Vulkan, 8GB/24GB VRAM, LM Studio guidance
- 14 config files in `configs/llama-cpp-s007-*.env`
- Scripts: `bench-matrix.sh`, `compare-matrix.py`, `vision-eval.sh`, `lib-bench-common.sh`

### Daniel's Component Ablation Study

Unsloth published 550+ GGUF variants with 120+ KLD evaluations at [unsloth/Qwen3.5-35B-A3B-Experiments-GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-Experiments-GGUF). Key findings:
- **Expert down projections** are the most sensitive component (5.1% PPL at iq2_xxs)
- **tok, out, shr** can be aggressively quantized with minimal impact
- **SSM pos3** and **attention pos4** are outlier-sensitive sub-components
- Fixed UD-Q4_K_XL (19.2 GiB): KLD 0.0137, 94.7% same-top-p — better than bartowski Q4_K_M (19.8 GiB): KLD 0.0182, 94.2%. **Independently confirmed in S007 E4**: our measurements show KLD 0.0145/94.46% vs 0.0286/92.46%
