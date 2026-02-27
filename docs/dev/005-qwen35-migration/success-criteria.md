# Session 005: Qwen3.5-35B-A3B Migration

## Mode: Build

Migrate from Qwen3-Next-80B-A3B (84.8 GB Q8_0, ~22 tok/s) to Qwen3.5-35B-A3B (36.9 GB Q8_0, expected 30-50+ tok/s). Head-to-head benchmark with partial MoE offloading.

## Done When

- [x] Qwen3.5-35B-A3B Q8_0 GGUF downloaded and verified
- [x] Vision mmproj file downloaded and verified (not 1KB/corrupt)
- [x] Docker image supports `qwen35moe` architecture (rebuild if needed, pin commit)
- [x] llama-server loads model and responds to `/health`
- [x] Benchmark matrix completed (at minimum C1 + C2) — ran C1-C8 (10 configs total)
- [x] Winner config documented with tok/s, VRAM, and rationale
- [x] CLAUDE.md updated with new production config

## Quant Quality Results (WikiText-2 PPL)

| Quant | Size | PPL | vs Q8_0 | Verdict |
|-------|------|-----|---------|---------|
| **Q8_0** | 36.9 GB | 6.5342 | baseline | Reference |
| **Q4_K_M** | ~20 GB | 6.6688 | +2.1% | RECOMMENDED — negligible quality loss |
| **UD-Q4_K_XL** | ~20 GB | 7.1702 | +9.7% | NOT RECOMMENDED — worse than Q4_K_M |

Method: `llama-perplexity` on WikiText-2 with KL divergence (Q8_0 as reference).
Script: `scripts/quant-quality.sh`

## Speed Benchmark Results (tok/s, RTX 5080 16GB)

All configs: 20 threads, 65k ctx, `--no-mmap`, KV cache q8_0 (unless noted).

| Config | Quant | Strategy | Short | Medium | Long | Multi-turn | VRAM (MB) | Status |
|--------|-------|----------|-------|--------|------|------------|-----------|--------|
| **C1** | Q8_0 | full offload (`-ot "exps=CPU"`) + KV q8_0 | 35.7 | 32.8 | 33.2 | 35.2 | 8064 | Done |
| **C2** | Q8_0 | `--n-cpu-moe 16` (24 on GPU) | — | — | — | — | — | OOM (21.6 GB needed) |
| **C2a** | Q8_0 | `--n-cpu-moe 32` (8 on GPU) | 34.4 | 36.5 | 38.8 | 34.7 | 14647 | Done |
| **C2b** | Q8_0 | `--n-cpu-moe 36` (4 on GPU) | 37.8 | 36.3 | 35.9 | 37.4 | 11387 | Done |
| **C3** | Q8_0 | full offload, no KV quant | 28.9 | 23.9 | 29.6 | 31.2 | 8681 | Done |
| **C4** | Q8_0 | `--fit on` (old build) | — | — | — | — | — | Failed (flag not supported) |
| **C4r** | Q8_0 | `--fit on` (b8149) | 40.5 | 40.3 | 39.6 | 40.3 | 14660 | Done |
| **C5** | Q4_K_M | full offload | 51.0 | 49.8 | 49.4 | 50.5 | 7217 | Done |
| **C6** | Q4_K_M | `--n-cpu-moe 16` (24 on GPU) | — | — | — | — | — | OOM (compute buffer) |
| **C7** | **Q4_K_M** | **`--n-cpu-moe 24` (16 on GPU)** | **69.6** | **67.0** | **65.7** | **69.2** | **14874** | **WINNER** |
| **C8** | Q4_K_M | `--fit on` | 67.4 | 62.3 | 64.1 | 62.1 | 14551 | Done |

## Speed Test Matrix (Original Plan)

| Config | Quant | Offload | KV Cache | Key Question |
|--------|-------|---------|----------|--------------|
| **C1** | Q8_0 | Full CPU (`-ot "exps=CPU"`) | q8_0 | Baseline: how fast with full offload? |
| **C2** | Q8_0 | Partial (`--n-cpu-moe N`, sweep) | q8_0 | How many MoE layers fit on GPU? |
| **C3** | Q8_0 | Full CPU | f16 (no quant) | Does KV quant hurt quality/speed? |
| **C4** | Q8_0 | `--fit on` | auto | Does auto-fit match manual tuning? |

Extended to C5-C8 after discovering Q4_K_M enables more layers on GPU.

## Phases

### Phase 1: Preparation (no GPU needed) — COMPLETE
1. Download Q8_0 GGUF (~36.9 GB)
2. Download mmproj file
3. Download/extract chat template from HF model
4. Create config files
5. Update bench.sh for new flags

### Phase 2: Docker Image (CPU build, GPU test) — COMPLETE
1. Check if current image supports `qwen35moe`
2. If not: pin llama.cpp to a commit with `qwen35moe` support, rebuild
3. Verify model loads

### Phase 3: Benchmark (GPU needed) — COMPLETE
1. Run C1-C8 (sequential, one at a time)
2. For C2: swept `--n-cpu-moe` values — 16 OOM, 24 optimal for Q4_K_M, 32/36 for Q8_0
3. Compare results
4. Test chat template fix (GGUF vs explicit)
5. Quick vision smoke test with mmproj

### Phase 4: Document — COMPLETE
1. Handoff doc with results
2. Update CLAUDE.md with winning config
3. Update session-index.md

## Key Risks (Resolved)

1. `qwen35moe` architecture may not be in our current llama.cpp build — **Resolved: rebuilt with pinned commit**
2. KV cache q8_0 may degrade at long context — **Not observed: consistent gains across all workloads**
3. GGUF chat template may be incomplete — **Resolved: template works correctly**
4. `--n-cpu-moe` VRAM fit depends on expert size x layers on GPU + KV cache — **Resolved: 16/40 layers fits at Q4_K_M**
5. Thinking mode is ON by default — **Noted: sampling params differ from old setup**

## Reference

- Community intel: `docs/dev/qwen35-35b-a3b-community-intel.md`
- Most relevant datapoint: u/JoNike, RTX 5080 16GB, `--n-cpu-moe 16`, 51 tok/s (UD-Q4_K_XL)
- Our result: **69.6 tok/s** with Q4_K_M + `--n-cpu-moe 24` — 36% faster than community reference
- Official HF page: https://huggingface.co/Qwen/Qwen3.5-35B-A3B
