# Session 011: Community Benchmark Suite + Gap Experiments

**Date**: 2026-03-01
**Mode**: Build
**Status**: Complete

## Motivation

Reddit comments show community members (bobaburger/5060Ti, maho_Yun/5060Ti, OsmanthusBloom/3060, Corosus/5070Ti) running benchmarks by hand and posting screenshot tables. `llm-bench` should be easy enough that they clone, install, and run — getting structured, comparable results.

Meanwhile, 4 comment threads ask about PP speed visibility, 1 asks about IQ4_KS, and several mention `--fit-target`/`--fit-ctx`/`--fuse-gate-up-exps` flags we've never tested.

This session combines tooling improvements with experiments that validate them.

## Done When

### Phase 1: Missing Flags + Model Auto-Detect (Tooling)

- [x] `--fit-target N` added to FLAG_MAP, CLI override, and config schema
- [x] `--fit-ctx N` added to FLAG_MAP, CLI override, and config schema
- [x] `--fuse-gate-up-exps` added to FLAG_MAP as boolean presence flag
- [x] `--n-cpu-moe N` added to FLAG_MAP (for non-CUDA users who can't use --fit)
- [x] `llm-bench setup` auto-detects .gguf files in `./models/` and lists them (already worked)
- [x] `llm-bench speed --model <name>` accepts just a filename and resolves from models/ dir

### Phase 2: Native llama-server Support (Tooling)

The #1 barrier to community adoption: Docker requirement. Most community members compile llama.cpp natively.

- [x] `llm-bench speed --native --server-bin /path/to/llama-server` mode added
- [x] Native mode: starts llama-server as subprocess (not Docker container)
- [x] Native mode: same health check, benchmark, cleanup lifecycle
- [x] Native mode: `--dry-run` prints the shell command instead of Docker command
- [x] Docker remains the default; `--native` is opt-in

### Phase 3: PP Speed Visibility (Tooling)

Community asks (mr_Owner, OsmanthusBloom, mdziekon, DHasselhoff77): "show PP speed beside TG"

- [x] `llm-bench compare` default table shows PP-512 column alongside TG columns
- [x] `llm-bench compare --pp-detail` shows all PP lengths (512/1024/4096/16384)
- [x] Hardware fingerprint (GPU model + RAM) saved in result JSON for cross-machine comparison
- [x] `llm-bench compare` shows hardware column when results come from different machines

### Phase 4: Experiments (validate tooling + answer community)

Run using the new flags added in Phase 1. Each experiment produces a benchmark JSON.

#### E1: `--fit-ctx` vs `-c` (Danmoreng) — DONE
- [x] Speed benchmark: `--fit-ctx 65536` vs current `-c 65536 --fit on`
- [x] Compare TG, PP, VRAM allocation
- [x] Document: does `--fit-ctx` change allocation behavior?

**Results**: Both configs produce identical performance (~45 tok/s TG), same VRAM (14534-14536 MB), and same allocation (41 layers offloaded, 18 overflow). `--fit-ctx` is equivalent to `-c` when used with `--fit on` — no behavioral difference.

#### E2: `--fit-target 1536` for vision (Danmoreng + maho_Yun) — DONE
- [x] Speed benchmark: vision config with `--fit-target 1536` vs `--fit-target 2000` (our S007 E5 baseline)
- [x] Compare TG, VRAM — maho_Yun's data suggests fit-target is critical

**Results**: `--fit-target 1536` = 43.2 tok/s TG, 19 overflow experts. `--fit-target 2000` = 41.4 tok/s TG, 20 overflow experts. Text-only baseline (no mmproj) = 45.1 tok/s, 18 overflow. Lower fit-target frees more VRAM for expert layers, yielding 4% faster TG. **Recommend `--fit-target 1536` for vision mode** (only 4% slower than text-only, vs 8% slower with fit-target 2000).

#### E3: `--fuse-gate-up-exps` PP speedup (CLAUDE.md ready-to-test) — SKIPPED
- [ ] ~~Speed benchmark with `--fuse-gate-up-exps` enabled~~
- [ ] ~~Compare PP at 512/1024/4096/16384 vs baseline (claim: ~12% PP speedup for MoE)~~
- [ ] ~~Note: may require re-quantized GGUF or specific llama.cpp build~~

**Skipped**: Flag not available in the b8149 build. Requires a newer llama.cpp build (b8164+). Revisit when Docker image is updated.

#### E4: IQ4_KS quality (Gringe8) — SKIPPED
- [ ] ~~Download bartowski IQ4_KS quant (if available for Qwen3.5-35B-A3B)~~
- [ ] ~~PPL + KLD vs Q4_K_M baseline~~
- [ ] ~~Speed benchmark (should fit ~fully in 16GB VRAM if smaller than Q4_K_M)~~
- [x] If IQ4_KS unavailable for this model, document why and skip

**Skipped**: IQ4_KS quant format does not exist for Qwen3.5-35B-A3B. No bartowski or other quantizer has produced this format for this model.

### Phase 5: Cleanup + Docs — DONE

- [x] Update CLAUDE.md with S011 findings
- [x] Update session-index.md
- [x] Q4_K_M identity investigation: confirmed always Unsloth's (not bartowski's)
- [x] Speed regression investigation: documented as open finding

### Major Finding: Speed Regression (~75 → ~50 tok/s)

Investigation revealed that TG speed dropped from ~75 tok/s (S006, Feb 27 PM) to ~50 tok/s (S007, Feb 28 AM onward) with identical hardware, Docker image (latest-fit, b8149), model, and config.

**What was ruled out:**
- GPU throttling: 2685 MHz graphics, P1 state, ~97W (well under 360W limit), no thermal slowdown
- CPU performance: `performance` governor, 5.6 GHz, no contention
- Docker image: same `latest-fit` image (built Feb 25, never rebuilt)
- Measurement methodology: wall-clock `completion_tokens / wall_ms` used in both bench.sh and bench-matrix.sh
- Prompts: identical text in both scripts
- API endpoint: both use `/v1/chat/completions`
- PCIe link: Gen 5 confirmed under load
- Model file: same Unsloth Q4_K_M (21,205,326,432 bytes)

**Current hypothesis**: Unknown system-level change between Feb 27 PM and Feb 28 AM. apt history rotated, so can't verify package updates. Could be CUDA runtime behavior, kernel scheduling, or NUMA configuration change. The `--fit` allocation looks the same (41 layers, 18 overflowing).

**Impact**: All S007+ benchmark numbers (~45-50 tok/s) are consistent with each other but ~40% lower than S006 (~74 tok/s). The S006 numbers may have been anomalously high, or something regressed overnight. All relative comparisons (quant vs quant, config vs config) remain valid.

### Q4_K_M Identity Correction

Investigation confirmed: the Q4_K_M file on disk was ALWAYS from Unsloth (file size 21,205,326,432 bytes matches Unsloth repo exactly, GGUF metadata: `quantized_by=Unsloth`). bartowski's Q4_K_M is 21,227,116,800 bytes (different). Sessions 005-008 incorrectly attributed it to bartowski.

### Comprehensive Benchmark Results (Full Runs)

| Config | TG Short | TG Med | TG Long | TG Multi | PP-512 | VRAM (MB) | Overflow |
|--------|----------|--------|---------|----------|--------|-----------|----------|
| Baseline (-c 65536 --fit on) | 45.8 | 44.8 | 45.5 | 43.8 | 1275 | 14507 | 18/41 |
| E1 (--fit-ctx 65536) | 44.0 | 44.1 | 44.9 | 43.0 | 1306 | 14605 | 18/41 |
| E2a (vision ft1536) | 44.8 | 44.9 | 44.0 | 42.5 | 1306 | 15171 | 19/41 |
| E2b (vision ft2000) | 44.0 | 44.4 | 42.9 | 42.0 | 1269 | 14592 | 20/41 |

## Scoping Notes

**In scope:**
- Native llama-server support (subprocess, not Docker)
- Missing flag support (fit-target, fit-ctx, fuse-gate-up-exps, n-cpu-moe)
- PP visibility improvements in compare output
- Hardware fingerprint in results
- 4 experiments using new flags

**Out of scope (future):**
- `llm-bench compare --merge` cross-machine aggregation (Phase 3 lays groundwork with hardware fingerprint)
- Custom workload YAML for PP/TG prompts
- Multi-GPU support
- AMD/ROCm/Vulkan backend auto-detection
- Q2/Q5 tier testing (per user: we don't go below Q4, Q5 doesn't fit well in 16GB)
- Web UI or result dashboard

## Effort Estimate

| Phase | Items | Complexity |
|-------|-------|-----------|
| 1 | 6 | Low — FLAG_MAP additions, minor CLI changes |
| 2 | 5 | Medium — new execution path, but mirrors Docker lifecycle |
| 3 | 4 | Low-Medium — compare output formatting |
| 4 | 4 experiments | Medium — downloads + benchmark runs |
| 5 | 3 | Low — documentation |

## Dependencies

- E3 (`--fuse-gate-up-exps`): Check if current llama.cpp build (b8149 or HEAD) supports this flag
- E4 (IQ4_KS): Check if bartowski has this quant for Qwen3.5-35B-A3B
- E2 (vision): Needs mmproj file (already downloaded from S007 E5)
