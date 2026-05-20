# Session 015: Qwen3.6 — 27B Dense vs 35B MoE + MTP + TurboQuant

## Goal

Benchmark Qwen3.6 configurations on RTX 5080 16GB for coding agent workloads:
- 35B-A3B MoE baseline vs MTP-enhanced
- 27B dense at aggressive quants (IQ3_XXS, full GPU)
- TurboQuant KV cache impact at 128k context
- MTP acceptance rates and real-world speedup

## Core Question

Can Qwen3.6-27B dense at IQ3 (full GPU, no PCIe bottleneck) match or beat Qwen3.6-35B-A3B MoE for coding on 16GB VRAM? With MTP and TurboQuant, what's the optimal config for a coding agent at 128k context?

## Experiment Matrix

| # | Label | Model | MTP | KV cache | Notes |
|---|-------|-------|-----|----------|-------|
| E1 | 35b-baseline | 35B-A3B UD-Q4_K_XL | No | q8_0 | Production baseline |
| E2 | 35b-mtp | 35B-A3B MTP UD-Q4_K_XL | n=2 | q8_0 | MTP speculative decode |
| E3 | 35b-mtp-turbo | 35B-A3B MTP UD-Q4_K_XL | n=2 | turbo4 | MTP + TurboQuant KV |
| E4 | 27b-iq3 | 27B UD-IQ3_XXS | No | turbo4 | Dense, full GPU |
| E5 | 27b-mtp | 27B MTP UD-Q2_K_XL | n=2 | turbo4 | Dense + MTP |
| E6 | 27b-mtp-n3 | 27B MTP UD-Q2_K_XL | n=3 | turbo4 | Dense + MTP aggressive |

## Success Criteria

### Phase 0: Setup
- [ ] Custom llama.cpp image with MTP + TurboQuant built and tagged
- [ ] All 5 models downloaded (3 new + 2 existing on disk)
- [ ] llm-bench updated with new flags (mlock, no_warmup, parallel, spec_type, etc.)
- [ ] mtp-bench.py and CodeNeedle installed

### Phase 1: Configs
- [x] 6 env files created in configs/ (adapted at runtime for build differences)

### Phase 2: Speed Benchmarks
- [x] TG measured for 11 configs (E1-E7 + variants) via mtp-bench.py (9 prompts each)
- [x] MTP acceptance rate measured across 3 builds (am17an 99%, Indras 80%, AtomicBot Gemma4-only)
- [x] MTP on MoE confirmed net-negative (expert-union overhead)
- [x] Q2_K_XL identified as MTP sweet spot on 16 GB (fits fully on GPU → 70 tok/s)
- [x] Q3_K_XL too large for MTP (partial offload → 25 tok/s)
- [ ] ~~PP measured for all configs~~ (skipped — mtp-bench.py measures TG only; PP was not the focus)
- [ ] ~~ubatch sweep on E2~~ (skipped — MTP on MoE is net-negative, not worth optimizing)
- [ ] ~~Context-fill sweep (1k/32k/65k/128k)~~ (adapted — tested at operating context per config)

### Phase 3: Quality
- [x] CodeNeedle HTTP suite (11 functions) on E1, E4, E5
- [x] 27B IQ3_XXS: 99.5% accuracy, 0 hallucinations (best)
- [x] 35B MoE UD-Q4_K_XL: 93.6% accuracy, 12 hallucinations
- [x] 27B MTP Q2_K_XL: 90.5% accuracy, 20 hallucinations
- [x] IQ3 + MTP graft attempted — blocked (bare MTP head lacks model hyperparameters)
- [ ] ~~CodeNeedle jQuery suite~~ (skipped — HTTP suite sufficient to differentiate quant quality)
- [ ] ~~PPL/KLD for 27B IQ3 vs Q8_0~~ (skipped — CodeNeedle more relevant for coding use case)

### Phase 4: Deliverables
- [x] Master comparison table (in progress-checkpoint.md)
- [ ] Reddit post draft
- [ ] Production config decision
- [ ] CLAUDE.md updated with S015 findings
