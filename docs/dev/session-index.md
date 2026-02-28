# Dev Sessions

| # | Date | Mode | Description | Status |
|---|------|------|-------------|--------|
| 001 | 2026-02-17 | Design | Research llama.cpp + MoE offloading for RTX 5080 | Complete |
| 002 | 2026-02-22 | Build | Infrastructure setup + A/B benchmark (llama.cpp vs ik_llama.cpp) | Complete — Winner: llama.cpp + KV q8_0 (~22 tok/s) |
| 003 | 2026-02-24 | Build | Optimization sweep: new flags + Unsloth Dynamic Q8_K_XL | Complete — No improvement over A2; rebuild regression detected |
| 004 | 2026-02-25 | Analyze | Speedup investigation: commit regression, speculative decoding, expert caching, thread sweep | Complete — 20 threads optimal (+27%), speculative decoding next |
| 005 | 2026-02-25 | Build | Model migration: Qwen3-Next-80B-A3B → Qwen3.5-35B-A3B + partial MoE offload benchmark | Complete — Winner: Q4_K_M + `--n-cpu-moe 24` (~70 tok/s, 3.2x speedup) |
| 006 | 2026-02-26 | Analyze | Community follow-up: 7 experiments (KV quality, KLD, Q4_K_L, fit-nobatch, spec decoding, 27B dense, MXFP4) | Complete — New winner: fit-nobatch ~74 tok/s, Q4_K_M confirmed best quant |
| 007 | 2026-02-28 | Build | Community experiments: KV deep dive (free lunch confirmed 4k-32k), PP/TG tradeoff (no-batch wins), MXFP4 (max 52 tok/s not 77), UD-Q4_K_XL (best Q4 quant), vision mode (works, -33% TG), community notes | Complete — UD-Q4_K_XL new best quant, KV q8_0 confirmed across contexts |
| 008 | 2026-02-28 | Build | Extended validation: context 65k-262k (KV free lunch confirmed), --no-kv-offload (-63% TG), AesSedai Q4_K_M (KLD 0.0095, best quality), lm-eval (echo=true bug, thinking chain truncation, PPL/KLD > task evals) | Complete |
| 009 | 2026-02-28 | Build | `llm-bench` reusable benchmark framework: Python CLI replacing bash scripts, 6 subcommands (setup/speed/quality/eval/compare/report), backward-compatible with env files and result JSON | Complete |
