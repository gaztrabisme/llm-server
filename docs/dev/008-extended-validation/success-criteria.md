# Session 008: Extended Validation & Production Switch

**Date**: 2026-02-28
**Mode**: Build
**Status**: In Progress

## Done When

### Phase A: Extended Context KV Quality
- [x] PPL measured at 65k, 128k ctx with KV q8_0 (UD-Q4_K_XL model): 5.8902, 5.3521
- [x] PPL measured at 65k, 128k ctx with KV f16 (baseline): 5.8965, 5.3590
- [x] PPL at 262k ctx attempted: WikiText-2 too small (297k tokens < 524k needed). No OOM though!
- [SKIPPED] KLD at 65k, 128k — base files ~533 GB each, prohibitive. PPL delta sufficient
- [x] Speed (TG+PP) at 65k, 128k, 262k measured: TG ~42-45, PP ~1287-1573, all contexts work

### Phase B: Config Flags & PP Speed
- [x] `--no-kv-offload` benchmarked: **-63% TG (16.1 vs 42.7 tok/s)** — do not use
- [x] `-ub 1024 -b 2048` benchmarked: TG -3.5%, PP-1024 +22% — mixed, not recommended
- [x] PP speed at 634/1235/4662/20176/32569/62882 tokens measured: 1390→1784→1695 tok/s

### Phase C: Real-World Evals
- [x] lm-eval-harness installed and sanity-checked
- [PATCHED] lm-eval openai_completions.py: llama.cpp returns new-format logprobs (content[]), lm-eval expected old format (token_logprobs[])
- [BUG] llama.cpp doesn't support echo=true — ALL loglikelihood benchmarks produce identical random-chance scores across all quants
- [DROPPED] ARC-Challenge, HellaSwag, Winogrande: loglikelihood-only tasks, broken with llama.cpp
- [DROPPED] MMLU-Pro, HumanEval, MBPP: generation-based but impractical (~50s/request with thinking mode)
- [DROPPED] ARC-Challenge generative variants: arc_challenge_chat (stop "." fires immediately), arc_challenge_llama (`<think>` tags exhaust max_gen_toks)
- [REVISED] Final benchmarks: GPQA generative + GSM8K (generation-only, limit 200, 0-shot)
- [x] Q8_0 ceiling: GPQA=0.415, GSM8K=0.570
- [x] UD-Q4_K_XL: GPQA=0.450, GSM8K=0.670
- [x] Q4_K_M (bartowski): GPQA=0.430, GSM8K=0.735
- [x] AesSedai Q4_K_M: GPQA=0.430, GSM8K=0.670
- [x] 4×2 score matrix completed
- [CAVEAT] Results unreliable for quant comparison: all Q4 quants outscore Q8_0 (theoretically impossible). Root cause: max_gen_toks=256 too small for thinking model, flexible-extract sensitivity to thinking chain length. PPL/KLD remain better metrics for quant quality.

### Phase D: AesSedai Q4_K_M
- [x] Model downloaded (split GGUF, 20.6 GiB)
- [x] PPL + KLD measured: PPL=6.3949, KLD=0.0095, same-top-p=95.74%
- [x] Speed: TG ~44.3 tok/s, PP-512 1257 tok/s
- [x] Three-way comparison complete: AesSedai wins all quality metrics (KLD 3x better than bartowski)

### Phase E: Vision & Community
- [x] Harder vision tests: code OCR (perfect), math solving (correct), table reading (perfect)
- [x] Large image PP speed: 350px=230tok/2.5s, 1024px=1045tok/2.9s, 2048px=4117tok/7.0s
- [ ] Reddit post updated with S008 findings
