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
- [REVISED] MMLU-Pro → dropped: uses generate_until (~50s/request with thinking), impractical (~97h/quant)
- [REVISED] HumanEval, MBPP → dropped: code execution overhead, generation-based, same speed issue
- [REVISED] Final benchmarks: ARC-Challenge, HellaSwag, GPQA, Winogrande (loglikelihood) + GSM8K (generate, limited)
- [PATCHED] lm-eval openai_completions.py: llama.cpp returns new-format logprobs (content[]), lm-eval expected old format (token_logprobs[])
- [ ] Q8_0 ceiling: 5 benchmarks (ARC-Challenge, HellaSwag, GPQA, Winogrande, GSM8K)
- [ ] UD-Q4_K_XL: same 5 benchmarks
- [ ] Q4_K_M (bartowski): same 5 benchmarks
- [ ] AesSedai Q4_K_M: same 5 benchmarks
- [ ] 4×5 score matrix completed

### Phase D: AesSedai Q4_K_M
- [x] Model downloaded (split GGUF, 20.6 GiB)
- [x] PPL + KLD measured: PPL=6.3949, KLD=0.0095, same-top-p=95.74%
- [x] Speed: TG ~44.3 tok/s, PP-512 1257 tok/s
- [x] Three-way comparison complete: AesSedai wins all quality metrics (KLD 3x better than bartowski)

### Phase E: Vision & Community
- [ ] Harder vision tests (code OCR, math, chart, document, photo)
- [ ] Large image PP speed (1024x1024, 2048x2048)
- [ ] Reddit post updated with S008 findings
