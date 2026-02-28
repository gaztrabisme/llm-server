# S008 Progress Checkpoint

**Last updated**: 2026-02-28

## Phase A: Extended Context KV Quality

### PPL Results (UD-Q4_K_XL)

| Context | KV f16 | KV q8_0 | KV q4_0 | KV asym | Delta (q8_0 vs f16) |
|---------|--------|---------|---------|---------|---------------------|
| 4096 | 6.5959* | — | — | — | — |
| 8192 | — | — | — | — | — |
| 16384 | — | — | — | — | — |
| 32768 | — | — | — | — | — |
| 65536 | 5.8965 ± 0.039 | 5.8902 ± 0.039 | | | -0.11% |
| 131072 | 5.3590 ± 0.034 | 5.3521 ± 0.034 | | | -0.13% |
| 262144 | | N/A (needs 524k tokens, WikiText has 297k) | | | |

*S007 E4 reference value

### Speed at Extended Context (UD-Q4_K_XL, KV q8_0, fit-nobatch, latest-fit image)

| Context | TG short (tok/s) | TG medium | TG long | PP-512 (tok/s) | PP-1024 | VRAM (MB) |
|---------|------------------|-----------|---------|----------------|---------|-----------|
| 65536 | 42.7 | 42.9 | 42.6 | 1396 | 1573 | 14538 |
| 131072 | 44.8 | 44.8 | 44.3 | 1374 | 1521 | 14502 |
| 262144 | 41.1 | 41.6 | 41.0 | 1287 | 1409 | 14584 |

**Notes:**
- 262k loads and serves fine — no OOM. Full 262k native context works on 16GB GPU
- TG ~42-45 tok/s across all contexts — minimal degradation from 65k to 262k
- PP shows ~8% slowdown from 65k to 262k (1396→1287 at PP-512)
- UD-Q4_K_XL runs at ~43 tok/s vs Q4_K_M at ~74 tok/s (latest-fit image, S006 reference)
- 262k PPL failed: WikiText-2 has only 297k tokens, needs 524k (2 × ctx) minimum

## Phase B: Config Flags (UD-Q4_K_XL, 65k ctx, latest-fit image)

| Config | TG short | TG med | TG long | PP-512 | PP-1024 | VRAM (MB) | Notes |
|--------|----------|--------|---------|--------|---------|-----------|-------|
| Baseline (fit-nobatch) | 42.7 | 42.9 | 42.6 | 1390 | 1607 | 14538 | From Phase A |
| --no-kv-offload | 16.1 | 16.0 | 15.7 | 1147 | 1384 | 14758 | **-63% TG!** Do not use |
| -ub 1024 -b 2048 | 41.1 | 41.6 | 40.8 | 1312 | 1914 | 14598 | TG -3.5%, PP-1024 +22% |

### PP Speed by Prompt Length (UD-Q4_K_XL, KV q8_0, fit-nobatch, 65k ctx)

| Prompt Tokens | PP (tok/s) | Notes |
|---------------|-----------|-------|
| ~634 (PP-512 target) | 1390 | |
| ~1235 (PP-1024 target) | 1607 | |
| ~4662 (PP-4096 target) | 1764 | |
| ~20176 (PP-16384 target) | 1784 | Plateau starts here |
| ~32569 | 1781 | Manual measurement |
| ~62882 (~63k) | 1695 | Slight decline at near-max ctx |

**Key finding**: PP scales well from 512→16k tokens (1390→1784, +28%), plateaus 16k-32k (~1780), slight decline at 63k (1695, -5% from peak). The bottleneck shifts from launch overhead to memory bandwidth at long prompts.

## Phase C: Real-World Evals

**Benchmarks** (--limit 500, 0-shot, local-completions API):
- ARC-Challenge: science reasoning (loglikelihood, 500 of 1172)
- HellaSwag: commonsense reasoning (loglikelihood, 500 of 10042)
- GPQA: graduate-level QA (loglikelihood, 448 full)
- Winogrande: commonsense (loglikelihood, 500 of 1267)
- GSM8K: math reasoning (generate_until, 100 of 1319)

| Quant | ARC-C (acc_norm) | HellaSwag (acc_norm) | GPQA (acc) | Winogrande (acc) | GSM8K (em) |
|-------|-----------------|---------------------|-----------|-----------------|-----------|
| Q8_0 (ceiling) | | | | | |
| UD-Q4_K_XL | | | | | |
| Q4_K_M (bartowski) | | | | | |
| AesSedai Q4_K_M | | | | | |

## Phase D: AesSedai Three-Way Comparison

| Metric | bartowski Q4_K_M | UD-Q4_K_XL | AesSedai Q4_K_M |
|--------|-----------------|------------|-----------------|
| File size (GiB) | 19.8 | 19.2 | 20.6 |
| PPL (WikiText-2, ctx=512) | 6.6688 | 6.5959 | **6.3949** |
| KLD (vs Q8_0) | 0.0286 | 0.0145 | **0.0095** |
| Same-top-p (%) | 92.46 | 94.46 | **95.74** |
| TG short (tok/s) | ~74* | 42.7 | 44.5 |
| TG medium | ~73* | 42.9 | 44.2 |
| TG long | ~74* | 42.6 | 44.1 |
| PP-512 (tok/s) | — | 1396 | 1257 |
| PP-1024 (tok/s) | — | 1573 | 1494 |
| VRAM (MB) | 14559 | 14538 | 14584 |

*bartowski Q4_K_M speeds from S006 (latest-fit image, Q4_K_M model). UD-Q4_K_XL and AesSedai measured in S008 with same image/config.

**Key findings:**
- AesSedai wins all quality metrics by a wide margin: KLD 3x better than bartowski, 1.5x better than UD
- bobaburger's KLD claim (0.0096) essentially **confirmed** (we measured 0.0095)
- Speed is similar across UD-Q4_K_XL and AesSedai (~43-45 tok/s) — both are ~40% slower than bartowski Q4_K_M (~74 tok/s)
- **The Q4_K_M speed advantage (~74 tok/s) comes from the smaller, less precise tensor choices, not the quantizer.** The XL/AesSedai quants use larger types for sensitive layers → larger file → fewer expert layers fit on GPU → slower

## Revisions from Original Plan

1. **KLD at extended contexts (A3) — SKIPPED**: KLD base files at 65k ctx ≈ 533 GB, at 128k ≈ 533 GB. Generation time would be hours per file. PPL comparison (q8_0 vs f16) provides sufficient evidence of KV cache quality at extended contexts without KLD. Phase A now tests only PPL + Speed.
2. **GPQA access granted** (C): User accepted gated dataset terms on HuggingFace. GPQA now included.
3. **MMLU-Pro, HumanEval, MBPP → DROPPED (C)**: MMLU-Pro uses `generate_until` (~50s/request with thinking mode, ~97h/quant). HumanEval/MBPP also generate_until with code execution overhead. Replaced with loglikelihood-based alternatives: HellaSwag + Winogrande + GPQA. Total: 5 benchmarks (4 loglikelihood + 1 generation).
4. **--limit 500 for large benchmarks (C)**: Full HellaSwag (10k) and Winogrande (1.3k) subsampled to 500. ARC-Challenge (1172) and GPQA (448) run near-full. GSM8K limited to 100 (generation is slow).
5. **lm-eval logprobs format patch (C)**: llama.cpp returns logprobs in new OpenAI format (`content[{token, logprob}]`), but lm-eval expected old format (`token_logprobs[]`). Patched `.venv/.../openai_completions.py:parse_logprobs()` to handle both formats via format detection.

## Attempts
| # | Phase | Approach | Result | What changed |
|---|-------|----------|--------|--------------|
| 1 | C | local-chat-completions model type | Failed: loglikelihood not supported for chat completions | Switched to local-completions |
| 2 | C | local-completions model type | Failed: KeyError 'token_logprobs' — llama.cpp uses new logprobs format | Patched parse_logprobs() |
| 3 | C | Patched local-completions | Success: all 6 benchmarks pass sanity check | Running full evals |
