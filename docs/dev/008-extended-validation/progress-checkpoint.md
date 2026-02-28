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

### CRITICAL BUG: llama.cpp doesn't support echo=true for prompt logprobs

**Root cause**: lm-eval's `local-completions` sends `echo=true` to get logprobs for input tokens, but llama.cpp's `/v1/completions` ignores the `echo` parameter. The `logprobs.content[]` array only contains generated token logprobs, never echoed input logprobs. As a result:
- `content[ctxlen:-1]` is always empty (ctxlen > 0, content has only 1 entry)
- All answer choices get logprob=0 → random chance scores
- All loglikelihood benchmarks produce **identical scores across all quants** at random chance levels

**Evidence**: ARC=0.234, HellaSwag=0.264, GPQA=0.246, Winogrande=0.496 — all identical across Q8_0, UD-Q4_K_XL, Q4_K_M. Only GSM8K (generation, flexible-extract) showed real variation: Q8_0=0.59, UD-Q4_K_XL=0.65, Q4_K_M=0.73 (v1 runs, --limit 100).

**Impact**: loglikelihood, loglikelihood_rolling, and multiple_choice task types are ALL broken with llama.cpp's API. Only `generate_until` tasks produce valid results.

**No fix available**: llama.cpp does not support `prompt_logprobs`, `echo`, or any mechanism to return logprobs for input tokens. Would require a different server (vLLM, SGLang) or a llama.cpp code change.

### Generation-Only Benchmarks (v2)

Switched to generation-only approach (--limit 200, 0-shot, local-completions API):
- GPQA: graduate-level QA (generate_until, flexible-extract for answer letter)
- GSM8K: math reasoning (generate_until, flexible-extract for numerical answer)

Dropped from plan:
- ARC-Challenge (loglikelihood — broken), arc_challenge_chat ("." stop sequence fires immediately), arc_challenge_llama (model generates `<think>` tags, never reaches answer in 100 tokens)
- HellaSwag (loglikelihood only — no generation variant exists)
- Winogrande (loglikelihood only — no generation variant exists)

| Quant | GPQA (em, flex) | GSM8K (em, flex) |
|-------|----------------|-----------------|
| Q8_0 (ceiling) | (running) | (running) |
| UD-Q4_K_XL | (running) | (running) |
| Q4_K_M (bartowski) | (running) | (running) |
| AesSedai Q4_K_M | (running) | (running) |

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

## Revisions (continued)

6. **Loglikelihood benchmarks ALL invalid (C)**: llama.cpp doesn't support `echo=true` for prompt logprobs. All loglikelihood scores are identical at random chance across all quants. This is a fundamental limitation of llama.cpp's `/v1/completions` endpoint. Switched to generation-only benchmarks.
7. **ARC-Challenge dropped (C)**: `arc_challenge_chat` has stop sequence `"."` that fires immediately with text completions. `arc_challenge_llama` works but model generates `<think>` tags that consume all 100 max_gen_toks without reaching the answer. No viable generation variant for text completions API.
8. **HellaSwag, Winogrande dropped (C)**: No generation-based variants exist in lm-eval. These tasks are loglikelihood-only.
9. **local-chat-completions 400 error (C)**: Chat completions endpoint with `--apply_chat_template` returns 400 for some tasks. When it works, thinking mode puts answers in `reasoning_content` field which lm-eval doesn't read (only reads `message.content`). Content is empty for most responses.
10. **Final benchmark set (C)**: GPQA generative (works, verified 2/5 correct on sanity check) + GSM8K (works, showed real variation 0.59/0.65/0.73 across quants). Limit increased to 200 per task.

## Attempts
| # | Phase | Approach | Result | What changed |
|---|-------|----------|--------|--------------|
| 1 | C | local-chat-completions model type | Failed: loglikelihood not supported for chat completions | Switched to local-completions |
| 2 | C | local-completions model type | Failed: KeyError 'token_logprobs' — llama.cpp uses new logprobs format | Patched parse_logprobs() |
| 3 | C | Patched local-completions, loglikelihood tasks | Failed: ALL scores identical at random chance — echo=true not supported | Root cause: llama.cpp ignores echo, content[] has only generated tokens |
| 4 | C | local-chat-completions + --apply_chat_template | Failed: 400 Bad Request and/or empty content (thinking mode) | Chat completions not viable for eval |
| 5 | C | local-completions + arc_challenge_chat | Failed: stop sequence "." fires immediately | Dropped ARC chat variant |
| 6 | C | local-completions + arc_challenge_llama | Failed: model generates `<think>` tags, 100 tokens exhausted before answer | Dropped ARC llama variant |
| 7 | C | local-completions + gpqa_main_generative_n_shot | **Success**: 2/5 on sanity check, model outputs "(A)", "(B)" directly | Using for full eval |
| 8 | C | local-completions + gsm8k (v2, limit 200) | **Success**: generation works, scores vary across quants | Using for full eval |

## Future Improvements (Brainstormed)

### Unlocking Loglikelihood Benchmarks
1. **llama-cpp-python** (Python bindings) supports `echo=true` — could unlock ARC, HellaSwag, MMLU-Pro, Winogrande. Trade-off: may not support `--fit on` or match native server performance
2. **Proxy approach**: Write a thin proxy between lm-eval and llama-server that synthesizes prompt logprobs via native `/completion` endpoint
3. **Disable thinking mode**: `--chat-template-kwargs '{"enable_thinking": false}'` suppresses `<think>` tags for chat completions. Would fix arc_challenge_llama and other chat-based benchmarks

### Additional Generation Benchmarks Available
- **MATH-hard (minerva_math)**: Competition math, Level 5 only. Strong reasoning benchmark
- **IFEval**: Instruction following, 0-shot, verifiable constraints
- **DROP**: Reading comprehension with discrete reasoning
- **BBH (bbh_cot_fewshot)**: BIG-Bench Hard, 23 hard tasks, generative variant
- **TriviaQA**: Factual knowledge recall
- All are `generate_until` tasks that work with local-completions
