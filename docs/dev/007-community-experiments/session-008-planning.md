# Session 008 Planning: Holistic Validation & Community Follow-up

Date: 2026-02-28
Status: Planning (for next session)

## Context

Session 007 completed 6 experiments (E1-E6) but several gaps were identified through self-review and community feedback. Daniel (Unsloth) published a major new post retiring MXFP4 and showing that PPL/KLD can be misleading — real-world evals diverge from perplexity metrics. New Reddit comments (bobaburger, OsmanthusBloom, CATLLM, KeldenL) suggest additional tests.

## Data Integrity Audit

**Status**: Background audit was running at end of S007. Preliminary spot-checks show numbers match raw logs (e.g., q8_0 4k PPL: claimed 5.9747, log shows Mean PPL(Q) = 5.974672). Full audit report pending — **check `/tmp/claude-1000/-home-bppc-projects-llm-server/tasks/a048de03c9e4fc740.output` at start of S008** (or re-run audit if file was cleaned up).

**Known risk**: S007 ran across two context windows with a compaction boundary. The E1 Full Tier results for 4k/8k were extracted before compaction, 16k/32k after. Need to verify all 28 E1 log files match the progress-checkpoint.md tables.

**Action**: Run the audit subagent again at session start if the output file is gone.

## Correction: Qwen3.5 Context Length

**IMPORTANT**: CLAUDE.md says 65536 context length. The actual native context is **262,144 tokens**, extensible up to **1,010,000 tokens** (via YaRN or similar). Our production config uses `-c 65536` which is a conservative choice, not the model's limit. S008 should test at the model's native context range.

## Gap Analysis

### Priority 1: Extended Context KV Quality (4k → 262k)

**Why**: We only tested 4k-32k. Our production config is 65k, and the model natively supports 262k. The community specifically asked about long-context KV degradation.

**What**: Extend E1 Full Tier to include:
- 65k (our production config — MUST test)
- 128k (common large-context use case)
- 262k (native model limit)

**Risk**: The int32 overflow bug we patched was for `i*nv` at ctx >= 8648. At ctx=262144 with i up to 131071, the product `131071 * 248320 = 32,541,466,720` — this fits in int64 (`size_t`) but need to verify our Dockerfile sed patches handle this correctly. The `n_ctx * nv = 262144 * 248320 = 65,108,213,760` also fits in size_t (64-bit).

**Practical issue**: KLD base file at 262k would be enormous (~16 TB for full WikiText-2). Must use very small chunk counts (2-5 chunks) or find an alternative methodology.

**Estimated time**: ~2-4 hours for PPL at 65k/128k/262k (few chunks each). KLD may not be feasible at 262k due to disk requirements.

### Priority 2: Real-World Downstream Evals

**Why**: Daniel's post explicitly says "lower PPL/KLD doesn't necessarily translate to better real-world performance." His MiniMax-M2.5 data shows UD IQ2_XXS outperforming AesSedai IQ3_S on LiveCodeBench/MMLU Pro despite worse PPL/KLD. Our entire S006/S007 methodology relies on PPL/KLD as proxies.

**What**: Run downstream task benchmarks comparing Q4_K_M vs UD-Q4_K_XL:
- **LiveCodeBench** (coding quality — most relevant for our use case)
- **MMMU Pro** (multimodal understanding)
- **GPQA** (graduate-level QA)
- **Math500** (mathematical reasoning)

**Tool**: lm-evaluation-harness (NOT lmms-eval which is broken). Can run against the OpenAI-compatible API.

**Alternative**: Manual eval with a curated set of 20-30 prompts covering our actual use cases (classification, synthetic data gen, agentic workflows). Less rigorous but faster and more relevant.

### Priority 3: AesSedai Q4_K_M Quant

**Why**: bobaburger's data shows AesSedai Q4_K_M with KLD 0.0096 (5060 Ti), which is significantly better than both our Q4_K_M (0.0286) and UD-Q4_K_XL (0.0145). However, it's the slowest in bobaburger's tests. Need to verify independently.

**What**:
1. Download AesSedai Q4_K_M GGUF
2. Run same benchmarks: PPL, KLD vs Q8_0, speed (TG + PP)
3. Head-to-head vs bartowski Q4_K_M vs UD-Q4_K_XL

**Note**: bobaburger's KLD numbers are from Unsloth's published data, not independent measurement. Our KLD methodology (80 chunks, ctx=512, vs Q8_0) should give comparable numbers.

### Priority 4: `--no-kv-offload` Flag

**Why**: CATLLM asked about this. Offloading KV cache to RAM could free GPU VRAM for more expert layers, potentially boosting speed. With only 10 KV cache layers (hybrid SSM architecture), the KV cache is small — so savings might be minimal.

**What**:
1. Test `--no-kv-offload` with Q4_K_M + `--fit on`
2. Measure VRAM (does it free significant space?)
3. Measure TG speed (does `--fit` use freed VRAM for more experts?)
4. Measure PP speed (does KV in RAM slow down attention?)

**Hypothesis**: Minimal impact since only 10/40 layers have KV cache, and the cache is already q8_0 quantized (small). But worth testing.

### Priority 5: ubatch 1024 Sweet Spot

**Why**: OsmanthusBloom's data (RTX 3060 6GB) shows ubatch 1024 gives 57% PP speedup over 512 with almost no TG penalty (23.62 vs 23.90 tok/s). We tested ubatch 512/2048/4096 but not 1024.

**What**: Test `-ub 1024` (with `-b 2048`) against our no-batch baseline:
- PP at 512, 1024, 4096, 16384 prompt lengths
- TG at 4 workloads
- VRAM impact

**Note**: Our E2 results showed batch flags hurt TG by 35% on RTX 5080. But that was with `-ub 2048` and `-ub 4096`. The -ub 1024 sweet spot might behave differently with `--fit on` since it uses less VRAM than 2048/4096.

### Priority 6: Harder Vision Benchmarks

**Why**: Our E5 vision test used trivially easy images (red square, color gradient). Real vision tasks involve OCR, diagrams, math equations, real-world scenes.

**What**:
1. Fix lmms-eval (try older version, e.g., 0.5.x, or install from GitHub HEAD)
2. If lmms-eval won't work, do manual testing with harder images:
   - Screenshot of code (test OCR + code understanding)
   - Math equation image (test LaTeX/math reasoning)
   - Chart/graph (test data extraction)
   - Real-world photo (test scene description)
   - Document page (test document understanding)
3. Measure PP speed with larger images (1024x1024, 2048x2048)
4. Test multi-image prompts

### Priority 7: PP Speed at Longer Prompts

**Why**: E2 only measured PP at 512 and 1024 tokens. mr_Owner and OsmanthusBloom both emphasize PP speed matters. For agentic workloads (Claude Code replacement use case), prompts can be 16k-65k tokens.

**What**: Extend PP measurement to 4096, 16384, 32768, and 65536 prompt lengths for:
- No-batch (production)
- ubatch 1024 (new sweet spot test)
- ubatch 2048 (for comparison)

### Priority 8: Updated Unsloth Quants

**Why**: Daniel's new post says:
- MXFP4 retired from all GGUF quants
- Tool calling chat template bug fixed
- 112B and 27B still being converted with new fixes
- Qwen3.5-35B-A3B GGUFs are updated — **may need to re-download UD-Q4_K_XL**

**What**:
1. Check if our UD-Q4_K_XL file is the latest version (compare hash with HF)
2. If updated, re-download and re-run E4 quality benchmarks
3. Consider testing the new UD-Q3_K_M (OsmanthusBloom used it, 15.53 GiB)

## Community Comments to Address (Our Post)

| Commenter | Question/Point | Response Plan |
|-----------|---------------|---------------|
| **KeldenL** | "whose Q4 quant are you using?" | Clarify in post update: bartowski Q4_K_M (production), compared against Unsloth UD-Q4_K_XL (new best) |
| **CATLLM** | "`--no-kv-offload` worth it?" | Test in S008 Priority 4 |
| **bobaburger** (5060 Ti 16GB) | Full quant comparison data (UD-Q4_K_M, AesSedai Q4_K_M, IQ3_S, MXFP4, UD-Q4_K_XL) | Acknowledge his data, test AesSedai Q4_K_M ourselves (Priority 3) |
| **OsmanthusBloom** | ubatch 1024 sweet spot, PP speed important | Test ubatch 1024 (Priority 5), extend PP measurements (Priority 7) |
| **mr_Owner** | Include PP speed alongside TG | Address in Priority 7 |

## Community Comments (Daniel's Post)

| Key Point | Relevance to Us |
|-----------|----------------|
| MXFP4 retired | Aligns with our E3 finding. Can stop testing MXFP4 |
| PPL/KLD can be misleading | Major methodological concern. Need real-world evals (Priority 2) |
| Tool calling chat template bug fixed | May affect our server config. Test with `--jinja` flag |
| BF16 used as KLD base (not Q8_0) | Our methodology uses Q8_0 as base. Numbers not directly comparable. Document the difference |
| `LLAMA_SET_ROWS=1 --fit off --batch-size 16384 --ubatch-size 16384 --ctx-size 512` | Daniel's perplexity config uses different settings than ours. Their batch/ubatch is much larger |
| 99.9% KLD metric used (we use Mean KLD) | Different metric — 99.9% KLD captures worst-case tokens. We should report both |

## Proposed Execution Order

| # | Experiment | Priority | Estimated Time | Dependencies |
|---|-----------|----------|---------------|-------------|
| 1 | Data audit (verify S007 numbers) | P0 | 15 min | Check if audit output exists |
| 2 | Context correction (update CLAUDE.md: 262k native) | P0 | 5 min | None |
| 3 | Extended context KV quality (65k, 128k) | P1 | 2-3 hrs | Verify overflow fix handles 65k+ |
| 4 | `--no-kv-offload` test | P4 | 30 min | None |
| 5 | ubatch 1024 sweet spot | P5 | 30 min | None |
| 6 | PP at longer prompts (4k-65k) | P7 | 45 min | None |
| 7 | AesSedai Q4_K_M download + bench | P3 | 1.5 hrs | Download ~20 GB |
| 8 | Check UD-Q4_K_XL version freshness | P8 | 10 min | None |
| 9 | Real-world evals (lm-eval-harness or manual) | P2 | 2-4 hrs | Setup lm-eval-harness |
| 10 | Harder vision benchmarks | P6 | 1-2 hrs | Fix lmms-eval or manual test |
| 11 | Reddit post update & reply to comments | — | 30 min | After experiments |

**Total estimated**: 8-12 hours (can be split across sessions)

## Key Files

- S007 results: `docs/dev/007-community-experiments/progress-checkpoint.md`
- S007 success criteria: `docs/dev/007-community-experiments/success-criteria.md`
- S007 audit output: `/tmp/claude-1000/-home-bppc-projects-llm-server/tasks/a048de03c9e4fc740.output` (may be cleaned up)
- Daniel's new post PDF: `docs/New Qwen3.5-35B-A3B Unsloth Dynamic GGUFs + Benchmarks _ r_LocalLLaMA.pdf`
- Community feedback PDF: `docs/lesson-from-reddit-2.pdf`
- Benchmark logs: `benchmarks/kl-divergence/e1-kld-*`, `benchmarks/matrix/e*`
- Docker image: `llm-server/llama-cpp:latest` (HEAD ecbcb7e, has overflow fix + vision support)
- Dockerfile with overflow patches: `docker/Dockerfile.llama-cpp`

## Model Context Correction

- ✅ `CLAUDE.md`: Updated to "262,144 native context (1,010,000 with extension), production config uses 65536"
- Production launch command: Consider testing with `-c 131072` or `-c 262144`
- KV cache budget: at 262k with q8_0, 10 attention layers → 2,560 MiB KV cache. This is significant and may affect `--fit` VRAM allocation

## Open Questions for User

1. Should we switch production to UD-Q4_K_XL based on S007 E4 results? (Better quality, similar speed)
2. How much time to invest in real-world evals vs more speed/quality matrix testing?
3. Should we test at 262k context or is 128k sufficient for practical use?
4. Priority between AesSedai Q4_K_M testing vs extended context testing?
5. Do we want to re-download UD-Q4_K_XL if Daniel updated it?
