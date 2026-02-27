# Session 007: Community Experiments — Matrix Benchmark Suite

Post: "Qwen3.5-35B-A3B: 74 tok/s on RTX 5080 16GB — full quantization + offload analysis"
Date: 2026-02-28 (planning), 2026-03-01+ (execution)

## Motivation

Session 006 posted results to Reddit and received significant community engagement. This session addresses the most-requested follow-up experiments, organized by demand and impact.

## Matrix Design

The experiment suite uses a **MATRIX** approach with four axes:

| Axis | Values |
|------|--------|
| **Quant** | Q4_K_M (bartowski), MXFP4_MOE (unsloth), UD-Q4_K_XL (unsloth, FIXED version), Q8_0 (reference) |
| **KV config** | f16, q8_0, q4_0, asymmetric (K=q8_0, V=q4_0) |
| **Batch config** | none (llama-server defaults: -b 2048 -ub 512), large (-b 4096 -ub 4096), asymmetric (-b 4096 -ub 2048) |
| **Context length** | 4k, 8k, 16k, 32k, 65k |

Not every cell runs. We define specific **slices** that answer community questions.

**Two-tier execution:**
- **Quick tier** (~30-60 min): Speed sweeps only. Any community member can reproduce on their hardware.
- **Full tier** (~hours): Speed + PPL + KLD quality matrix. For us and patient community members.

---

## Experiment 1: KV Cache Deep Dive

**Priority**: Highest (4 users asked, challenges "free lunch" claim)

**Who asked**: u/PhilippeEiffel, u/ArckToons, u/MrMisterShin, u/WittyAmbassador7340

**Background**: Session 006 E1 tested KV q8_0/q4_0 at 65k context only, showing < 0.4% PPL difference. Community asks: does this hold at shorter contexts? Does KV q8_0 degrade at long contexts? ArckToons specifically requested KLD (not just PPL) across KV levels.

**Matrix slice**:
- Fix quant: Q4_K_M
- Fix batch: none (no -b/-ub flags, production config)
- Sweep: 4 KV configs x 4 context lengths

| | 4k ctx | 8k ctx | 16k ctx | 32k ctx |
|---|--------|--------|---------|---------|
| KV f16 (baseline) | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM |
| KV q8_0 | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM |
| KV q4_0 | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM |
| KV asym (K=q8_0, V=q4_0) | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM | PPL, KLD, TG, PP, VRAM |

**Total runs**: 16 PPL, 16 KLD, 16 speed benchmarks (quick tier: speed only = 16 runs)

**Measurements**:
- PPL: WikiText-2 perplexity via `llama-perplexity` (580 chunks at 4k/8k, 80 chunks at 16k/32k)
- KLD: KL divergence vs f16 KV baseline at each context length (80 chunks)
- TG: Token generation speed (tok/s) from 4 workloads (short/medium/long/multi-turn)
- PP: Prompt processing speed (tok/s) at prompt lengths 512, 1024
- VRAM: Peak GPU memory from nvidia-smi

**Configs**: `llama-cpp-s007-e1-kvf16-ctx{4k,8k,16k,32k}.env`, `llama-cpp-s007-e1-kvq8-ctx{4k,8k,16k,32k}.env`, etc.
Note: For quick tier, use only the 4 KV configs at 32k context (4 speed runs, ~40 min).

**Success criteria**:
- [ ] PPL measured for all 16 cells (4 KV configs x 4 context lengths)
- [ ] KLD measured for 12 non-baseline cells (3 KV configs x 4 context lengths, vs f16 baseline at same ctx)
- [ ] TG speed (tok/s) measured for all 16 cells
- [ ] PP speed (tok/s) measured at 512 and 1024 prompt lengths for all 16 cells
- [ ] VRAM (MB) captured for all 16 cells
- [ ] Table shows whether KV q8_0 PPL delta increases with context length (the key question)
- [ ] KLD numbers for KV q8_0/q4_0/asym vs f16 at each context length (addresses ArckToons)
- [ ] Clear statement: "KV q8_0 free lunch holds/does-not-hold across context lengths"
- [ ] VRAM savings quantified: f16 vs q8_0 vs q4_0 vs asym at each context length

---

## Experiment 2: PP vs TG Tradeoff

**Priority**: High (5 users flagged as critical missing data)

**Who asked**: u/KierkegaardsSisyphus, u/Chromix_, u/guiopen, u/wisepal_app, u/DonkeyBonked

**Background**: Session 006 showed that removing -b/-ub flags boosts TG by +7% (fit-nobatch). But we never measured the PP (prompt processing) cost. Users doing RAG, long-document ingestion, or batch inference need PP speed. KierkegaardsSisyphus specifically suggested asymmetric -b 4096 -ub 2048 as a middle ground.

**Matrix slice**:
- Fix quant: Q4_K_M
- Fix KV: q8_0
- Fix context: 32k
- Sweep: 3 batch configs

| Batch config | PP@512 | PP@1024 | PP@4096 | PP@16384 | TG (4 workloads) | VRAM |
|---|---|---|---|---|---|---|
| none (no -b/-ub, production) | tok/s | tok/s | tok/s | tok/s | tok/s | MB |
| large (-b 4096 -ub 4096) | tok/s | tok/s | tok/s | tok/s | tok/s | MB |
| asymmetric (-b 4096 -ub 2048) | tok/s | tok/s | tok/s | tok/s | tok/s | MB |

**Total runs**: 3 speed benchmarks (each measures PP at 4 lengths + TG at 4 workloads)

**Measurements**:
- PP: Use `/completion` endpoint with `n_predict: 1`, extract `timings.prompt_per_second` at prompt lengths 512, 1024, 4096, 16384
- TG: Use `/v1/chat/completions`, extract generation speed from 4 workloads
- VRAM: Peak GPU memory from nvidia-smi

**PP measurement method**: Send a prompt of the target length with `n_predict: 1` (generate only 1 token). The server returns `timings.prompt_per_second` which is the pure PP speed uncontaminated by generation. Use a fixed text corpus (e.g., WikiText excerpt) to ensure consistent prompt content across runs.

**Configs**: `llama-cpp-s007-e2-batch-none.env`, `llama-cpp-s007-e2-batch-large.env`, `llama-cpp-s007-e2-batch-asym.env`

**Success criteria**:
- [ ] PP speed measured at 512, 1024, 4096, 16384 prompt lengths for all 3 batch configs
- [ ] TG speed measured (4 workloads) for all 3 batch configs
- [ ] VRAM captured for all 3 configs
- [ ] PP ratio calculated: `PP_nobatch / PP_large` to quantify the PP penalty of removing -b/-ub
- [ ] Asymmetric config (-b 4096 -ub 2048) compared: does it get most of the PP benefit with minimal TG cost?
- [ ] Clear recommendation: which batch config for PP-heavy workloads vs TG-heavy workloads
- [ ] If PP penalty is > 30%, document the tradeoff and suggest asymmetric as compromise

---

## Experiment 3: MXFP4 Redemption

**Priority**: High (4 users disputed our S006 result, one claims 77 tok/s)

**Who asked**: u/KierkegaardsSisyphus, u/ayylmaonade, u/jumpingcross, u/danielhanchen

**Background**: Session 006 E7 showed MXFP4_MOE at ~47 tok/s (34-42% slower than Q4_K_M). KierkegaardsSisyphus reports 77 tok/s on their 5080 with `--fit-target 1500` and `-b 4096 -ub 2048`. This is a massive discrepancy — either our config was suboptimal or their measurement differs. We need to test their exact config.

**Configs to test**:

| Config | Description |
|--------|-------------|
| MXFP4 + fit-target 1500 | KierkegaardsSisyphus's config |
| MXFP4 + fit-target 1500 + batch asym | KierkegaardsSisyphus's full config (-b 4096 -ub 2048) |
| MXFP4 + fit default (no fit-target) | Our S006 config (baseline comparison) |
| Q4_K_M + fit-nobatch | Our production baseline (control) |

**Measurements**:
- TG: 4 workloads (short/medium/long/multi-turn)
- PP: At prompt lengths 512, 1024, 4096
- VRAM: Peak GPU memory from nvidia-smi
- PPL: WikiText-2 perplexity (full tier only, 40 chunks due to MXFP4 memory leak)
- KLD: KL divergence vs Q8_0 (full tier only, 31 chunks)

**Prerequisites**:
- Download MXFP4_MOE model: `huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-MXFP4_MOE.gguf --local-dir models/`
- Model was deleted in S006 cleanup, needs re-download (~18.4 GB)

**Configs**: `llama-cpp-s007-e3-mxfp4-fit1500.env`, `llama-cpp-s007-e3-mxfp4-fit1500-batchasym.env`, `llama-cpp-s007-e3-mxfp4-fitdefault.env`

**Success criteria**:
- [ ] MXFP4 tested with `--fit-target 1500` (KierkegaardsSisyphus's config)
- [ ] MXFP4 tested with `--fit-target 1500` + `-b 4096 -ub 2048` (full KierkegaardsSisyphus config)
- [ ] MXFP4 baseline (fit default, no fit-target) reproduced from S006
- [ ] Q4_K_M baseline reproduced for control
- [ ] TG speed comparison: can MXFP4 reach > 70 tok/s with fit-target 1500?
- [ ] If MXFP4 > 70 tok/s: document the config and recommend it. If not: quantify the gap and explain why our result differs from KierkegaardsSisyphus
- [ ] PP speed measured at 512, 1024, 4096 for MXFP4 configs (addresses batch interaction question)
- [ ] VRAM usage compared: does fit-target 1500 allocate more expert layers to GPU?
- [ ] PPL/KLD measured (full tier only): confirm MXFP4 quality vs Q4_K_M

---

## Experiment 4: Fixed UD-Q4_K_XL

**Priority**: Medium-High (Daniel's new data shows bug was fixed)

**Who asked**: u/danielhanchen (Unsloth creator)

**Background**: Session 006 E2 showed UD-Q4_K_XL with 3.9x worse KLD than Q4_K_M (0.109 vs 0.028). danielhanchen acknowledged a bug in the old quant and says it has been fixed. His new numbers: UD-Q4_K_XL KLD 0.0137 vs bartowski Q4_K_M KLD 0.0182. If true, UD-Q4_K_XL would be the better quant. We must verify independently.

**Steps**:
1. Download the FIXED UD-Q4_K_XL from unsloth (NOT the old buggy file deleted in S006)
2. Run quality benchmarks: PPL, KLD (same methodology as S006 E2)
3. Run speed benchmark (same workloads as S006)
4. Compare against bartowski Q4_K_M (our production)

**Download command**:
```bash
huggingface-cli download unsloth/Qwen3.5-35B-A3B-GGUF Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf --local-dir models/
```

**Measurements**:
- PPL: WikiText-2 perplexity (580 chunks, full run)
- KLD: KL divergence vs Q8_0 reference (80 chunks)
- TG: 4 workloads (short/medium/long/multi-turn) with fit-nobatch config
- VRAM: Peak GPU memory from nvidia-smi

**Configs**: `llama-cpp-s007-e4-ud-q4kxl-speed.env`, `llama-cpp-s007-e4-ud-q4kxl-ppl.env`, `llama-cpp-s007-e4-ud-q4kxl-kld.env`

**Success criteria**:
- [ ] FIXED UD-Q4_K_XL downloaded fresh from unsloth (verify file hash differs from S006 version)
- [ ] PPL measured: full 580 chunks on WikiText-2
- [ ] KLD measured: 80 chunks vs Q8_0 base logits
- [ ] "Same top %" measured from KLD output
- [ ] TG speed measured: 4 workloads with fit-nobatch config
- [ ] VRAM captured
- [ ] Head-to-head comparison table:

| Metric | Q4_K_M (bartowski) | UD-Q4_K_XL (FIXED) | S006 UD-Q4_K_XL (BUGGY) |
|--------|-------------------|---------------------|--------------------------|
| PPL | 6.6688 | measured | 7.1702 |
| KLD | 0.0282 | measured | 0.1087 |
| Same top % | 92.4% | measured | 86.2% |
| TG speed | ~74 tok/s | measured | N/A |
| File size | 20 GB | measured | 19.8 GB |

- [ ] If FIXED UD KLD < 0.0282 AND speed >= 70 tok/s: recommend as new production quant
- [ ] If FIXED UD KLD > 0.0282: document improvement over buggy version but keep Q4_K_M as production
- [ ] Verify Daniel's claim: KLD 0.0137 for UD vs 0.0182 for Q4_K_M (within measurement noise?)

---

## Experiment 5: Vision Mode

**Priority**: Medium (3 users asked, infrastructure ready)

**Who asked**: u/guiopen, u/wisepal_app, u/moahmo88

**Background**: mmproj-BF16.gguf (~861 MB) is already downloaded. Qwen3.5-35B-A3B supports vision via the multimodal projector. llama.cpp b8177+ supports multi-modal prompt caching. Need to measure VRAM impact and run quality evaluation.

**Steps**:
1. Rebuild Docker image pinned to b8177+ (`--build-arg LLAMA_CPP_REF=b8177`)
2. Test server startup with `--mmproj /models/mmproj-BF16.gguf --fit-target 2000`
3. Measure VRAM with and without mmproj loaded
4. Measure TG speed impact (text-only prompts, with mmproj loaded vs without)
5. Measure PP speed with image input (single image, 224x224 and 768x768)
6. Run lmms-eval quality benchmarks

**VRAM budget**: mmproj-BF16.gguf is ~861 MB. `--fit-target 2000` reserves 2000 MB headroom (vs default ~500 MB). This should leave enough for the vision encoder while keeping most expert layers on GPU.

**Quality evaluation**: Install lmms-eval in a venv, run against local server:
```bash
python3 -m lmms_eval \
  --model openai_compatible \
  --model_args "base_url=http://localhost:8080/v1" \
  --tasks ocrbench,mmmu_val,mathvista_testmini,realworldqa,ai2d \
  --limit 10 \
  --output_path benchmarks/vision/
```

**Benchmarks** (5 tasks, --limit 10 each = 50 samples total):

| Benchmark | Tests | Official Qwen3.5 Score | Pass Threshold |
|-----------|-------|----------------------|----------------|
| OCRBench | OCR accuracy | TBD (look up) | > 50% of official |
| MMMU (val) | Multi-discipline understanding | TBD | > 50% of official |
| MathVista (testmini) | Math visual reasoning | TBD | > 50% of official |
| RealWorldQA | Real-world visual QA | TBD | > 50% of official |
| AI2D | Science diagram understanding | TBD | > 50% of official |

Note: With --limit 10, results will have high variance. Goal is smoke test (does it work? is quality ballpark reasonable?) not rigorous evaluation.

**Configs**: `llama-cpp-s007-e5-vision.env`

**Docker**: Needs new image build: `docker build --build-arg LLAMA_CPP_REF=b8177 -t llm-server/llama-cpp:latest-vision -f docker/Dockerfile.llama-cpp .`

**Success criteria**:
- [ ] Docker image built pinned to b8177+ with vision support
- [ ] Server starts successfully with `--mmproj` and `--fit-target 2000`
- [ ] VRAM measured: with mmproj loaded vs without (quantify the overhead)
- [ ] TG speed measured: text-only prompts with mmproj loaded (quantify slowdown vs no mmproj)
- [ ] PP speed measured: single image input at 224x224 and 768x768 resolutions
- [ ] lmms-eval completes for all 5 benchmarks (50 samples total)
- [ ] Accuracy table filled for all 5 benchmarks
- [ ] Scores compared against official Qwen3.5 published benchmarks
- [ ] If score < 50% of official on any benchmark: flag as potential quantization degradation issue
- [ ] Clear recommendation: is vision mode usable in production? What VRAM/speed tradeoff?

---

## Experiment 6: Community Notes (Documentation)

**Priority**: Medium (addresses non-NVIDIA users, no benchmarks needed)

**Who asked**: u/Corosus (Vulkan), u/Psyko38 (ROCm), multiple 8GB/24GB VRAM users

**Background**: Our results are NVIDIA-specific with `--fit on`. Community reports show `--fit` is broken on Vulkan and underperforms on ROCm. Users with different VRAM amounts need guidance.

**Deliverables**:

### AMD/ROCm Guidance
- `--fit on` underperforms: Psyko38 reports 15.82 tok/s with --fit vs 37.74 tok/s manual (-ngl 999, --n-cpu-moe 24)
- Recommended config: `-ngl 999 --n-cpu-moe 24` (matches our S005 C7 approach)
- Estimate: ~38 tok/s on RX 7900 XTX 24GB (from Psyko38's data)
- Note: ROCm llama.cpp builds may not support all flags

### Vulkan Guidance
- `--fit on` is broken on Vulkan: Corosus reports 13 tok/s with --fit vs 33 tok/s manual
- Recommended config: manual layer split with `--n-cpu-moe`
- Estimate: ~33 tok/s on RTX 3090 24GB via Vulkan (from Corosus's data)

### 8GB VRAM Guidance
- Q4_K_M (20 GB) won't fit meaningfully on 8GB GPU
- Best approach: full CPU offload `-ot "exps=CPU"` or `-ngl 0`
- Expected speed: ~35-50 tok/s depending on RAM bandwidth (DDR5 vs DDR4)
- Alternative: use Q4_0 or IQ4_XS for smaller model size

### 24GB VRAM Guidance
- Q4_K_M (20 GB): most/all expert layers fit on GPU
- Try: `--fit on` (should put 30-40/40 MoE layers on GPU)
- Expected speed: ~90-110 tok/s (extrapolation from our 16/40 = 74 tok/s)
- Alternative: Q8_0 (37 GB) with partial offload may be viable

### LM Studio Limitations
- LM Studio uses llama.cpp backend but may not expose all flags
- `--fit on` may not be available in LM Studio GUI
- Manual `-ngl` and `--n-cpu-moe` should work
- Recommend llama-server directly for full control

**Success criteria**:
- [ ] AMD/ROCm section written with Psyko38's data (15.82 vs 37.74 tok/s)
- [ ] Vulkan section written with Corosus's data (13 vs 33 tok/s)
- [ ] 8GB VRAM guidance written with expected performance range
- [ ] 24GB VRAM guidance written with expected performance range
- [ ] LM Studio limitations documented
- [ ] All community data properly attributed
- [ ] Each section includes a recommended launch command
- [ ] Document saved to `docs/dev/007-community-experiments/community-notes.md`

---

## Execution Order

| # | Experiment | Tier | Estimated Time | Dependencies |
|---|-----------|------|---------------|-------------|
| 1 | E2: PP vs TG Tradeoff | Quick | ~40 min | None — uses existing Q4_K_M model |
| 2 | E1: KV Cache Deep Dive (speed only) | Quick | ~60 min | None — uses existing Q4_K_M model |
| 3 | E3: MXFP4 Redemption | Quick | ~30 min | Download MXFP4_MOE (~18.4 GB) |
| 4 | E4: Fixed UD-Q4_K_XL | Quick+Full | ~60 min | Download FIXED UD-Q4_K_XL (~20 GB) |
| 5 | E1: KV Cache Deep Dive (PPL+KLD) | Full | ~3 hrs | E1 speed complete |
| 6 | E5: Vision Mode | Full | ~90 min | Docker rebuild (b8177+), lmms-eval venv |
| 7 | E6: Community Notes | Docs | ~30 min | None |

**Total quick tier**: ~2.5 hours
**Total full tier**: ~6+ hours (including quick tier)

---

## New Infrastructure

### Scripts

| Script | Purpose | Details |
|--------|---------|---------|
| `scripts/bench-matrix.sh` | Universal benchmark runner | Takes matrix axes as arguments, measures PP + TG + VRAM |
| `scripts/compare-matrix.py` | Matrix comparison tool | Reads matrix results, outputs tables grouped by any axis |
| `scripts/vision-eval.sh` | Vision quality evaluation | Runs lmms-eval against local server |

### Results Directory

```
benchmarks/
  matrix/          # New: all matrix benchmark results
    {quant}-{kv}-{batch}-{ctx}-{timestamp}.json
  vision/          # New: lmms-eval results
  perplexity/      # Existing: PPL logs
  kl-divergence/   # Existing: KLD logs and base logits
```

### Docker Images

| Tag | Base commit | Purpose |
|-----|-------------|---------|
| `latest-fit` | b8149 | Production (--fit on support) — existing |
| `latest-vision` | b8177+ | Vision mode (--mmproj + multi-modal prompt caching) — E5 only |

---

## Deliverables

1. **Benchmark data**: All measurements in `benchmarks/matrix/` and `benchmarks/vision/`
2. **Success criteria**: This document, with all checkboxes filled
3. **Community notes**: `docs/dev/007-community-experiments/community-notes.md`
4. **Reddit follow-up post**: Results formatted for r/LocalLLaMA
5. **CLAUDE.md updates**: If any experiment changes the production config or recommendation
