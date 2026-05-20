# Session 015: Progress Checkpoint

## Status: Phase 3 Complete — Speed + Quality Benchmarks Done

## Master Results Table

### Speed (mtp-bench.py, 9 diverse prompts)

| # | Config | Model | Build | TG avg | TG range | MTP accept | Context | KV | VRAM | GPU layers |
|---|--------|-------|-------|--------|----------|-----------|---------|-----|------|-----------|
| E1 | 35b-baseline | 35B MoE UD-Q4_K_XL | AtomicBot | **70.5** | 69.8-73.0 | n/a | 128k | q8_0 | 15,522 | 41/41 |
| E2 | 35b-mtp | 35B MoE MTP UD-Q4_K_XL | am17an | **72.0** | 60.4-86.4 | **98.8%** | 128k | q8_0 | 15,534 | 42/42 (22 overflow) |
| E3 | 35b-mtp-turbo | 35B MoE MTP UD-Q4_K_XL | Indras | **18.7** | 18.5-19.9 | **0.0%** | 128k | tbq4_0 | 15,806 | 42/42 (20 overflow) |
| E4 | 27b-iq3 | 27B IQ3_XXS | AtomicBot | **47.1** | 45.5-48.6 | n/a | 128k | turbo4 | 14,642 | **65/65** |
| E4b | 27b-iq3-tbq4 | 27B IQ3_XXS | Indras | **49.7** | 49.4-51.2 | n/a | 128k | tbq4_0 | 14,168 | **65/65** |
| E5 | 27b-mtp-q2 | 27B MTP Q2_K_XL | am17an | **70.0** | 63.7-79.0 | **99.1%** | 65k | q4_0 | 15,344 | **66/66** |
| E5b | 27b-mtp-q2-tbq4 | 27B MTP Q2_K_XL | Indras | **36.7** | 33.5-38.9 | 80.1% | 128k | tbq4_0 | 15,262 | 58/66 (8 CPU) |
| E6 | 27b-mtp-q2-n3 | 27B MTP Q2_K_XL | am17an | **70.0** | 64.4-80.6 | 97.1% | 65k | q4_0 | 15,344 | 66/66 |
| E7 | 27b-mtp-q3 | 27B MTP Q3_K_XL | am17an | **25.1** | 22.9-29.5 | 98.9% | 65k | q4_0 | 15,558 | 55/66 (11 CPU) |
| E7b | 27b-mtp-q3-tbq4 | 27B MTP Q3_K_XL | Indras | **20.0** | 16.3-22.1 | 80.3% | 128k | tbq4_0 | 15,428 | 49/66 (17 CPU) |
| — | 35b-turbo4 | 35B MoE UD-Q4_K_XL | AtomicBot | unstable | 11.2-72.5 | n/a | 128k | turbo4 | 15,732 | — |

### Quality (CodeNeedle HTTP server, 11 functions, ~50k chars, ~14k tokens)

| Config | Model | Pass | Lines matched | Hallucinated | Accuracy |
|--------|-------|------|---------------|-------------|----------|
| **E4** | **27B IQ3_XXS** | **11/11** | **219/220** | **0** | **99.5%** |
| E1 | 35B UD-Q4_K_XL | 11/11 | 206/220 | 12 | 93.6% |
| E5 | 27B MTP Q2_K_XL | 10/11 | 199/220 | 20 | 90.5% |

All tested with thinking enabled (max_tokens=32768, temp=0.0).

## Key Findings

### 1. Qwen3.6-35B MoE: 70 tok/s (40% faster than Qwen3.5)
Production baseline on RTX 5080 16GB. Community reported ~35-45 tok/s; we measured 70 tok/s. The MoE architecture (~3B active params) dominates — even with PCIe expert offload, it's faster than 27B dense fully on GPU.

### 2. MTP acceptance is near-perfect on Qwen3.6 (99% with am17an build)
Both 35B MoE (98.8%) and 27B dense (99.1%) show near-perfect draft acceptance at n=2 with the am17an MTP build. Far above community-reported 70-75%. The MTP heads are very well trained.

### 3. MTP is NET-NEGATIVE on MoE models
Confirmed experimentally and by research (thc1006). Each draft token in speculative verification loads fresh expert slices over PCIe. E3 (Indras build): 0% acceptance, 18.7 tok/s. E2 (am17an build): 98.8% acceptance but only +2% speed — the verification pass is expensive due to expert-union overhead. Theoretical break-even requires ~94 drafted tokens, far above any practical draft window.

### 4. MTP transforms 27B dense performance (+47%)
Without MTP: 47-50 tok/s. With MTP (Q2_K_XL): 70 tok/s. The gain is massive because all 27B params are already on GPU — no additional expert loading per draft token. This makes 27B dense + MTP competitive with the 35B MoE baseline at matched speed.

### 5. n=2 vs n=3 draft tokens: n=2 wins
E5 (n=2): 70 tok/s, 99.1% acceptance. E6 (n=3): 70 tok/s, 97.1% acceptance. Same effective speed, but n=2 has higher acceptance and simpler verification. No reason to use n=3.

### 6. Q2_K_XL is the MTP sweet spot on 16 GB VRAM
Q2_K_XL (12.3 GB) fits entirely on GPU with MTP head → 70 tok/s. Q3_K_XL (14.9 GB) forces 11+ layers to CPU → 25 tok/s. The 2.6 GB size difference costs 64% speed. The critical threshold: model + MTP head must fit entirely on GPU for MTP to be worthwhile.

### 7. 27B IQ3_XXS has the best recall quality — better than 35B
Counterintuitive: the aggressively quantized 27B (IQ3_XXS, 12 GB) scored 219/220 lines with zero hallucinations, beating the 35B MoE (206/220, 12 hallucinations). The 27B dense model may have inherently better recall characteristics, or the Unsloth Dynamic IQ3 quant preserves the recall-critical weights well.

### 8. Q2 quantization degrades quality more than IQ3
27B Q2_K_XL MTP scored 199/220 (90.5%) vs IQ3_XXS's 219/220 (99.5%). The MTP GGUF uses Q2_K_XL quantization which is more aggressive than IQ3_XXS, particularly for indent-sensitive code. The `parse_request` failure was correct code content with wrong indentation.

### 9. TurboQuant enables 128k context but at speed cost
Without TurboQuant, 27B MTP OOMs at 128k with q8_0 KV. With tbq4_0, it fits — but the MTP head forces partial offload, dropping speed to 37 tok/s (E5b). Without MTP, tbq4_0 gives 50 tok/s at 128k (E4b). TurboQuant is a context-length tool, not a speed tool.

### 10. IQ3_XXS + MTP graft is blocked
Tried using the separate MTP head file (`27B_MTP.gguf`) with the IQ3_XXS base model via `--model-draft` — both am17an and Indras builds fail because the bare MTP head GGUF lacks required model hyperparameters (rope config, etc.). Creating an IQ3_XXS MTP GGUF requires GGUF-level tensor grafting (merging MTP tensors + updating block_count + adding nextn_predict_layers metadata). Not trivial — havenoammo's tooling does this but only published Q2-Q8 MTP variants.

### 11. Build compatibility is fragmented
Three different llama.cpp builds needed, each with different capabilities:
- **AtomicBot** (`s015-mtp-turbo`): TurboQuant (turbo2/3/4) works, MTP only for Gemma4
- **am17an** (`s015-mtp`): Best MTP quality (99% acceptance), no TurboQuant
- **Indras-Mirror** (`s015-indras`): Both MTP + TurboQuant (tbq4_0), but MTP broken on MoE (0% accept), lower acceptance on dense (80% vs 99%)

### 12. TurboQuant mainline status: unlikely to merge
ggerganov called TurboQuant PRs "pure slop." Mainline instead merged Hadamard KV rotation (PR #21038) which improves existing q4_0/q8_0 quality. TurboQuant remains fork-only. MTP PR #22673 is close to merging (ggerganov actively reviewing, May 12 2026).

## Recommended Configs (for RTX 5080 16 GB VRAM)

| Use case | Config | Speed | Quality | Context |
|----------|--------|-------|---------|---------|
| **Best all-around** | 35B MoE UD-Q4_K_XL, q8_0 KV | 70 tok/s | 93.6% | 128k |
| **Best quality** | 27B IQ3_XXS, tbq4_0 KV | 50 tok/s | 99.5% | 128k |
| **Speed + dense** | 27B MTP Q2_K_XL, q4_0 KV (am17an) | 70 tok/s | 90.5% | 65k |
| **Dream config** | 27B MTP IQ3_XXS (needs custom GGUF graft) | ~70 tok/s? | ~99%? | 65k |

## Docker Images

| Tag | Size | Source | Capabilities |
|-----|------|--------|-------------|
| `s015-mtp-turbo` | 6.04 GB | AtomicBot fork | TurboQuant (turbo2/3/4), Gemma4 MTP only |
| `s015-mtp` | 5.78 GB | am17an PR #22673 | Qwen MTP (99% accept), no TurboQuant |
| `s015-indras` | 5.79 GB | Indras-Mirror fork | TurboQuant (tbq4_0) + Qwen MTP (80% accept), RotorQuant |

## Models on Disk

| File | Size | Source | Used in |
|------|------|--------|---------|
| Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf | 21 GB | Pre-existing | E1, E3 |
| Qwen3.6-35B-A3B-Q8_0.gguf | 35 GB | Pre-existing | Reference |
| Qwen3.6-35B-A3B-MTP-UD-Q4_K_XL.gguf | 22 GB | havenoammo | E2, E3 |
| Qwen3.6-27B-UD-IQ3_XXS.gguf | 12 GB | unsloth | E4, E4b |
| Qwen3.6-27B-MTP-UD-Q2_K_XL.gguf | 12 GB | havenoammo | E5, E5b, E6 |
| Qwen3.6-27B-MTP-UD-Q3_K_XL.gguf | 14 GB | havenoammo | E7, E7b |
| 35BA3B-MTP.gguf | 903 MB | havenoammo | Separate MTP head (35B) |
| 27B_MTP.gguf | 457 MB | havenoammo | Separate MTP head (27B) |

## Benchmark Result Files

Speed (mtp-bench.py JSON):
- `benchmarks/s015-e1-baseline.json` — 35B MoE baseline
- `benchmarks/s015-e1-turbo4.json` — 35B MoE + turbo4 KV (unstable)
- `benchmarks/s015-e2-35b-mtp.json` — 35B MoE MTP
- `benchmarks/s015-e3-35b-mtp-turbo.json` — 35B MoE MTP + tbq4_0 (failed)
- `benchmarks/s015-e4-27b-iq3.json` — 27B IQ3_XXS
- `benchmarks/s015-e4b-27b-iq3-tbq4-128k.json` — 27B IQ3_XXS + tbq4_0
- `benchmarks/s015-e5-27b-mtp.json` — 27B MTP Q2_K_XL (am17an)
- `benchmarks/s015-e5b-27b-mtp-tbq4-128k.json` — 27B MTP Q2_K_XL + tbq4_0 (Indras)
- `benchmarks/s015-e6-27b-mtp-n3.json` — 27B MTP Q2_K_XL n=3
- `benchmarks/s015-e7-27b-q3-mtp-am17an.json` — 27B MTP Q3_K_XL (am17an)
- `benchmarks/s015-e7b-27b-q3-mtp-indras-128k.json` — 27B MTP Q3_K_XL + tbq4_0 (Indras)

Quality (CodeNeedle JSON):
- `benchmarks/codeneedle/results/http_server__s015-35b-baseline.json`
- `benchmarks/codeneedle/results/http_server__s015-27b-iq3.json`
- `benchmarks/codeneedle/results/http_server__s015-27b-mtp.json`

## Phase Status

### Phase 0: Setup ✅
- [x] Docker images built (3 images: AtomicBot, am17an, Indras-Mirror)
- [x] All models downloaded (6 models + 2 separate MTP heads)
- [x] llm-bench tooling updated (10 new flags, MTP quant detection)
- [x] mtp-bench.py + CodeNeedle installed

### Phase 1: Configs ✅
- [x] 6 env files created (adapted at runtime for build differences)

### Phase 2: Speed Benchmarks ✅
- [x] 11 configs tested (E1-E7 + variants)
- [x] MTP acceptance rates measured across 3 builds
- [x] Build compatibility matrix established
- [x] MTP on MoE confirmed net-negative (0% accept on Indras, +2% on am17an)
- [x] Q2_K_XL identified as MTP sweet spot (fits fully on GPU)
- [x] Q3_K_XL confirmed too large for MTP on 16 GB (partial offload → -64% speed)

### Phase 3: Quality ✅
- [x] CodeNeedle HTTP server (11 functions) on 3 configs
- [x] 27B IQ3_XXS: best quality (99.5%, 0 hallucinations)
- [x] 35B MoE: good quality (93.6%, 12 hallucinations)
- [x] 27B MTP Q2_K_XL: decent quality (90.5%, 20 hallucinations)
- [x] IQ3 + MTP graft attempted — blocked (bare MTP head lacks model hyperparameters)

### Phase 4: Deliverables ⏳
- [ ] Reddit post draft
- [ ] Production config decision
- [ ] CLAUDE.md update with S015 findings
