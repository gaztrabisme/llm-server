# Session 007 Progress Checkpoint

Date: 2026-02-28 ~04:20

## Completed

### E2: PP vs TG Tradeoff (Quick Tier)
All 3 batch configs benchmarked at 32k context with Q4_K_M + KV q8_0.

| Batch config | PP@512 | PP@1024 | TG short | TG med | TG long | VRAM |
|---|---|---|---|---|---|---|
| **none** (production) | **1332** | 1605 | **57.2** | **43.5** | **43.8** | 14550 |
| large (-b 4096 -ub 4096) | 1085 | 1627 | 37.1 | 35.8 | 35.5 | 14616 |
| asym (-b 4096 -ub 2048) | 1234 | **1734** | 38.5 | 36.4 | 37.4 | 14608 |

**Finding**: Batch flags hurt TG by ~35% and don't help PP at short prompts. Only at 1024 tokens, asym is 8% faster PP. **No-batch wins** for TG-heavy workloads.

### E1: KV Cache Deep Dive (Speed Tier)
All 16 cells (4 KV configs × 4 context lengths) benchmarked.

**Key finding**: All KV configs show remarkably similar speed (~45-57 tok/s TG, ~1330-1700 PP). VRAM usage consistent (~14490-14630 MB). `--fit on` auto-adjusts expert layers to fill available VRAM regardless of KV type. **No speed degradation at longer contexts**.

### E3: MXFP4 Redemption
| Config | PP@512 | PP@1024 | TG (avg) | VRAM |
|---|---|---|---|---|
| MXFP4 + fit-target 1500 | 1586 | 1908 | 46.4 | 14079 |
| MXFP4 + fit-target 1500 + batch asym | 1428 | 2113 | 45.7 | 14081 |
| **MXFP4 + fit default** | **1618** | **1959** | **52.4** | 14627 |
| Q4_K_M baseline | 1332 | 1630 | 46.0 | 14515 |

**Finding**: MXFP4 does NOT reach 77 tok/s. Best is 52.8 tok/s with fit default. fit-target 1500 reduces VRAM (14079 vs 14627) which hurts TG. MXFP4 has better PP due to smaller model, but TG is similar/slightly better than Q4_K_M.

### E4: UD-Q4_K_XL (Speed + PPL)
| Metric | Q4_K_M (bartowski) | UD-Q4_K_XL (FIXED) | UD-Q4_K_XL (BUGGY) |
|---|---|---|---|
| TG (avg) | 46.0 | 48.1 | N/A |
| PP@512 | 1400 | 1362 | N/A |
| PP@1024 | 1566 | 1645 | N/A |
| VRAM | 14515 | 14579 | N/A |
| **PPL (580 chunks)** | **6.6688** | **6.5959** | **7.1702** |

**Massive improvement!** Fixed UD-Q4_K_XL PPL = 6.5959 (vs 7.1702 buggy, vs 6.6688 for Q4_K_M). Speed is similar. KLD comparison in progress.

### E6: Community Notes
Written to `docs/dev/007-community-experiments/community-notes.md`.

### Downloads
- UD-MXFP4_MOE: 19.5 GB ✓
- UD-Q4_K_XL: 20.6 GB ✓

### E1: KV Cache Deep Dive (Full Tier — PPL + KLD)
All 16 PPL cells and 12 KLD cells measured across 4 KV configs × 4 context lengths (4k/8k/16k/32k).

**Bug found & fixed**: llama-perplexity KLD base mode has int32 overflow when `n_ctx * n_vocab > INT_MAX` (ctx >= 8648 for Qwen3.5 n_vocab=248320). Five overflow sites in `perplexity.cpp` — patched via sed in Dockerfile, rebuilt image. Filed as upstream issue.

**PPL (WikiText-2)**:

| KV Config | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| f16 (baseline) | 5.9803 | 5.9948 | 5.6290 | 6.5582 |
| q8_0 | 5.9747 | 5.9991 | 5.6374 | 6.5758 |
| q4_0 | 5.9964 | 6.0079 | 5.6414 | 6.5887 |
| asym (K=q8_0, V=q4_0) | 5.9908 | 6.0199 | 5.6458 | 6.6000 |

**KLD vs f16 baseline (Mean KLD / Same-top-p)**:

| KV Config | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| q8_0 | 0.0176 / 95.30% | 0.0157 / 95.28% | 0.0139 / 95.33% | 0.0186 / 95.15% |
| q4_0 | 0.0241 / 94.04% | 0.0237 / 94.02% | 0.0211 / 94.04% | 0.0267 / 93.89% |
| asym | 0.0219 / 94.41% | 0.0200 / 94.43% | 0.0180 / 94.46% | 0.0226 / 94.32% |

**KV Cache Size (MiB, 10 attention layers)**:

| KV Config | 4k | 8k | 16k | 32k |
|---|---|---|---|---|
| f16 | 80 | 160 | 320 | 640 |
| q8_0 | 42.5 | 85 | 170 | 340 |
| q4_0 | 22.5 | 45 | 90 | 180 |
| asym | 32.5 | 65 | 130 | 260 |

**Key findings**:
1. **KV q8_0 is a confirmed free lunch**: PPL delta < 0.3% and KLD < 0.019 across ALL context lengths
2. **No degradation at longer contexts**: KLD actually DECREASES from 4k→16k, slight uptick at 32k but still well within acceptable range
3. **q4_0 is also usable**: ~37% worse KLD than q8_0, but absolute KLD still very low (< 0.027)
4. **asym (K=q8_0, V=q4_0) is between q8_0 and q4_0**: decent middle ground
5. **KV cache savings**: q8_0 saves 47% vs f16, q4_0 saves 72%, asym saves 59%. These savings free VRAM for more expert layers via `--fit on`
6. **Hybrid architecture**: Qwen3.5 has only 10 KV cache layers (full attention every 4th layer), rest use SSM — so KV cache is much smaller than expected

### E5: Vision Mode
Docker image (latest, HEAD ecbcb7e) supports `--mmproj`. Server starts with `--mmproj /models/mmproj-BF16.gguf --fit-target 2000`.

**Speed and VRAM**:

| Metric | Vision config | Production (no mmproj) | Delta |
|---|---|---|---|
| TG (text-only, tok/s) | 49.5 | ~74 | -33% |
| PP (text-only, tok/s) | 218 | ~1332 | N/A (different prompt lengths) |
| VRAM idle (MB) | 14664 | 14550 | +114 |
| VRAM peak (MB) | 14748 | 14550 | +198 |

**Image processing**:

| Image size | Decode time | Prompt tokens | PP (tok/s) | TG (tok/s) |
|---|---|---|---|---|
| 100x100 | 56-68 ms | 32 | 143-155 | 49-52 |
| 768x768 | 458 ms | 599 | 838 | 46.6 |

**Quality smoke test**: Model correctly identified a solid red image as "Red" (RGB 255,0,0) and described a 768x768 gradient pattern accurately including color transitions.

**lmms-eval**: Blocked. v0.6.1 has 0 registered tasks (task YAML configs missing in this release, Python-only task definitions not loading). Manual testing confirms vision works.

**Finding**: Vision mode works but `--fit-target 2000` reduces TG by ~33% (49.5 vs 74 tok/s) because it reserves extra VRAM headroom, putting fewer expert layers on GPU. The mmproj itself adds minimal overhead (~114 MB VRAM, 56-458ms decode time). Usable for occasional vision queries but not recommended as default config for text-heavy workloads. Image PP is fast (838 tok/s at 768x768).

## Remaining
- Final compilation and success criteria update
