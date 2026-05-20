# Credits

This project's benchmarks, optimizations, and findings were shaped by the r/LocalLLaMA community. Every experiment tagged "community-requested" exists because someone asked a question, shared data, or challenged an assumption.

## Tooling and Models

| Who | Contribution | Sessions |
|-----|-------------|----------|
| **ggerganov** | Created llama.cpp. Merged Hadamard KV rotation (PR #21038), `llama-eval` (PR #21152). MTP mainline merge review (b9190, May 2026) | All |
| **danielhanchen** (Unsloth) | Unsloth Dynamic quants (UD-Q4_K_XL, IQ3_XXS, Q2_K_XL). Published 550+ GGUF variants with 120+ KLD evaluations. Confirmed and fixed UD-Q4_K_XL MXFP4 bug. Component ablation study identifying expert down projections as most sensitive | S003, S005-S008, S012, S015 |
| **am17an** | Original MTP implementation for llama.cpp (PR #22673), merged mainline at b9190. mtp-bench.py benchmark script. 99% draft acceptance on Qwen3.6 | S015, S016 |
| **havenoammo** | Published MTP GGUF variants (Q2_K_XL through Q8_0) for both 27B and 35B-A3B. Separate MTP head files enabling graft experiments. 35B MTP-UD-Q4_K_XL used as S016 production model | S015, S016 |
| **Indras-Mirror** | llama.cpp fork combining MTP + TurboQuant + RotorQuant. Enabled testing MTP+TurboQuant combos that no other build supported | S015 |
| **TheTom** | TurboQuant fork (turbo2/3/4, tbq4_0). KV cache quantization types at 4.25 and 3.125 bits | S015 |
| **AtomicBot** | Combined TurboQuant repo. Gemma4 MTP support | S015 |
| **bartowski** | GGUF quants (Q4_K_M, Q4_K_L) used as production baseline through S005-S011. Q4_K_L quality evaluation drove the "not worth speed penalty" finding | S005-S008, S011 |
| **AesSedai** | Q4_K_M quant with KLD 0.0095 — best quality Q4 quant measured. Confirmed bobaburger's claim independently | S008 |
| **alexziskind1** | CodeNeedle positional recall benchmark. Critical for discovering 27B IQ3_XXS's 99.5% accuracy advantage over 35B MoE | S015 |

## Community Research and Data

| Who | Contribution | Sessions |
|-----|-------------|----------|
| **thc1006** | Research on MTP being net-negative on MoE models (expert-union overhead). Confirmed experimentally in our E2/E3 results | S015 |
| **bobaburger** | Full quant comparison data on 5060 Ti 16GB (UD-Q4_K_M, AesSedai Q4_K_M, IQ3_S, MXFP4, UD-Q4_K_XL). Tipped us off to AesSedai's superior Q4_K_M | S008 |
| **maho_Yun** | Confirmed `--fit-target` only needed when CLIP/mmproj is loaded. Vision benchmarks on 5060 Ti. `--fit-target 1536` recommendation validated in S011 | S007, S011 |
| **Lucis_unbra** | Reported KV q8_0 causes 10 tok/s drop (75 to 65) on 3090+Windows with dense models. Led to the important caveat that KV q8_0 "free lunch" is MoE-specific | S007 |
| **WarthogConfident4039** | Ongoing tips and collaboration via Reddit DMs. Pointed us to `llama-eval` (PR #21152) on day of merge. Requested the Qwen3.6 benchmarking round that became S016 Phase 5 | S015, S016 |

| **janvitos** | 80 tok/s MTP config on 12GB RTX 4070 Super (635 upvotes). Documented `-fitt`, `--ctx-checkpoints`, `--mlock` flags for VRAM-constrained MTP setups | S016 |
| **coder543** | Ubatch PP trick: `-ub 8192` gives 5.5x prompt processing speedup for `--n-cpu-moe` partial offload. Mechanism: reduces PCIe transfer overhead per token during prefill | S016 |
| **Still-Notice8155** | GTX 1070 8GB MTP benchmarks with TurboQuant. CodeNeedle results at IQ4_XS. Proved MTP works even on ancient GPUs | S015, S016 |
| **raketenkater** | `--ai-tune` auto-tuner concept, `--run-time-repack`, `-khad` (Hadamard KV), `--defrag-thold 0.1` flags documentation | S016 |
| **moflinCASIO** | 4060 Ti 16GB benchmarks: IQ2_M ~81 tok/s, IQ3_XXS ~74 tok/s. Cross-GPU speed reference for the community | S016 |

## Community Feedback That Drove Experiments

| Who | What they asked/challenged | What it led to |
|-----|---------------------------|----------------|
| **JermMX5** | Challenged PPL-only methodology, cited "Accuracy is Not All You Need" paper | KLD analysis in S006 — revealed UD-Q4_K_XL gap is worse than PPL showed |
| **Chromix_** | Insight about batch flags eating VRAM under `--fit` | fit-nobatch config: 74.7 tok/s, new fastest in S006 |
| **Psyko38** | AMD RX 7900 XTX benchmarks (15.82 vs 37.74 tok/s with/without `--fit`) | Community notes documenting `--fit on` is CUDA-specific |
| **Corosus** | Vulkan benchmarks on 5070 Ti (13 vs 33 tok/s with `--fit`) | Confirmed `--fit on` broken on non-CUDA backends |
| **OsmanthusBloom** | ubatch 1024 sweet spot data on RTX 3060. PP speed matters for agentic workloads | Extended PP measurements, ubatch sweep in S008 |
| **CATLLM** | Asked about `--no-kv-offload` | Tested in S008: -63% TG, never use |
| **KeldenL** | "Whose Q4 quant are you using?" | Clarified Q4_K_M provenance, led to Q4_K_M identity investigation in S011 |
| **InternationalNebula7** | Why is Ollama 3x slower? (21.6 vs 70 tok/s on 5080) | Ollama architecture analysis and recommendations |
| **wisepal_app** | Pre-built binaries vs building from source for Blackwell | Documented Blackwell-specific build considerations |
| **danielhanchen** | Confirmed UD-Q4_K_XL bug, recommended MXFP4 | MXFP4 evaluation (max 52 tok/s, not 77 claimed). Fixed UD-Q4_K_XL became best quant |
| **ayylmaonade**, **jumpingcross** | MXFP4 testing requests | MXFP4 benchmark in S006 E7 |
| **guiopen**, **DonkeyBonked** | `--fit` tuning questions | fit-nobatch discovery |
| **mr_Owner**, **mdziekon**, **DHasselhoff77** | "Show PP speed beside TG" | PP-512 column added to compare output in S011 |
| **Danmoreng** | Vision + fit-target interaction | Vision mode evaluation in S007 E5 |

| **OsmanthusBloom** | Earlier ubatch discovery predating coder543's post. Multiple community posts documenting the PP speedup effect | S008, S016 |
| **CodProfessional3712** | 5060 Ti speed help thread — community consensus around llama.cpp direct over LM Studio drove community guidance | S016 |

## Speed Reports Across GPUs

Community members who reported speeds on their hardware, building a cross-GPU comparison dataset:

u/jslominski, u/jiegec, u/Corosus, u/DeedleDumbDee, u/Monad_Maya, u/l33t-Mt, u/kkb294, u/zmanning, u/Additional-Action566

## How to Get Credited

Run benchmarks, share data, ask hard questions, or challenge assumptions on r/LocalLLaMA. If your contribution leads to an experiment or finding, it gets documented here.
