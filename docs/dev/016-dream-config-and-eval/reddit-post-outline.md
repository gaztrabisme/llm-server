# Reddit Post Outline — S016 Phase 5

**Title**: "RTX 5080 16GB: Qwen3.6 27B IQ3 vs 35B MoE — both with mainline MTP, full benchmarks"

**Promised to**: WarthogConfident4039

## Structure

### 1. TL;DR Table (top of post)
| | 27B IQ3+MTP | 35B Q4_K_XL+MTP | 35B Q8_0+MTP |
|---|---|---|---|
| Model size | 12.45 GB | ~22 GB | ~36 GB |
| Fits on GPU | Fully (66/66) | Partial offload | Heavy offload |
| TG (tok/s) | 74 avg, 83 peak | TBD | TBD |
| MTP accept | 74% | TBD | TBD |
| Max context (q8_0 KV) | 56k | TBD | TBD |
| Max context (q4_0 KV) | 110k | TBD | TBD |
| CodeNeedle (220 lines) | 220/220 | TBD | TBD |
| GSM8K (100 cases) | ~90% | TBD | TBD |
| Best for | Coding agents, long context | TBD | TBD |

### 2. Setup
- RTX 5080 16GB, Ryzen 9 9950X, 128GB RAM
- llama.cpp mainline b9204 (MTP merged May 16)
- All configs: `-np 1 --fit on --fit-target 0 -fa on -t 20 --no-mmap -ctk q8_0 -ctv q8_0 --spec-type draft-mtp --spec-draft-n-max 2`
- **Critical finding**: `--spec-type draft-mtp` (not `mtp`), must use `-np 1` (default 4 slots pushes layers to CPU)

### 3. Speed (T1: mtp-bench)
- Per-prompt breakdown table
- Compare mainline vs am17an fork numbers (74 vs 76 tok/s)
- Why acceptance rate differs (74% vs 90.6%) but throughput is similar

### 4. Context Limits (T2: context push)
- Community asked about 128k for coding agents
- 27B: q8_0 KV → 56k, q4_0 KV → 110k (near 128k!)
- 35B: TBD
- OOM is MTP compute buffer (529 MiB), not KV cache
- Trade-off: q4_0 KV is near-lossless (Phase 2a: 218/220 vs 220/220 at 56k)

### 5. Quality (T4: CodeNeedle + T5: GSM8K)
- CodeNeedle: positional recall in 50k char source file
- GSM8K: reasoning baseline
- GPQA caveat: 91.6% non-truncated but likely contaminated, NOT cited as capability

### 6. Ubatch PP Trick (T3, partial offload only)
- Credit coder543 (May 18): `-ub 8192` gives 5.5x PP on partial offload
- Our results for 35B configs
- No effect on 27B (fully on GPU)

### 7. Recommendation
- "Which should you run?"
- 27B IQ3: if you want speed + context (coding agents, long docs)
- 35B MoE: if you want broader knowledge (more params active per token is only 3B, but the routing covers 35B total)
- 35B Q8_0: if you want max quality and can accept slower speed

### 8. Credits
- janvitos — 80 tok/s MTP config, 635 upvotes
- coder543 — ubatch PP trick (May 18)
- havenoammo — MTP GGUF variants + graft script
- am17an — original MTP implementation (PR #22673)
- ggerganov — llama-eval, MTP mainline merge
- Still-Notice8155 — GTX 1070 MTP benchmarks
- raketenkater — run-time-repack, defrag-thold concepts
- WarthogConfident4039 — requested this round
- moflinCASIO — 4060 Ti benchmarks
- OsmanthusBloom — earlier ubatch discovery

### 9. Methodology Notes
- All benchmarks reproducible: configs in `configs/llama-cpp-s016-*.env`
- graft-mtp.py available for building your own MTP GGUFs
- GSM8K sampling: temp=1.0, top_k=20, top_p=0.95 (Qwen3.5 model card thinking defaults)
- MTP compute buffer is the context bottleneck, not KV — this may improve in future llama.cpp versions

### 10. What's Next
- Session 017: vLLM vs llama.cpp head-to-head (MTP, TurboQuant, PagedAttention)
- TurboQuant KV when mainline support lands
- Expert caching when available
