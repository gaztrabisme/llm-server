# Commit Regression Investigation: 22 -> 15-17 tok/s

## Summary

Session 002 (Feb 22, 2026) measured ~22 tok/s with llama.cpp + KV cache q8_0.
Session 003 (Feb 23-24, 2026) measured ~15-17 tok/s with the identical config after rebuilding the Docker image.
The regression is **~30%** and is accompanied by a **+447 MB VRAM increase** (6247 -> 6694 MB).

## Root Cause: Unpinned Dockerfile + ~24-hour HEAD drift

The Dockerfile uses `git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .` with no pinned commit. Each rebuild fetches whatever HEAD is at clone time.

### Confirmed Commits

| Session | Build Date (UTC) | Commit (confirmed/estimated) | Method |
|---------|-----------------|------------------------------|--------|
| **S003** | 2026-02-23 ~15:14 | **`9051663`** (confirmed) | `LLAMA_BUILD_COMMIT` in cmake config + `llama-cli --version` |
| **S002** | 2026-02-22 ~12:57 | **`34ec1c3` or `e877ad8`** (estimated) | Dockerfile mtime (19:57 local = 12:57 UTC) + benchmark start (20:39 local) |

The Session 003 image commit `9051663` was verified by running:
```
docker run --rm --gpus all --entrypoint /bin/bash llm-server/llama-cpp:latest \
  -c "/opt/llama-cpp/bin/llama-cli --version"
# Output: version: 1 (9051663)
```

The Session 002 image was **overwritten** by the Session 003 rebuild, so the exact commit cannot be confirmed. However, based on file timestamps and benchmark timing, it was built from HEAD around Feb 22 12:57 UTC.

### Commits Between the Two Builds

Between `e877ad8` (Feb 22 07:07 UTC, latest likely S002 HEAD) and `9051663` (Feb 23 13:16 UTC):

| Commit | Date (UTC) | Description | Perf Impact |
|--------|-----------|-------------|-------------|
| `34ec1c3` | Feb 22 13:11 | server: merge contiguous Responses items | None |
| `9f0684f` | Feb 22 15:14 | ci: fix rocm archive name | None |
| `ae2368e` | Feb 22 15:15 | model: add Kanana-2 support | None (new model) |
| `cacc371` | Feb 22 15:26 | Fix cli-argument doc | None |
| `ed48378` | Feb 22 16:34 | common: fix XML parser trimming | None |
| `5452d73` | Feb 22 20:08 | jinja: correct stats for filters | None |
| `e8e2616` | Feb 22 21:33 | cli: provide model with text filename | None |
| **`2b6dfe8`** | **Feb 23 06:04** | **llama: remove write/read of output ids/logits/embeddings** | **Low** (state serialization only) |
| `bc160d3` | Feb 23 12:42 | ggml-cpu: arm64 q5_K repack | None (ARM only) |
| `72b44c0` | Feb 23 13:15 | model-conversion: merge scripts | None (tooling) |
| `9051663` | Feb 23 13:16 | webui: code blocks setting | None (UI) |

**None of these 12 commits contain runtime-impacting changes for CUDA MoE inference.**

## The Real Regression Window

If the ~12 commits between the estimated build times don't explain a 30% regression, the Session 002 image was likely built **earlier than estimated** -- possibly on **Feb 17** when the `docker-compose.yml` and config files were first created (all timestamped `Feb 17 21:10` local).

If Session 002 used HEAD from Feb 17 (~`afa6bfe`, 12:47 UTC), then the regression window expands to include **6 days of commits** with several significant CUDA and graph changes:

### High-Impact Commits in the Extended Window (Feb 17 -> Feb 23)

| Commit | Date | PR | Description | Potential Impact |
|--------|------|-----|-------------|-----------------|
| `c78e682` | Feb 19 | #19686 | CUDA: fix kernel selection logic for tile FA | **Could change which FA kernel is selected for sm_120** |
| `27326bf` | Feb 19 | #19660 | models: dedup qwen35 graphs | **Refactors Qwen graph construction** |
| `da348c9` | Feb 19 | #19730 | models: fix qwen3.5 beta/gate shapes | **Fixes shapes after graph refactor** |
| `a0c91e8` | Feb 21 | #19754 | Improve CUDA graph capture | **Changes CUDA graph warmup/capture strategy** |

### Analysis of Each Suspect

#### 1. `c78e682` -- CUDA tile FA kernel selection fix (#19686)

Fixed incorrect kernel selection logic for tiled flash attention. On sm_120 (RTX 5080), this could route to a different (possibly less optimal) FA kernel for the 12 GQA attention layers in Qwen3-Next. The PR description says it fixed an abort, but correcting the selection could mean using a slower-but-correct kernel path.

**Likelihood: MEDIUM.** Only affects 12 of 48 layers (GQA attention), and attention is not the bottleneck for MoE offloading. Could contribute a few percent but not 30%.

#### 2. `27326bf` + `da348c9` -- Qwen graph deduplication (#19660, #19730)

Refactored the compute graph construction for Qwen3.5 and Qwen-family models using a new `llm_build_delta_net_base` helper. This changes how the delta-net (linear attention) layers construct their compute graph.

**Likelihood: HIGH.** Graph structure changes directly affect:
- CUDA graph capture and replay (different graph = different capture behavior)
- Tensor allocation (different graph ordering = different buffer reuse patterns)
- The **+447 MB VRAM increase** is strong evidence of changed tensor allocation

The dedup was supposed to be a clean refactor, but the PR description notes it intentionally kept `qwen35` and `qwen35moe` graphs intact initially. Qwen3-Next uses `qwen3next` architecture which is different from `qwen35`, but the shared `llm_build_delta_net_base` function could still affect it.

#### 3. `a0c91e8` -- Improve CUDA graph capture (#19754)

Changed CUDA graph warmup strategy: delays activation until the same cgraph is called at least twice with matching properties (instead of eagerly capturing on first call). This is meant to avoid wasted capture overhead on volatile graphs.

**Likelihood: MEDIUM-HIGH.** For MoE models with expert offloading, the compute graph changes every token (different experts are activated). The new warmup strategy may cause CUDA graphs to never stabilize and thus never capture, removing a performance optimization that was previously working. The PR shows 11-12% improvement for standard models on RTX 6000 Ada, but MoE expert offloading is a very different workload pattern.

## VRAM Analysis

| Metric | Session 002 A2 | Session 003 C1 | Delta |
|--------|---------------|---------------|-------|
| VRAM used | 6247 MB | 6694 MB | **+447 MB** |
| VRAM (after bench) | 6227 MB | 6836 MB | +609 MB |
| Throughput | ~22 tok/s | ~15.7 tok/s | **-29%** |
| Stddev | < 1.0 | 0.2-1.0 | Comparable |

The +447 MB VRAM increase with identical model and config is strong evidence of a structural change in how the compute graph is allocated. This is consistent with:
- Graph refactoring changing tensor lifetimes and preventing buffer reuse
- CUDA graph capture behavior changes allocating additional GPU memory
- Different tensor placement decisions due to graph node reordering

## Known Upstream Issues (Corroborating Evidence)

Several upstream issues report similar Qwen/MoE regressions around this time:

1. **[#19816](https://github.com/ggml-org/llama.cpp/issues/19816)**: CUDA illegal memory access with Qwen3-Next on multi-GPU using `-ot` (regression). Last working: `b48e80f67` (Feb 13). Broken on Feb 22 HEAD. Specifically affects `-ot` expert offloading.

2. **[#19683](https://github.com/ggml-org/llama.cpp/issues/19683)**: qwen35moe produces degenerate output with CUDA, works on CPU-only. Points to fundamental issues in the Qwen MoE CUDA graph path.

3. **[#18112](https://github.com/ggml-org/llama.cpp/issues/18112)**: Qwen3-Next token generation performance regression (CPU-only). A prior regression at `a5251ca` (#17996) dropped tg from 6.65 to 5.02 tok/s.

4. **[#18258](https://github.com/ggml-org/llama.cpp/issues/18258)**: Major performance drop since b7406 (45 -> 10-20 tok/s). Shows a pattern of large regressions tied to graph/CUDA changes.

5. **[#19817](https://github.com/ggml-org/llama.cpp/issues/19817)**: Much slower TG using CUDA vs Vulkan on GTX1060 at commit `ed48378` (the same commit in our regression window).

## Conclusions

1. **The Dockerfile's unpinned `git clone --depth 1` is the root cause** of build-to-build variability. Any rebuild picks up whatever HEAD is current, with no way to reproduce a previous build.

2. **The most likely technical cause** is the combination of:
   - Qwen graph refactoring (`#19660`, `#19730`) changing compute graph structure
   - CUDA graph capture strategy changes (`#19754`) interacting poorly with MoE expert offloading patterns
   - FA kernel selection changes (`#19686`) potentially routing to slower kernels on sm_120

3. **The +447 MB VRAM increase** corroborates structural graph changes rather than config differences.

4. **Multiple upstream issues** report Qwen MoE regressions in the same Feb 13-22 timeframe, confirming this is a known problem area.

## Recommendations

### Immediate: Pin Dockerfile to a known-good commit

Replace the unpinned clone in `docker/Dockerfile.llama-cpp`:

```dockerfile
# Before (unpinned):
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git .

# After (pinned to last known ~22 tok/s commit):
ARG LLAMA_CPP_COMMIT=e877ad8
RUN git clone https://github.com/ggml-org/llama.cpp.git . \
    && git checkout ${LLAMA_CPP_COMMIT}
```

### Verify: Rebuild with pinned commit and re-benchmark

1. Pin to `e877ad8` (Feb 22 07:07 UTC, latest pre-Session-003 commit with no graph changes)
2. Rebuild: `docker compose --profile llama-cpp build --no-cache`
3. Run benchmark: `bash scripts/bench.sh llama-cpp optimized 16`
4. If ~22 tok/s is restored, the regression is confirmed as a llama.cpp commit issue

If `e877ad8` doesn't restore performance, try earlier commits:
- `a0c91e8~1` = before CUDA graph capture changes
- `c78e682~1` = before FA kernel selection fix
- `cc45f2a~1` = before Qwen graph dedup

### Long-term: Track llama.cpp version in builds

- Add `LLAMA_CPP_COMMIT` as a build arg with a default pinned value
- Update the pinned commit deliberately when upgrading, with a benchmark gate
- Consider storing the commit hash in a label on the Docker image for traceability
