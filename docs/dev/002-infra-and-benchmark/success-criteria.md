# Session 002: Infrastructure & A/B Benchmark — Success Criteria

## Done When

- [ ] Qwen3-Next-80B-A3B-Instruct Q8_0 GGUF downloaded to `models/`
- [ ] `docker build -f docker/Dockerfile.llama-cpp .` succeeds
- [ ] `docker build -f docker/Dockerfile.ik-llama-cpp .` succeeds
- [ ] `docker compose --profile llama-cpp up` starts server, `/health` returns OK
- [ ] `docker compose --profile ik-llama up` starts server, `/health` returns OK
- [ ] `llama-bench` runs inside both containers (pp512, tg128 minimum)
- [ ] `scripts/bench.sh llama-cpp baseline` completes all 4 workloads
- [ ] `scripts/bench.sh ik-llama baseline` completes all 4 workloads
- [ ] `scripts/bench.sh llama-cpp optimized` completes with KV cache quant
- [ ] `scripts/bench.sh ik-llama optimized` completes with ik-specific flags
- [ ] `scripts/compare-results.py` produces comparison table
- [ ] Winner documented in `handoff.md` with specific flags and rationale
- [ ] `docker-compose.yml` updated with winning config as default
- [ ] `CLAUDE.md` updated with production launch command

## Test Matrix

| ID | Engine | Config | Description |
|----|--------|--------|-------------|
| A1 | llama.cpp | baseline | Standard MoE offload, -t 16 |
| A2 | llama.cpp | optimized | + KV cache q8_0, thread sweep |
| B1 | ik_llama.cpp | baseline | Same as A1, no ik flags |
| B2 | ik_llama.cpp | optimized | + --merge-qkv -gr -fmoe 1 -rtr 1 |

## Workloads

| Workload | Prompt | Gen | Runs | Measures |
|----------|--------|-----|------|----------|
| short_prompt | ~128 tok | 256 | 5 | Generation speed |
| medium_prompt | ~1024 tok | 256 | 5 | Balanced |
| long_prompt | ~4096 tok | 256 | 5 | Prefill speed |
| multi_turn | 5×256 tok | 256/turn | 5 | KV cache reuse |
