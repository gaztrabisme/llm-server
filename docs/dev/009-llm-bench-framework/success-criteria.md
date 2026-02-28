# Session 009: `llm-bench` — Reusable Benchmark Framework

## Done When

- [x] `pip install -e .` works in a fresh venv
- [x] `llm-bench --help` shows all 6 subcommands (setup, speed, quality, eval, compare, report)
- [x] `llm-bench setup` detects RTX 5080, generates valid llm-bench.yaml
- [x] `llm-bench speed --dry-run` produces correct Docker + API commands
- [x] `llm-bench speed --env configs/llama-cpp-s006-e4-fit-nobatch.env --dry-run` works (backward compat)
- [x] `llm-bench compare --dir benchmarks/matrix/` reads all 32 existing result files
- [x] `llm-bench compare` supports table/csv/json/markdown formats, --group-by, --filter, --sort-by
- [x] `llm-bench quality --ppl-only --dry-run` produces correct Docker command
- [x] `llm-bench eval --tasks gsm8k --limit 20 --dry-run` produces correct lm-eval command
- [x] `llm-bench report` generates readable markdown from existing results
- [x] Old scripts NOT deleted, deprecation comments added
- [x] Session docs created

## Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Foundation (package skeleton + core modules) | Complete |
| 2 | Speed command | Complete |
| 3 | Quality command | Complete |
| 4 | Compare + Report commands | Complete |
| 5 | Setup + Eval commands | Complete |
| 6 | Validation (dry-run + compare against existing data) | Complete |
| 7 | Documentation + Cleanup | Complete |

## Delivered Files

### New files (24)
| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, entry point, dependencies |
| `llm-bench.example.yaml` | Documented example config |
| `llm_bench/__init__.py` | Version string |
| `llm_bench/__main__.py` | `python -m llm_bench` support |
| `llm_bench/cli.py` | Click group + subcommand registration |
| `llm_bench/commands/__init__.py` | Package init |
| `llm_bench/commands/setup.py` | Hardware detection, config generation |
| `llm_bench/commands/speed.py` | PP/TG benchmarks (ports bench-matrix.sh) |
| `llm_bench/commands/quality.py` | PPL/KLD evaluation (ports quant-quality.sh) |
| `llm_bench/commands/eval.py` | lm-eval-harness wrapper (ports run-eval-suite.sh) |
| `llm_bench/commands/compare.py` | Result comparison (ports compare-matrix.py) |
| `llm_bench/commands/report.py` | Markdown report generation |
| `llm_bench/core/__init__.py` | Package init |
| `llm_bench/core/config.py` | YAML config, defaults, CLI overrides, flag conversion |
| `llm_bench/core/docker.py` | Docker lifecycle, health check, signal handlers |
| `llm_bench/core/gpu.py` | GPU detection, VRAM capture via nvidia-smi |
| `llm_bench/core/hardware.py` | CPU/RAM detection from /proc |
| `llm_bench/core/stats.py` | Mean/stddev with warmup skip |
| `llm_bench/core/results.py` | Result JSON I/O, backward-compatible schema |
| `llm_bench/compat/__init__.py` | Package init |
| `llm_bench/compat/env_parser.py` | Parse legacy LLAMA_ARG_* .env files |
| `llm_bench/compat/legacy_results.py` | Normalize old bench.sh JSON format |
| `llm_bench/workloads/__init__.py` | Package init |
| `llm_bench/workloads/defaults.py` | Default TG prompts (short/medium/long/multi-turn) |
| `llm_bench/workloads/loader.py` | Custom workload YAML loading |

### Modified files (5)
| File | Change |
|------|--------|
| `.gitignore` | Added `*.egg-info/`, `llm-bench.yaml` |
| `scripts/bench-matrix.sh` | Added deprecation notice |
| `scripts/compare-matrix.py` | Added deprecation notice |
| `scripts/quant-quality.sh` | Added deprecation notice |
| `scripts/run-eval-suite.sh` | Added deprecation notice |

## Test Evidence

```
$ llm-bench --version
llm-bench, version 0.1.0

$ llm-bench --help
Commands: compare, eval, quality, report, setup, speed

$ llm-bench setup
GPU: NVIDIA GeForce RTX 5080 (16303 MB)
CPU: 32 cores (AMD Ryzen 9 9950X 16-Core Processor)
Models: 5 found, Docker: OK, NVIDIA Toolkit: OK

$ llm-bench compare --dir benchmarks/matrix/ --sort-by tg_mean
(32 results loaded, sorted by TG speed, winners marked)

$ llm-bench compare --filter s007-e2 --group-by batch
(3 groups: asym, large, none — correct grouping)

$ llm-bench speed --env configs/llama-cpp-s006-e4-fit-nobatch.env --dry-run
(Correct Docker command, API requests, workload list)

$ llm-bench quality --model models/Qwen3.5-35B-A3B-Q4_K_M.gguf --ppl-only --dry-run
(Correct llama-perplexity Docker command)

$ llm-bench eval --tasks gsm8k --limit 20 --dry-run --skip-server
(Correct lm-eval command)

$ llm-bench report --dir benchmarks/matrix/
(32 results, markdown report with hardware/speed/findings)
```
