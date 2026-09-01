# InferenceGateway

A production-shaped LLM inference engine on one GPU: Ray Serve ingress in
front of vLLM's `AsyncLLMEngine`, an HF Transformers baseline to measure it
against, and a benchmark harness that turns serving folklore — continuous
batching, quantization, queue policy — into reproducible numbers.

This repo is the **engine layer** of a two-repo stack:
[llm-serving-platform](https://github.com/xiyiji/llm-serving-platform) is
the **gateway layer** above it — cross-engine routing, request
micro-batching, prefix caching, canary releases and an ops console. Point
that platform's backend config at this server and the two form one serving
path: gateway → engine → GPU.

## Architecture

| Component | Tech | Role |
|---|---|---|
| Ingress | Ray Serve + FastAPI | OpenAI-compatible `/v1/chat/completions`, streaming |
| Engine | vLLM `AsyncLLMEngine` | continuous batching, `enable_prefix_caching=True` |
| Queue policy | `max_ongoing_requests` + timeouts | backpressure and 429s instead of an unbounded queue |
| Baseline | HF `generate()` behind FastAPI | one request at a time — the "why batching matters" control |
| Metrics | vLLM `/metrics` + DCGM → Prometheus → Grafana | GPU utilisation, KV-cache occupancy, latency |
| Load gen | `bench/run_bench.py` (asyncio + httpx) | fixed-seed prompt mix, concurrency sweeps |

## Metrics the harness reports

- **TTFT** — time to first token: prefill cost plus queue wait
- **TPOT** — time per output token: decode steady state
- **E2E latency** — p50 / p95 / p99, request start to last token
- **Throughput** — output tokens/s across all concurrent requests
- **Goodput** — requests/s meeting the SLO (p95 E2E < 1 s at 256-token outputs)
- **GPU util / KV-cache %** — sampled at 1 s from DCGM and vLLM gauges

## Experiment matrix

1. HF baseline vs vLLM at concurrency 1 / 4 / 16 — the batching multiplier
2. `max_num_seqs` sweep — find the knee of the throughput-vs-p95 curve
3. `gpu_memory_utilization` sweep — KV-cache blocks vs preemption
4. BF16 vs FP8 vs AWQ-int4 — speed with a quality check attached
5. Burst load (0→300→0 concurrency in 60 s) — queue depth, 429 rate, p95

Every table reproduces with `make bench`; each result records hardware,
model, commit hash and vLLM version. Tests run without a GPU (engine
mocked): `make test`.

## Run

```bash
pip install -r requirements.txt
python serve/app.py                      # vLLM gateway on :8000
python serve/baseline_hf.py              # HF baseline on :8001
make bench                               # sweep both, write results/
docker compose -f monitoring/docker-compose.yml up   # Prometheus + Grafana
```

See [SPEC.md](SPEC.md) for goals, non-goals and acceptance criteria.
