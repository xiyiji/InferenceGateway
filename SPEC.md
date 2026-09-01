# InferenceGateway — SPEC

## 1. Goal

Build and measure a production-shaped LLM inference service on one GPU. The deliverable is not the server — it is the **evidence**: a reproducible benchmark showing how continuous batching, quantization, and queue policy change throughput and tail latency, plus live GPU observability.

Interview framing: *"Measure first, then tune. Every claim in the README maps to a JSON file in `results/`."*

## 2. Non-goals

- Training or fine-tuning
- Multi-node / multi-GPU tensor parallelism (documented as "next step")
- Custom CUDA kernels — vLLM is used as-is; the value is in system integration + measurement

## 3. Components

| Component | Tech | Notes |
|---|---|---|
| Ingress | Ray Serve + FastAPI | OpenAI-compatible `/v1/chat/completions`, streaming on |
| Engine | vLLM `AsyncLLMEngine` | one replica per GPU; `enable_prefix_caching=True` |
| Queue policy | Ray Serve `max_ongoing_requests`, request timeout, 429 on overflow | backpressure instead of unbounded queue |
| Autoscale | Ray Serve autoscaling on `target_ongoing_requests` | demo with 2 replicas on a 2-GPU box, or same GPU with small models |
| Baseline | HF Transformers `generate()` behind FastAPI | one request at a time → shows why batching matters |
| Metrics | vLLM `/metrics` + NVIDIA DCGM exporter → Prometheus → Grafana | dashboard JSON committed |
| Load gen | `bench/run_bench.py` (asyncio + httpx) | ShareGPT-style prompt mix, fixed seed |

## 4. Metrics (definitions — put these in the README verbatim)

- **TTFT** — time to first token (ms). Prefill cost + queue wait.
- **TPOT** — time per output token (ms). Decode steady-state.
- **E2E latency** — request start → last token. Report p50 / p95 / p99.
- **Throughput** — output tokens/s across all concurrent requests.
- **Goodput** — requests/s that meet an SLO (p95 E2E < 1 s for 256-token outputs).
- **GPU util / KV cache %** — from DCGM and vLLM gauges, sampled at 1 s.

## 5. Experiments

| # | Variable | Fixed | Output |
|---|---|---|---|
| E1 | HF vs vLLM | model, prompts, concurrency ∈ {1, 4, 16} | throughput multiplier, p95 |
| E2 | `max_num_seqs` ∈ {32, 64, 128, 256} | vLLM BF16 | throughput vs p95 curve — find the knee |
| E3 | `gpu_memory_utilization` ∈ {0.7, 0.85, 0.95} | E2 best | KV cache blocks vs OOM/preemption count |
| E4 | dtype ∈ {BF16, FP8, AWQ-int4} | E2 best | speed + 5-task quality check (MMLU-subset / GSM8K-100) |
| E5 | bursty load (0→300→0 conc. in 60 s) | autoscale on/off | queue depth, 429 rate, p95 during burst |

## 6. Milestones

| Day | Deliverable | Done when |
|---|---|---|
| 1 | GPU box up, HF baseline serving, bench harness runs | `results/hf_c1.json` exists |
| 2 | vLLM + Ray Serve gateway, streaming works | `curl` chat completion returns tokens |
| 3 | Prometheus + DCGM + Grafana; dashboard JSON committed | screenshot in `docs/` |
| 4 | E1 + E2 complete | `results/summary.md` has both tables |
| 5 | E3 + E4 complete | quantization table with quality column |
| 6 | E5 + backpressure/timeout behavior | burst plot in `docs/` |
| 7 | README results filled, tests green, resume bullets drafted | `pytest` passes; README has real numbers |

## 7. Acceptance criteria

- One command reproduces every table (`make bench`)
- Every number in README maps to a JSON file in `results/`
- Tests run without a GPU (engine mocked)
- Dashboard JSON imports cleanly into a fresh Grafana
- README states hardware, model, commit hash, and vLLM version for each result

## 8. Resume bullet template (fill from results)

> Built an OpenAI-compatible LLM inference gateway (Ray Serve + vLLM) serving a 7B model on a single A100; continuous batching and FP8 quantization raised throughput **__×** over an HF baseline while holding p95 latency under **__ ms** at **__** concurrent requests. Instrumented with Prometheus/DCGM/Grafana; added queue timeouts and backpressure that cut burst-period p99 by **__%**.

## 9. Interview talking points to prepare

- Why continuous batching beats static batching (decode steps have variable length)
- What paged KV cache solves (fragmentation) and its cost (block table lookups)
- Where TTFT comes from vs where TPOT comes from; which knob moves which
- Why unbounded queues are worse than 429s under overload
- FP8 vs int4: when quality drops, how you measured it
