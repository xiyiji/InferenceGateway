# InferenceGateway

High-concurrency, low-latency LLM serving on a single GPU — **vLLM + Ray Serve + continuous batching + GPU observability**, benchmarked against a plain Hugging Face baseline.

## What it demonstrates

| Concern | How it's handled |
|---|---|
| Throughput | vLLM continuous batching + paged KV cache |
| Latency tail | Request queue with timeouts and backpressure in Ray Serve |
| Scale-out | Ray Serve replicas + autoscaling on queue depth |
| Memory | `gpu_memory_utilization`, FP8 / AWQ quantization experiments |
| Observability | Prometheus (vLLM metrics + DCGM) → Grafana dashboard |
| Evidence | Reproducible benchmark harness: p50/p95/p99 latency, TTFT, TPOT, tokens/s |

## Architecture

```
client (bench/) ──HTTP──▶ Ray Serve ingress (FastAPI, OpenAI-compatible /v1/chat/completions)
                              │  queue · timeout · backpressure · autoscale
                              ▼
                        vLLM engine replica(s)   ──▶ GPU
                              │
        Prometheus ◀── /metrics (vLLM) + DCGM exporter (GPU)
              │
           Grafana  (TTFT · TPOT · tokens/s · KV-cache usage · GPU util · queue depth)
```

## Quickstart

```bash
# 1. GPU box (A10 24GB / A100 40GB / L4). Tested on RunPod + Lambda.
pip install -r requirements.txt

# 2. Baseline (HF Transformers, no batching)
python serve/baseline_hf.py --model Qwen/Qwen2.5-7B-Instruct --port 8001

# 3. vLLM + Ray Serve gateway
serve run serve.app:deployment --model Qwen/Qwen2.5-7B-Instruct --port 8000

# 4. Monitoring
docker compose -f monitoring/docker-compose.yml up -d   # prometheus:9090 grafana:3000

# 5. Benchmark both
python bench/run_bench.py --url http://localhost:8001 --concurrency 1 4 16 --tag hf
python bench/run_bench.py --url http://localhost:8000 --concurrency 1 16 64 128 300 --tag vllm
python bench/report.py results/   # -> results/summary.md + plots
```

## Results

> Fill in after running. Keep the numbers; they are the resume.

| Config | Concurrency | tokens/s | p50 ms | p95 ms | TTFT p95 ms | GPU util |
|---|---|---|---|---|---|---|
| HF baseline | 1 | | | | | |
| HF baseline | 16 | | | | | |
| vLLM | 64 | | | | | |
| vLLM + FP8 | 128 | | | | | |
| vLLM + Ray 2 replicas | 300 | | | | | |

Grafana screenshots: `docs/grafana-*.png`

## Experiments (see SPEC.md §5)

1. HF vs vLLM at equal concurrency
2. `max_num_seqs` sweep — throughput vs p95
3. `gpu_memory_utilization` sweep — KV cache headroom vs OOM
4. FP8 / AWQ vs BF16 — quality (few-shot eval) vs speed
5. Ray Serve autoscaling under a bursty load profile

## Repo layout

```
serve/        Ray Serve + vLLM deployment, HF baseline server
bench/        load generator, metrics collection, report/plots
monitoring/   prometheus.yml, DCGM exporter, Grafana dashboard JSON
tests/        API contract + benchmark-harness unit tests (run without a GPU)
docs/         screenshots, result tables
SPEC.md       goals, metrics definitions, milestones, acceptance criteria
```

## Tests

```bash
pytest tests/          # no GPU needed: mocks the engine, checks API contract + bench math
```
