Panels to build (export the dashboard JSON into this folder once done):

- TTFT p95: `histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[30s]))`
- TPOT p95: `histogram_quantile(0.95, rate(vllm:time_per_output_token_seconds_bucket[30s]))`
- tokens/s: `rate(vllm:generation_tokens_total[30s])`
- running / waiting: `vllm:num_requests_running`, `vllm:num_requests_waiting`
- KV cache %: `vllm:gpu_cache_usage_perc`
- GPU util: `DCGM_FI_DEV_GPU_UTIL`
- GPU mem: `DCGM_FI_DEV_FB_USED`
