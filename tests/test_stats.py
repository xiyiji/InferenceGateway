from bench.stats import summarize


def test_summarize_basic():
    rows = [{"e2e_s": 0.5, "ttft_s": 0.1, "tokens": 100, "status": 200}] * 10 + \
           [{"e2e_s": 2.0, "ttft_s": None, "tokens": 0, "status": 503}]
    s = summarize(rows, wall_s=5.0, concurrency=8)
    assert s["requests"] == 11 and s["errors"] == 1
    assert s["tokens_per_s"] == 200.0
    assert s["e2e_p50_ms"] == 500.0
    assert s["ttft_p95_ms"] == 100.0
    assert s["goodput_rps"] == 2.0


def test_summarize_empty():
    s = summarize([], wall_s=0.0, concurrency=1)
    assert s["tokens_per_s"] == 0.0 and s["e2e_p50_ms"] is None
