"""Pure functions so they can be unit-tested without a GPU."""
import numpy as np


def pct(xs, q):
    xs = [x for x in xs if x is not None]
    return float(np.percentile(xs, q)) if xs else None


def summarize(rows: list[dict], wall_s: float, concurrency: int) -> dict:
    ok = [r for r in rows if r["status"] == 200]
    e2e = [r["e2e_s"] * 1000 for r in ok]
    ttft = [r["ttft_s"] * 1000 for r in ok if r["ttft_s"] is not None]
    tokens = sum(r["tokens"] for r in ok)
    return {
        "concurrency": concurrency,
        "requests": len(rows),
        "errors": len(rows) - len(ok),
        "wall_s": round(wall_s, 2),
        "tokens_per_s": round(tokens / wall_s, 1) if wall_s else 0.0,
        "req_per_s": round(len(ok) / wall_s, 2) if wall_s else 0.0,
        "e2e_p50_ms": pct(e2e, 50),
        "e2e_p95_ms": pct(e2e, 95),
        "e2e_p99_ms": pct(e2e, 99),
        "ttft_p50_ms": pct(ttft, 50),
        "ttft_p95_ms": pct(ttft, 95),
        "goodput_rps": round(sum(1 for x in e2e if x < 1000) / wall_s, 2) if wall_s else 0.0,
    }
