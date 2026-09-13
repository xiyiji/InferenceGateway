"""Async load generator. Records per-request TTFT, E2E, output tokens; writes results/<tag>_c<N>.json."""
import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import httpx

from bench.stats import summarize

PROMPTS = [
    "Explain continuous batching in LLM serving in three sentences.",
    "Write a Python function that merges two sorted lists.",
    "Summarize the causes of the 2008 financial crisis.",
    "What is the difference between TTFT and TPOT?",
    "Give me a 5-item packing list for a weekend hike.",
]


async def one(client: httpx.AsyncClient, url: str, max_tokens: int, stream: bool) -> dict:
    body = {"messages": [{"role": "user", "content": random.choice(PROMPTS)}],
            "max_tokens": max_tokens, "stream": stream}
    t0 = time.perf_counter()
    ttft, tokens, status = None, 0, 200
    try:
        if stream:
            done, usage = False, None
            async with client.stream("POST", f"{url}/v1/chat/completions", json=body) as r:
                status = r.status_code
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            done = True
                            continue
                        item = json.loads(payload)
                        if item.get("error"):
                            raise ValueError(item["error"])
                        content = any(c.get("delta", {}).get("content") for c in item.get("choices", []))
                        if content and ttft is None:
                            ttft = time.perf_counter() - t0
                        if item.get("usage") is not None:
                            usage = item["usage"]
            if not done or usage is None or ttft is None:
                raise ValueError("Incomplete stream or missing exact usage")
            tokens = usage["completion_tokens"]
        else:
            r = await client.post(f"{url}/v1/chat/completions", json=body)
            status = r.status_code
            j = r.json()
            r.raise_for_status()
            tokens = j["usage"]["completion_tokens"]
    except Exception:
        status = 599
    return {"ttft_s": ttft, "e2e_s": time.perf_counter() - t0, "tokens": tokens, "status": status}


async def run(url: str, concurrency: int, n: int, max_tokens: int, stream: bool) -> dict:
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=120) as client:
        async def guarded():
            async with sem:
                return await one(client, url, max_tokens, stream)
        t0 = time.perf_counter()
        rows = await asyncio.gather(*[guarded() for _ in range(n)])
        wall = time.perf_counter() - t0
    return summarize(rows, wall, concurrency)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--requests", type=int, default=200)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--stream", action="store_true")
    p.add_argument("--tag", required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    Path("results").mkdir(exist_ok=True)
    for c in a.concurrency:
        res = asyncio.run(run(a.url, c, a.requests, a.max_tokens, a.stream))
        out = Path("results") / f"{a.tag}_c{c}.json"
        out.write_text(json.dumps(res, indent=2))
        print(f"{a.tag:>8} c={c:<4} tok/s={res['tokens_per_s']:>8.1f} "
              f"p50={res['e2e_p50_ms'] or 0:>7.0f}ms p95={res['e2e_p95_ms'] or 0:>7.0f}ms errors={res['errors']}")
