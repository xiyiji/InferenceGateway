import asyncio
import httpx
from bench.run_bench import one


def measure(body):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(200, text=body))) as client:
            return await one(client, 'http://test', 128, True)
    return asyncio.run(run())


def test_stream_counts_usage_not_chunks():
    row = measure('data: {"choices":[{"delta":{"content":"many tokens"}}]}\n\n'
                  'data: {"choices":[],"usage":{"completion_tokens":7}}\n\n'
                  'data: [DONE]\n\n')
    assert row['tokens'] == 7
    assert row['status'] == 200


def test_truncated_stream_is_failure():
    row = measure('data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')
    assert row['status'] == 599
