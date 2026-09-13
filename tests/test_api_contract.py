"""Checks the request/response shape without loading vLLM or Ray."""
import sys
import types

# stub ray.serve so serve.app imports on a laptop
ray = types.ModuleType("ray"); rs = types.ModuleType("ray.serve")
rs.deployment = lambda **k: (lambda c: c); rs.ingress = lambda a: (lambda c: c)
ray.serve = rs; sys.modules.setdefault("ray", ray); sys.modules.setdefault("ray.serve", rs)

from serve.app import ChatRequest  # noqa: E402
from serve.app import Gateway
import asyncio
import json


def test_request_defaults():
    r = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert r.max_tokens == 256 and r.stream is False and r.messages[0].role == "user"


def gateway_with_fake_engine():
    gateway = Gateway.__new__(Gateway)
    gateway.SamplingParams = lambda **kw: kw
    gateway.tokenizer = types.SimpleNamespace(apply_chat_template=lambda *a, **kw: "prompt")
    async def generate(*args):
        yield types.SimpleNamespace(outputs=[types.SimpleNamespace(text="hello world", token_ids=[1, 2, 3], finish_reason="length")],
                                    prompt_token_ids=[4, 5], finished=True)
    gateway.engine = types.SimpleNamespace(generate=generate)
    return gateway


def test_nonstream_exact_usage():
    response = asyncio.run(gateway_with_fake_engine().chat(ChatRequest(messages=[{"role":"user", "content":"hi"}])))
    assert response["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert response["choices"][0]["finish_reason"] == "length"


def test_stream_final_usage():
    async def collect():
        response = await gateway_with_fake_engine().chat(ChatRequest(messages=[{"role":"user", "content":"hi"}], stream=True))
        return [chunk async for chunk in response.body_iterator]
    chunks = asyncio.run(collect())
    assert chunks[-1] == "data: [DONE]\n\n"
    assert json.loads(chunks[-2][6:])["usage"]["completion_tokens"] == 3
