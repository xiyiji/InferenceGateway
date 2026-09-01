"""Checks the request/response shape without loading vLLM or Ray."""
import sys
import types

# stub ray.serve so serve.app imports on a laptop
ray = types.ModuleType("ray"); rs = types.ModuleType("ray.serve")
rs.deployment = lambda **k: (lambda c: c); rs.ingress = lambda a: (lambda c: c)
ray.serve = rs; sys.modules.setdefault("ray", ray); sys.modules.setdefault("ray.serve", rs)

from serve.app import ChatRequest  # noqa: E402


def test_request_defaults():
    r = ChatRequest(messages=[{"role": "user", "content": "hi"}])
    assert r.max_tokens == 256 and r.stream is False and r.messages[0].role == "user"
