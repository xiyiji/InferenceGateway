"""Ray Serve + vLLM gateway. OpenAI-compatible chat completions with streaming.

Run:  serve run serve.app:deployment --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ray import serve

app = FastAPI()


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    max_tokens: int = 256
    temperature: float = 0.7
    stream: bool = False


@serve.deployment(
    ray_actor_options={"num_gpus": 1},
    max_ongoing_requests=128,          # backpressure: beyond this Ray queues, then 503s
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 2,
        "target_ongoing_requests": 64,
    },
)
@serve.ingress(app)
class Gateway:
    def __init__(self, model: str, dtype: str, max_num_seqs: int, gpu_mem: float, quantization: str | None):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # import inside the actor

        self.SamplingParams = SamplingParams
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                dtype=dtype,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_mem,
                quantization=quantization,
                enable_prefix_caching=True,
            )
        )
        self.tokenizer = None

    async def _prompt(self, messages: list[ChatMessage]) -> str:
        if self.tokenizer is None:
            self.tokenizer = await self.engine.get_tokenizer()
        return self.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )

    async def _generate(self, req: ChatRequest, rid: str) -> AsyncIterator[str]:
        params = self.SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        prompt = await self._prompt(req.messages)
        prev = ""
        async for out in self.engine.generate(prompt, params, rid):
            text = out.outputs[0].text
            delta, prev = text[len(prev):], text
            if delta:
                yield delta

    @app.post("/v1/chat/completions")
    async def chat(self, req: ChatRequest):
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        if req.stream:
            async def sse():
                async for delta in self._generate(req, rid):
                    chunk = {"id": rid, "object": "chat.completion.chunk",
                             "choices": [{"delta": {"content": delta}, "index": 0}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text = "".join([d async for d in self._generate(req, rid)])
        return {
            "id": rid,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"latency_ms": round((time.perf_counter() - t0) * 1000, 1)},
        }

    @app.get("/healthz")
    async def health(self):
        return {"ok": True}


def deployment(args: dict | None = None):
    args = args or {}
    return Gateway.bind(
        model=args.get("model", "Qwen/Qwen2.5-7B-Instruct"),
        dtype=args.get("dtype", "bfloat16"),
        max_num_seqs=int(args.get("max_num_seqs", 128)),
        gpu_mem=float(args.get("gpu_memory_utilization", 0.9)),
        quantization=args.get("quantization"),
    )
