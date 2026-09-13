"""Ray Serve + vLLM gateway. OpenAI-compatible chat completions with streaming.

Run:  serve run serve.app:deployment --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import inspect
import json
import os
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
        "max_replicas": 1,
        "target_ongoing_requests": 64,
    },
)
@serve.ingress(app)
class Gateway:
    def __init__(self, model: str, dtype: str, max_num_seqs: int, gpu_mem: float, quantization: str | None,
                 max_num_batched_tokens: int = 4096, max_model_len: int = 4096):
        from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams  # import inside the actor

        self.SamplingParams = SamplingParams
        self.engine = AsyncLLMEngine.from_engine_args(
            AsyncEngineArgs(
                model=model,
                dtype=dtype,
                max_num_seqs=max_num_seqs,
                max_num_batched_tokens=max_num_batched_tokens,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_mem,
                quantization=quantization,
                enable_prefix_caching=True,
            )
        )
        self.tokenizer = None

    async def _prompt(self, messages: list[ChatMessage]) -> str:
        if self.tokenizer is None:
            tokenizer = self.engine.get_tokenizer()
            # Support both synchronous and awaitable tokenizer contracts.
            if inspect.isawaitable(tokenizer):
                tokenizer = await tokenizer
            self.tokenizer = tokenizer
        return self.tokenizer.apply_chat_template(
            [m.model_dump() for m in messages], tokenize=False, add_generation_prompt=True
        )

    async def _generate(self, req: ChatRequest, rid: str) -> AsyncIterator[dict]:
        params = self.SamplingParams(max_tokens=req.max_tokens, temperature=req.temperature)
        prompt = await self._prompt(req.messages)
        prev = ""
        finished = False
        try:
            async for out in self.engine.generate(prompt, params, rid):
                completion = out.outputs[0]
                text = completion.text
                delta, prev = text[len(prev):], text
                finished = out.finished
                prompt_tokens = len(out.prompt_token_ids)
                completion_tokens = len(completion.token_ids)
                yield {"delta": delta, "finish_reason": completion.finish_reason,
                       "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                                 "total_tokens": prompt_tokens + completion_tokens}}
        finally:
            if not finished:
                aborted = self.engine.abort(rid)
                if inspect.isawaitable(aborted):
                    await aborted

    @app.post("/v1/chat/completions")
    async def chat(self, req: ChatRequest):
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        t0 = time.perf_counter()

        if req.stream:
            async def sse():
                usage = None
                async for item in self._generate(req, rid):
                    usage = item["usage"]
                    chunk = {"id": rid, "object": "chat.completion.chunk",
                             "choices": [{"delta": {"content": item["delta"]}, "index": 0,
                                          "finish_reason": item["finish_reason"]}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                if usage is not None:
                    yield f'data: {json.dumps({"id": rid, "choices": [], "usage": usage})}\n\n'
                yield "data: [DONE]\n\n"
            return StreamingResponse(sse(), media_type="text/event-stream")

        text = ""
        last = None
        async for item in self._generate(req, rid):
            text += item["delta"]
            last = item
        if last is None:
            raise RuntimeError("Engine returned no output")
        return {
            "id": rid,
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": last["finish_reason"]}],
            "usage": last["usage"],
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    @app.get("/healthz")
    async def health(self):
        return {"ok": True}


def deployment(args: dict | None = None):
    args = args or {}
    return Gateway.bind(
        model=args.get("model", "Qwen/Qwen2.5-7B-Instruct"),
        dtype=args.get("dtype", "bfloat16"),
        max_num_seqs=int(args.get("max_num_seqs", os.getenv("VLLM_MAX_NUM_SEQS", "128"))),
        max_num_batched_tokens=int(args.get("max_num_batched_tokens", os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "4096"))),
        max_model_len=int(args.get("max_model_len", os.getenv("VLLM_MAX_MODEL_LEN", "4096"))),
        gpu_mem=float(args.get("gpu_memory_utilization", 0.9)),
        quantization=args.get("quantization"),
    )
