"""HF Transformers baseline: one request at a time, no batching. Exists to be beaten."""
import argparse
import threading
import time

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()
lock = threading.Lock()


class Req(BaseModel):
    messages: list[dict]
    max_tokens: int = 256
    temperature: float = 0.7


@app.post("/v1/chat/completions")
def chat(req: Req):
    t0 = time.perf_counter()
    prompt = tok.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with lock:  # serialize: this is the point of the baseline
        out = model.generate(**ids, max_new_tokens=req.max_tokens, do_sample=req.temperature > 0,
                             temperature=max(req.temperature, 1e-5))
    n_in = ids["input_ids"].shape[1]
    text = tok.decode(out[0][n_in:], skip_special_tokens=True)
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"completion_tokens": int(out.shape[1] - n_in),
                      "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--port", type=int, default=8001)
    a = p.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="cuda")
    uvicorn.run(app, host="0.0.0.0", port=a.port)
