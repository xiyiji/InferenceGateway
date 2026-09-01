"""results/*.json -> results/summary.md + throughput_vs_p95.png"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

d = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
rows = []
for f in sorted(d.glob("*.json")):
    j = json.loads(f.read_text())
    j["tag"] = f.stem.rsplit("_c", 1)[0]
    rows.append(j)
df = pd.DataFrame(rows)[["tag", "concurrency", "tokens_per_s", "e2e_p50_ms", "e2e_p95_ms",
                         "ttft_p95_ms", "goodput_rps", "errors"]]
(d / "summary.md").write_text(df.to_markdown(index=False))
print(df.to_markdown(index=False))

fig, ax = plt.subplots()
for tag, g in df.groupby("tag"):
    ax.plot(g["tokens_per_s"], g["e2e_p95_ms"], marker="o", label=tag)
ax.set_xlabel("throughput (tokens/s)"); ax.set_ylabel("p95 E2E latency (ms)"); ax.legend()
fig.savefig(d / "throughput_vs_p95.png", dpi=150, bbox_inches="tight")
