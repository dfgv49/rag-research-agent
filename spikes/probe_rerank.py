"""Probe DashScope/Bailian rerank availability. stdlib only, no new deps.

Prints real HTTP status, real response body, and real wall-clock latency in ms.
The child process cannot write files under the DSH sandbox, so everything goes
to stdout only.
"""

import json
import os
import time
import urllib.error
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env():
    path = os.path.join(ROOT, ".env")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def mask(s):
    if not s:
        return "<empty>"
    return s[:4] + "***" + s[-2:] if len(s) > 6 else "***"


QUERY = "LangGraph 的检查点机制如何持久化对话状态？"
DOCS = [
    "LangGraph checkpointers save graph state after each super-step, enabling time travel and resumption.",
    "BM25 是一种基于词频与逆文档频率的稀疏检索算法，适合关键词精确匹配。",
    "向量检索通过 embedding 的余弦相似度召回语义相近的文档。",
    "今天天气晴朗，适合出门散步。",
]


def call(url, payload, key):
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", "replace")
            return resp.status, raw, (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace"), (time.perf_counter() - t0) * 1000, "HTTPError"
    except Exception as e:  # noqa: BLE001
        return None, repr(e), (time.perf_counter() - t0) * 1000, type(e).__name__


def main():
    load_env()
    key = os.environ.get("DASHSCOPE_API_KEY", "")
    print("KEY_PRESENT=%s KEY=%s LEN=%d" % (bool(key), mask(key), len(key)))
    print("QUERY=%s" % QUERY)
    native = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    # 1) Full response body for the working model
    payload = {
        "model": "gte-rerank-v2",
        "input": {"query": QUERY, "documents": DOCS},
        "parameters": {"return_documents": True, "top_n": 4},
    }
    status, body, ms, err = call(native, payload, key)
    print("\n=== FULL_RESPONSE gte-rerank-v2 status=%s err=%s latency_ms=%.1f ===" % (status, err, ms))
    try:
        print(json.dumps(json.loads(body), ensure_ascii=False, indent=2))
    except Exception:
        print(body)

    # 2) Latency sample: 5 sequential calls
    print("\n=== LATENCY_SAMPLE gte-rerank-v2 x5 ===")
    lats = []
    for i in range(5):
        s, b, m, e = call(native, payload, key)
        lats.append(m)
        print("  run%d status=%s latency_ms=%.1f" % (i + 1, s, m))
    lats_sorted = sorted(lats)
    print("  min=%.1f p50=%.1f max=%.1f mean=%.1f" % (
        lats_sorted[0], lats_sorted[len(lats_sorted) // 2], lats_sorted[-1],
        sum(lats) / len(lats)))

    # 3) No return_documents: confirm score field survives and payload trims
    payload2 = {"model": "gte-rerank-v2", "input": {"query": QUERY, "documents": DOCS},
                "parameters": {"return_documents": False, "top_n": 2}}
    s, b, m, e = call(native, payload2, key)
    print("\n=== return_documents=False status=%s latency_ms=%.1f ===" % (s, m))
    print(b[:800])

    # 4) Chinese-only pair sanity: does it score Chinese-Chinese sensibly?
    zh_query = "混合检索怎么做"
    zh_docs = ["混合检索把 BM25 与向量召回用 RRF 融合", "红烧肉的做法是先焯水再慢炖"]
    payload3 = {"model": "gte-rerank-v2", "input": {"query": zh_query, "documents": zh_docs},
                "parameters": {"return_documents": True, "top_n": 2}}
    s, b, m, e = call(native, payload3, key)
    print("\n=== CHINESE_ONLY status=%s latency_ms=%.1f ===" % (s, m))
    print(b[:800])


if __name__ == "__main__":
    main()
