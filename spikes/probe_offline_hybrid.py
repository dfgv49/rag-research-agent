"""Offline equivalent-validation prototype for hybrid retrieval + rerank.

Answers: with Docker/ES unavailable, can we still validate the RRF fusion and
ridemension rerank math? This runs entirely in-process:

  dense  : DashScope text-embedding-v4 (real API, the project's own model)
  sparse : pure-python BM25 (k1=1.5, b=0.75), no new dependency
  fusion : RRF (k=60), written here exactly as the production plan would
  rerank : DashScope gte-rerank-v2 /rerank (real API)

Everything goes to stdout; the DSH sandbox blocks child-process file writes.
"""

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

QUERIES = [
    "How do I inspect a runnable's input and output schemas?",
    "What are the different types of message chunks in streaming?",
    "How do I build an agent with tools?",
]

TOP_WINDOW = 20   # per-route candidate window (production plan: k=20)
FINAL_K = 4       # what the agent finally sees (current production k=4)
RRF_C = 60        # RRF rank constant
RERANK_N = 10     # how many fused docs go to the reranker


def load_env():
    with open(os.path.join(ROOT, ".env"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


# ---------------- dense route: real DashScope embedding ----------------

def embed(texts):
    from langchain_openai import OpenAIEmbeddings
    enc = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key=os.environ["DASHSCOPE_API_KEY"],
        base_url=os.environ.get("DASHSCOPE_BASE_URL",
                                "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=20,
    )
    t0 = time.perf_counter()
    vecs = enc.embed_documents(texts)
    return np.array(vecs, dtype=np.float32), (time.perf_counter() - t0) * 1000


def dense_search(qvec, dmat, k):
    t0 = time.perf_counter()
    sims = dmat @ qvec / (np.linalg.norm(dmat, axis=1) * np.linalg.norm(qvec) + 1e-9)
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order], (time.perf_counter() - t0) * 1000


# ---------------- sparse route: pure-python BM25 ----------------

TOKEN_RE = re.compile(r"[a-z0-9_]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


def build_bm25(docs, k1=1.5, b=0.75):
    toks = [tokenize(d) for d in docs]
    df = Counter()
    for t in toks:
        df.update(set(t))
    avgdl = sum(len(t) for t in toks) / max(len(toks), 1)
    return {"toks": toks, "df": df, "avgdl": avgdl, "N": len(toks), "k1": k1, "b": b}


def bm25_search(idx, query, k):
    t0 = time.perf_counter()
    q = tokenize(query)
    scores = []
    for i, toks in enumerate(idx["toks"]):
        tf = Counter(toks)
        dl = len(toks)
        s = 0.0
        for term in q:
            f = tf.get(term, 0)
            if f == 0:
                continue
            idf = math.log(1 + (idx["N"] - idx["df"][term] + 0.5) / (idx["df"][term] + 0.5))
            s += idf * (f * (idx["k1"] + 1)) / (f + idx["k1"] * (1 - idx["b"] + idx["b"] * dl / idx["avgdl"]))
        scores.append((i, s))
    scores.sort(key=lambda x: -x[1])
    return [x for x in scores[:k] if x[1] > 0], (time.perf_counter() - t0) * 1000


# ---------------- RRF fusion ----------------

def rrf(rank_lists, c=RRF_C):
    t0 = time.perf_counter()
    acc = Counter()
    for lst in rank_lists:
        for rank, (doc_id, _score) in enumerate(lst, start=1):
            acc[doc_id] += 1.0 / (c + rank)
    fused = sorted(acc.items(), key=lambda x: -x[1])
    return fused, (time.perf_counter() - t0) * 1000


# ---------------- rerank route: real DashScope /rerank ----------------

def rerank(query, docs, top_n):
    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    payload = {
        "model": "gte-rerank-v2",
        "input": {"query": query, "documents": docs},
        "parameters": {"return_documents": False, "top_n": top_n},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + os.environ["DASHSCOPE_API_KEY"],
                 "Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
        ms = (time.perf_counter() - t0) * 1000
        return [(int(x["index"]), float(x["relevance_score"]))
                for x in body["output"]["results"]], ms, None
    except urllib.error.HTTPError as e:
        return [], (time.perf_counter() - t0) * 1000, "HTTP %s: %s" % (e.code, e.read()[:200])
    except Exception as e:  # noqa: BLE001
        return [], (time.perf_counter() - t0) * 1000, repr(e)


def brief(docs, ids, n=FINAL_K):
    return [docs[i][:60].replace("\n", " ") for i, _ in ids[:n]]


def main():
    load_env()
    raw = json.load(open(os.path.join(ROOT, "src", "sample_docs.json"), encoding="utf-8"))
    docs = [d["page_content"] if isinstance(d, dict) else d for d in raw]
    print("CORPUS_SIZE=%d  TOP_WINDOW=%d FINAL_K=%d RRF_C=%d RERANK_N=%d"
          % (len(docs), TOP_WINDOW, FINAL_K, RRF_C, RERANK_N))

    dmat, embed_ms = embed(docs)
    print("EMBED corpus: ms=%.0f shape=%s" % (embed_ms, dmat.shape))
    idx = build_bm25(docs)

    for q in QUERIES:
        print("\n" + "=" * 78)
        print("QUERY:", q)
        qvec, q_ms = embed([q])
        qvec = qvec[0]

        dn, d_ms = dense_search(qvec, dmat, TOP_WINDOW)
        sp, s_ms = bm25_search(idx, q, TOP_WINDOW)
        fused, f_ms = rrf([dn, sp])

        top_fused = fused[:RERANK_N]
        rr, r_ms, r_err = rerank(q, [docs[i] for i, _ in top_fused], FINAL_K)
        # map rerank's positional index back to corpus id
        rr_ids = [(top_fused[pos][0], score) for pos, score in rr] if not r_err else []

        print("  latency: embed_query=%.0fms dense=%.1fms bm25=%.1fms rrf=%.2fms rerank=%.0fms err=%s"
              % (q_ms, d_ms, s_ms, f_ms, r_ms, r_err))
        print("  dense  top4:", brief(docs, dn))
        print("  bm25   top4:", brief(docs, sp))
        print("  rrf    top4:", brief(docs, top_fused))
        print("  rerank top4:", brief(docs, rr_ids))
        if rr_ids:
            print("  rerank scores:", [round(s, 4) for _, s in rr_ids])
            print("  rrf->rerank 顺序变化:",
                  [top_fused.index(x) for x in rr_ids if x in top_fused])
        dset = {i for i, _ in dn[:FINAL_K]}
        fset = {i for i, _ in top_fused[:FINAL_K]}
        rset = {i for i, _ in rr_ids}
        print("  overlap dense∩rrf=%d  rrf∩rerank=%d  dense∩rerank=%d (of %d)"
              % (len(dset & fset), len(fset & rset), len(dset & rset), FINAL_K))


if __name__ == "__main__":
    main()
