"""Measure real artifact sizes so the local-reranker decision is evidence-based.

Fetches PyPI JSON (no downloads, nothing written to disk) and the HuggingFace
model metadata API. Stdlib only.
"""

import json
import urllib.request

def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "spike/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def pypi_wheel(pkg, want=("cp313", "win_amd64")):
    try:
        data = get("https://pypi.org/pypi/%s/json" % pkg)
    except Exception as e:  # noqa: BLE001
        return "FETCH_FAIL: %r" % (e,)
    ver = data["info"]["version"]
    best = []
    for u in data["urls"]:
        fn = u["filename"]
        if u["packagetype"] != "bdist_wheel":
            continue
        if all(w in fn for w in want):
            best.append((fn, u["size"] / 1024 / 1024))
    best.sort(key=lambda x: x[1], reverse=True)
    return "version=%s | cp313-win_amd64 wheels: %s" % (
        ver, [(f, round(s, 1)) for f, s in best[:3]] or "NONE (no cp313 win_amd64 wheel published)")

def hf_model(model):
    try:
        data = get("https://huggingface.co/api/models/%s?blobs=true" % model)
    except Exception as e:  # noqa: BLE001
        return "FETCH_FAIL: %r" % (e,)
    total = 0
    parts = []
    for s in data.get("siblings", []):
        name = s.get("rfilename", "")
        size = s.get("size") or 0
        if name.endswith((".safetensors", ".bin", ".onnx", ".json")):
            total += size
            parts.append((name, round(size / 1024 / 1024, 1)))
    parts.sort(key=lambda x: x[1], reverse=True)
    return "files(MB): %s | SUM=%.1f MB" % (parts[:6], total / 1024 / 1024)

for pkg in ("torch", "transformers", "sentence-transformers"):
    print("[PyPI]", pkg, "->", pypi_wheel(pkg))
print()
for m in ("BAAI/bge-reranker-v2-m3", "BAAI/bge-reranker-base"):
    print("[HF]", m, "->", hf_model(m))
