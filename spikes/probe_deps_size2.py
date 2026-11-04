"""Round 2: correct the universal-wheel query and test HF connectivity."""

import json
import socket
import time
import urllib.request

def get_json(url, timeout=25):
    req = urllib.request.Request(url, headers={"User-Agent": "spike/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))

def pypi_any(pkg):
    try:
        data = get_json("https://pypi.org/pypi/%s/json" % pkg)
    except Exception as e:  # noqa: BLE001
        return "FETCH_FAIL: %r" % (e,)
    ver = data["info"]["version"]
    rows = [(u["filename"], round(u["size"] / 1024 / 1024, 2))
            for u in data["urls"] if u["packagetype"] == "bdist_wheel"]
    rows.sort(key=lambda x: x[1], reverse=True)
    return "version=%s | wheels: %s" % (ver, rows[:4])

def probe_host(host, port=443):
    t0 = time.perf_counter()
    try:
        s = socket.create_connection((host, port), timeout=8)
        s.close()
        return "TCP OK in %.0f ms" % ((time.perf_counter() - t0) * 1000)
    except Exception as e:  # noqa: BLE001
        return "TCP FAIL after %.0f ms: %r" % ((time.perf_counter() - t0) * 1000, e)

for p in ("transformers", "sentence-transformers"):
    print("[PyPI]", p, "->", pypi_any(p))
print()
for h in ("huggingface.co", "hf-mirror.com", "pypi.org", "dashscope.aliyuncs.com"):
    print("[TCP]", h, "->", probe_host(h))
