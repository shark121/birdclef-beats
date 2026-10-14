"""Stream the BirdCLEF 2026 zip from Kaggle and extract on the fly (no zip kept on disk)."""
import os, time, requests
from stream_unzip import stream_unzip
OUT = "/workspace/birdclef26"
tok = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
r = requests.get("https://www.kaggle.com/api/v1/competitions/data/download-all/birdclef-2026",
                 headers={"Authorization": "Bearer " + tok}, allow_redirects=False, timeout=60)
url = r.headers["Location"]
resp = requests.get(url, stream=True, timeout=120)
resp.raise_for_status()
total = int(resp.headers.get("Content-Length", 0))
seen = [0]; t0 = time.time(); last = [0.0]
def chunks():
    for c in resp.iter_content(1 << 20):
        seen[0] += len(c)
        if time.time() - last[0] > 15:
            last[0] = time.time()
            rate = seen[0] / (time.time() - t0) / 1e6
            print(f"{seen[0]/1e9:.2f}/{total/1e9:.2f} GB  {rate:.0f} MB/s", flush=True)
        yield c
n = 0
for name, size, data in stream_unzip(chunks()):
    name = name.decode()
    path = os.path.join(OUT, name)
    if name.endswith("/"):
        os.makedirs(path, exist_ok=True)
        for _ in data: pass
        continue
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        for b in data: fh.write(b)
    n += 1
print(f"DONE {n} files, {seen[0]/1e9:.2f} GB in {(time.time()-t0)/60:.1f} min", flush=True)
