"""Grabs a few open-access arxiv papers on ML for particle reconstruction.
Check each id yourself before using it in your eval set."""
import time
import urllib.request
from pathlib import Path

IDS = [
    "2003.11603",  # GNNs for particle reconstruction in HEP detectors
    "1902.07987",  # distance-weighted graph networks (GravNet / GarNet)
    "2101.08578",  # MLPF
    "1902.08570",  # ParticleNet
    "1807.02876",  # ML in HEP community white paper
]

out = Path(__file__).resolve().parent.parent / "data" / "pdfs"
out.mkdir(parents=True, exist_ok=True)

for pid in IDS:
    dest = out / f"{pid}.pdf"
    if dest.exists():
        print("have", dest.name)
        continue
    print("downloading", pid)
    req = urllib.request.Request(f"https://arxiv.org/pdf/{pid}", headers={"User-Agent": "research-agent-demo"})
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        f.write(r.read())
    time.sleep(3)  # be polite to arxiv
