import json
import urllib.request
from pathlib import Path

sources_file = Path(__file__).parent / "sources.json"
sources = json.loads(sources_file.read_text(encoding="utf-8"))

Path("knowledge_base").mkdir(exist_ok=True)
for name, url in sources.items():
    out = Path("knowledge_base") / name
    if out.exists():
        continue
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, out)
print("Done.")
