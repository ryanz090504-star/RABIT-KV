"""READ-ONLY recursive listing of the modelscope-llama31-cache Modal volume (no writes, no commit)."""
import datetime
import json
import sys
import warnings

warnings.simplefilter("ignore")
import modal

vol = modal.Volume.from_name("modelscope-llama31-cache")
rows = []
for e in vol.listdir("/", recursive=True):
    rows.append({
        "path": e.path,
        "type": str(e.type).split(".")[-1],
        "size": e.size,
        "mtime_utc": datetime.datetime.fromtimestamp(e.mtime, datetime.timezone.utc).isoformat(),
    })
rows.sort(key=lambda r: r["path"])
json.dump(rows, open(sys.argv[1], "w", encoding="utf-8"), indent=1)
for r in rows:
    print(f'{r["type"]:5} {r["size"]:>12} {r["mtime_utc"]} {r["path"]}')
