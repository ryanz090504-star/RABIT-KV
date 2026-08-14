import json, sys

path = sys.argv[1]
with open(path) as f:
    for line in f:
        r = json.loads(line)
        # Print all keys + values in a compact format
        for k, v in sorted(r.items()):
            if k in ('metadata', 'memory_breakdown', 'error_by_layer'):
                continue
            print(f"  {k}: {v}")
        print("---")
