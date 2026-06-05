import json
from pathlib import Path

base_dir = Path(__file__).parent

with open(base_dir / "out" / "ai_results.json") as f:
    res = json.load(f)

# list out keys that contains space 
for key in res.keys():
    if " " in key:
        print(key)