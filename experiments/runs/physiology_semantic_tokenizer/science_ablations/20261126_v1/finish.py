from pathlib import Path
import subprocess
import sys
import time
import json

run=Path(__file__).resolve().parent
while not ((run/"wg/summary.json").exists()
           and len(list((run/"models").glob("*/score.json")))==44
           and len(list((run/"hb_replication/models").glob("*/score.json")))==24
           and len(list((run/"hb_replication/models").glob("*/wg_done.json")))==24):
    state={}
    for unit in ("resume17","resume43","score","wg","hb17","hb43","hbscore","hbwg"):
        state[unit]=subprocess.check_output(["systemctl","--user","show",f"science-ablations-{unit}","-p","Result","--value"],text=True).strip()
    if any(v not in ("success","") for v in state.values()):raise RuntimeError(state)
    time.sleep(12)
subprocess.run([sys.executable,"experiments/scripts/run_science_hb_replication.py","summary"],check=True)
subprocess.run([sys.executable,"experiments/scripts/summarize_science_ablations.py"],check=True)
