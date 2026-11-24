"""Wait for both supervised continuous scorers before aggregating the paired development panel."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path(__file__).parent
for mode in ('wave', 'dynamic'):
    status = root / f'continuous_{mode}_status.json'
    unit = f'single-vq-factorial-{mode}-20261125.service'
    while json.loads(status.read_text())['status'] != 'completed':
        if subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit]).returncode:
            if json.loads(status.read_text())['status'] != 'completed':
                raise RuntimeError(f'{unit} stopped before evaluation completed')
            break
        time.sleep(5)
os.execv(sys.executable, [sys.executable, '-u', 'experiments/scripts/run_single_vq_dynamic_factorial.py', 'summarize'])
