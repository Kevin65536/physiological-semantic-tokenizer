"""Two independent workers, each pinned to its PCI-resolved permitted device."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json, os, subprocess, sys
from datetime import datetime, timezone

root=Path.cwd()
base=Path(__file__).resolve().parent
action=sys.argv[1]
assert action in ('pilot','formal')
entry=root/'experiments/scripts/run_eeg_encoder_diagnostics.py'
gpu_info=subprocess.check_output(['nvidia-smi','--query-gpu=pci.bus_id,uuid','--format=csv,noheader'],text=True)
mapping=dict(line.split(', ') for line in gpu_info.splitlines())
devices=[mapping['00000000:'+s+':00.0'] for s in ('4B','CA')]
assert len(set(devices))==2
jobs=[('privileged','continuous'),('observed','continuous'),('wide','continuous'),('deep','continuous'),
      ('observed','hard'),('wide','hard'),('deep','hard')]
pending=list(jobs)
env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',
         VQ_MMAP_PARENT_CACHE='4096',PYTHONPATH=str(root))
def work(arch,fit,uuid):
    phases=['pilot'] if action=='pilot' else ['train','score']
    with (base/f'{arch}_{fit}_{action}.log').open('a',buffering=1) as log:
        for phase in phases:
            cmd=[sys.executable,'-u',str(entry),phase,'--arch',arch,'--fit',fit]
            print(datetime.now(timezone.utc).isoformat(),phase,cmd,file=log)
            subprocess.run(cmd,check=True,env=dict(env,CUDA_VISIBLE_DEVICES=uuid),stdout=log,stderr=subprocess.STDOUT)
active={}; failed=[]
with ThreadPoolExecutor(2) as pool:
    while pending or active:
        for uuid in devices:
            if any(v[0]==uuid for v in active.values()):
                continue
            for job in pending:
                if not any(v[1][0]==job[0] for v in active.values()):
                    pending.remove(job)
                    active[pool.submit(work,*job,uuid)]=uuid,job
                    break
        for future in wait(active,return_when=FIRST_COMPLETED).done:
            uuid,job=active.pop(future)
            try: future.result()
            except Exception as e:
                failed.append([job,str(e)])
                print(job,str(e),flush=True)
if failed:
    (base/(action+'_failures_'+str(os.getpid())+'.json')).write_text(json.dumps(failed,indent=2))
    sys.exit(1)
if action=='formal':
    subprocess.run([sys.executable,str(entry),'summarize'],env=dict(env,CUDA_VISIBLE_DEVICES=''),check=True)
