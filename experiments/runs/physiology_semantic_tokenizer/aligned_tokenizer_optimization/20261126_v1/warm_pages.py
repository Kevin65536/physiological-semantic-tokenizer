"""Read validated fit files in physical disk order, bounded to one sequential reader."""
import fcntl, json, os, struct, sys, time
from pathlib import Path
sys.path.insert(0, str(Path.cwd()/'experiments/scripts'))
from run_aligned_tokenizer_optimization import OUT, ROOT, ORIGINAL, settings
from src.data.dual_component_tokenizer import base_tokenizer_training_data
cfg,_=settings('eeg')
meta,*_=base_tokenizer_training_data(cfg,ROOT,recorded_root=ORIGINAL)
fit={tuple(k) for k in json.loads((OUT/'eeg/split.json').read_text())['fit']}
files=[]
for r in meta['rows']:
    if r['split']!='train' or (r['dataset'],r['subject']) not in fit or not any(x['positions'] for x in r['inputs']['train']):
        continue
    for k in ('x','s','o','mask'):
        p=Path(r['folder'])/(k+'.npy')
        assert not p.is_symlink()
        with p.open('rb') as f:
            buf=bytearray(88)
            struct.pack_into('=QQIIII',buf,0,0,2**64-1,0,0,1,0)
            fcntl.ioctl(f,0xc020660b,buf,True)
            offset=struct.unpack_from('=Q',buf,40)[0]
        files.append((offset,p))
start=time.monotonic();total=0
for i,(_,p) in enumerate(sorted(files)):
    with p.open('rb') as f:
        while (chunk:=f.read(1024*1024)):total+=len(chunk)
    if i%1000==0: print(i,len(files),total,round(time.monotonic()-start,1),flush=True)
print('complete',len(files),total,time.monotonic()-start,flush=True)
