#!/usr/bin/env python3
"""Export the frozen, offline 2 s observation baseline, with inference-only dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[2]
SUITE=ROOT/'experiments/runs/physiology_semantic_tokenizer/observation_factorization'
PARENT=SUITE/'20261120_development_v3'; A=SUITE/'20261120_allocation_v1'; G=SUITE/'20261120_gain_v1'
G0=ROOT/'experiments/runs/physiology_semantic_tokenizer/dual_component_tokenizer/20261117_base_g0_v2'


def load(path): return json.loads(path.read_text())
def save(path, x): path.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


DEMO='''"""Portable inference smoke test. Only synthetic inputs; no data/cache setup or training."""
from pathlib import Path
import json
import numpy as np
from src.tokenizers.observation_baseline import ObservationBaseline
root=Path(__file__).resolve().parent
baseline=ObservationBaseline(root)
rng=np.random.default_rng(2207); stats={}
for modality, channels, ticks in [('eeg',1,400),('hb',2,20)]:
    x=rng.normal(size=(7,channels,ticks)).astype('float32') # complete 2 s, already in training coordinates
    ids=baseline.encode(x,modality) # one nominal 0..15 per patch
    code=baseline.encode(x,modality,detail=True)
    np.testing.assert_array_equal(ids['pattern'],code['pattern'])
    record={}
    for detail,values in [(False,ids),(True,code)]:
        wire=baseline.pack(values,modality,detail=detail)
        mode='full' if detail else 'ids'; bits=baseline.manifest['modalities'][modality]['total_bits_per_patch'] if detail else 4
        assert len(wire)==(len(x)*bits+7)//8
        recv=baseline.unpack(wire,len(x),modality,detail=detail)
        y=baseline.decode(recv,modality,detail=detail)
        np.testing.assert_array_equal(y,baseline.decode(values,modality,detail=detail))
        assert y.shape==x.shape and np.isfinite(y).all()
        record[mode]={'bits_per_item':bits,'batch_bytes':len(wire)}
    stats[modality]=dict(**record,n_items=len(x))
print(json.dumps({'status':'passed','synthetic_only':True,'payload':stats},indent=2))
'''


def export(out):
    parent,alloc,gain=[load(r/'summary.json') for r in (PARENT,A,G)]
    if (any(v['status']!='completed' for v in (parent,alloc,gain)) or parent['experiment_id']!='TK-OBS-FACTOR-v1'
            or alloc['experiment_id']!='TK-OBS-ALLOCATION-v1'): raise ValueError('all exact generations must be complete')
    out=out.resolve();
    if out.parent!=SUITE or out in [PARENT,A,G]: raise ValueError('separate suite release directory required')
    out.mkdir(exist_ok=False); bundle=out/'observation_baseline'; bundle.mkdir(); (bundle/'models').mkdir()
    (bundle/'src/tokenizers').mkdir(parents=True);(bundle/'src/__init__.py').write_text('');(bundle/'src/tokenizers/__init__.py').write_text('')
    for name in ['factorized_observation','adaptive_observation_detail','gain_shape_observation','observation_baseline']:
        sub=Path('src/tokenizers')/(name+'.py');shutil.copy2(ROOT/sub,bundle/sub)
    if digest(bundle/'src/tokenizers/factorized_observation.py')!=parent['sources']['src/tokenizers/factorized_observation.py']:
        raise ValueError('parent codec has changed')
    provenance={r.name:{'experiment_id':v['experiment_id'],'summary_sha256':digest(r/'summary.json')} for r,v in [(PARENT,parent),(A,alloc),(G,gain)]}
    for r in [A,G]:
        source=load(r/'source_manifest.json');
        for rel,h in source.items():
            if rel.startswith('src/tokenizers/') and (bundle/rel).is_file() and digest(bundle/rel)!=h: raise ValueError('model/decoder source drift')
    models={}
    for mod,run,file,report in [('eeg',G,'eeg_gain_shape.joblib',gain),('hb',A,'hb_adaptive.joblib',alloc)]:
        selected=report['modalities'][mod];path=Path('models')/file;router=Path('models')/f'{mod}_router.joblib'
        shutil.copy2(run/file,bundle/path);shutil.copy2(PARENT/f'{mod}_codec.joblib',bundle/router)
        models[mod]=dict(router=str(router),model=str(path),mode='conditional',pattern_bits_per_patch=4,
            total_bits_per_patch=int(selected['selected_bits']),passed_development_macro_screen=bool(selected['met_all_screens']),
            observed_x_nmse=selected['x_nmse'],centered_cosine=selected['centered_cosine'])
    scales=load(G0/'scales.json');expected='d8b6634e55fa73bbc1243650'
    if (hashlib.sha256(json.dumps(scales,sort_keys=True).encode()).hexdigest()[:24]!=expected or scales['fitting_split']!='train'):
        raise ValueError('unexpected train scaling provenance')
    subset={m:{db:{p:{'scale':v['scale'],'X_target_variance':v['target_variance']['x']} for p,v in tab.items()} for db,tab in scales['positions'][m].items()} for m in models}
    save(bundle/'training_x_scales.json',subset);shutil.copy2(PARENT/'atlas.csv',bundle/'pattern_atlas.csv');(bundle/'demo.py').write_text(DEMO)
    deps={d:version(d) for d in ['numpy','scipy','scikit-learn','joblib','threadpoolctl']};(bundle/'requirements.txt').write_text(''.join(f'{n}=={v}\n' for n,v in deps.items()))
    e,h=models['eeg'],models['hb']
    helptext=f'''# 两秒双流观测 tokenizer｜开发版

用途：先保留可解释的观测词，然后让波形消费者按需要读取细节。**这是四库公开开发面板上的工程基线，并非独立确认、生理状态标签、临床模型或 S/O 分离器。** 心算验证没有显示对盲波形词的优势。模型及推断代码随附，可在不恢复研究数据库的情况下运行。

```bash
# 在解压后的 observation_baseline 目录；使用已有仓库环境或按 requirements.txt 配置兼容 Python
python demo.py
```

```python
from pathlib import Path
from src.tokenizers.observation_baseline import ObservationBaseline
b = ObservationBaseline(Path('.'))
# x: 完整、已过 G0 预处理且已除训练位点尺度的 numpy 数组，见下文
ids = b.encode(x, 'eeg')['pattern']                 # 与 full 的 pattern 逐项相同，下游可只读该列
code = b.encode(x, 'eeg', detail=True)             # 只含整数，无实时浮点数送入 decoder
wire = b.pack(code, 'eeg', detail=True)
y = b.decode(b.unpack(wire, len(x), 'eeg', detail=True), 'eeg', detail=True)
```

|输入|词流|完整波形负载/2秒/原生位置|开发整体工程线|
|---|---|---:|---|
|EEG：`[N,1,400]`，200 Hz，一电极|4 bit：相对频谱|{e['total_bits_per_patch']} bit：含ID、8-bit均值、6-bit增益及整数形状|{'达到' if e['passed_development_macro_screen'] else '未达到；最高档为回退'}|
|Hb：`[N,2,20]`，10 Hz；同一位置依次 HbO、HbR|4 bit：两色水平与斜率|{h['total_bits_per_patch']} bit：含ID与逐词分配的整数细节|{'达到' if h['passed_development_macro_screen'] else '未达到；最高档为回退'}|

**坐标与边界**：不用原始电压/光强直接调用。使用指定 G0 预处理（含既有离线滤波、Hb 校准）；若只有未除尺度的 *G0 X*，先 `x=b.scale_g0(g0_x,'eeg',dataset,position)`，它只除以已有训练尺度，不减片内均值，不增加验证校准。Hb 两色共用输入尺度但评估方差分色。无效或不完整片保留 mask，不能补零生成假词。在 G0 以外的设备/人群，不能默认此坐标、码分布或阈值适用。

**下游模块**：传 `(dataset, subject, record, modality, position, native_time, valid, ID)`；ID 是类别，不是有序数，EEG 和 Hb 同号不对应。`pattern_atlas.csv` 是每个固定 ID 在训练与验证中真实属性的四分位数与支持量；罕见/零支持 ID 不做状态命名。`detail=False` 解码只给训练原型，不能用它恢复 EEG 相位。完整码率包括每片所有测量信息；版本/位置/时钟/计数/有效性和固定字典为双方预先共享，仅整批末字节补零。

**验证与安全**：`manifest.json` 记录库版本、模型校验和、运行及坐标身份。`demo.py` 只生成合成输入，核对 ID 一致、真实字节数和解码回放。joblib 只能从可信的本地发布者加载；清单校验和不是签名。用于研究而非诊疗。
''';(bundle/'README.md').write_text(helptext)
    files=sorted(p for p in bundle.rglob('*') if p.is_file());manifest=dict(schema='observation_baseline_bundle_v1',source='public development; observation only; no protected evaluation',
        python=sys.version.split()[0],versions=deps,scale_identity=expected,modalities=models,evidence=provenance,sha256={str(p.relative_to(bundle)):digest(p) for p in files});save(bundle/'manifest.json',manifest)
    archive=out/'observation_baseline.zip'
    with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
        for f in sorted([*files,bundle/'manifest.json']):z.write(f,f.relative_to(out))
    # Fresh isolated subprocess uses the ARCHIVE and packaged source, never the workspace imports.
    with tempfile.TemporaryDirectory(prefix='obs-bundle-check-') as tmp:
        with ZipFile(archive) as z: z.extractall(tmp)
        env={k:v for k,v in os.environ.items() if k not in ['PYTHONPATH','PYTHONHOME']};env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        check=subprocess.run([sys.executable,'demo.py'],cwd=Path(tmp)/'observation_baseline',env=env,capture_output=True,text=True,check=True)
    result=json.loads(check.stdout); assert result['status']=='passed';save(out/'release.json',dict(created=datetime.now(timezone.utc).isoformat(),status='verified',archive=archive.name,sha256=digest(archive),size_bytes=archive.stat().st_size,portable_demo=result,modalities=models));return archive


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--output',type=Path,required=True);print(export(cli.parse_args().output))
