"""Export a frozen fold0 EEG/Hb111 research pair and verify an extracted copy."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.tokenizers.band_exchange_words import BandExchangeEEG
from src.tokenizers.eeg_transient_gain import load_model as load_source
from experiments.scripts.apply_joint_tokenizer import apply


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def entry(path,root):return dict(path=str(path.relative_to(root)),sha256=sha(path))


def prepare(args):
    from src.data.dual_component_tokenizer import base_tokenizer_training_data,base_tokenizer_joint_eeg
    run=args.run;parent=run/'component_source_transfer/band_exchange_words_v1';out=args.output
    decision=json.loads((parent/'summary.json').read_text())['arms']['exchange_balanced_energy']
    if not (decision['checks']['actual_ID_S'] and decision['checks']['actual_ID_O'] and not decision['passed']):
        raise ValueError('this export is the retained pattern-passed waveform-failed research reference')
    package=out/args.package_name;package.mkdir(parents=True,exist_ok=False);models=package/'models';models.mkdir()
    cfg=yaml.safe_load((ROOT/'experiments/configs/physiology_semantic_tokenizer/dual_component_base_g1_v2.yaml').read_text())
    manifest,scales,_,_=base_tokenizer_training_data(cfg,ROOT);rows=manifest['rows']
    population=json.loads((run/'component_source_transfer/hard_as_v1/local_information_v1/population.json').read_text())
    if (population['cfg_identity'],population['scales_identity'])!=(manifest['cfg_identity'],manifest['scales_identity']):
        raise ValueError('original G0 training coordinate required')
    books={b:[] for b in 'SO'};cov={};sources={};weights={};scope={};observed={};expected={};identities=[]
    for b in 'SO':
        for role in range(5):
            with np.load(parent/'books'/f'fold_0_{b}_{role}.npz',allow_pickle=False) as z:
                m=json.loads(str(z['metadata']))
                if (m['fold'],m['branch'],m['role'])!=(0,b,role):raise ValueError('fold0 book identity changed')
                books[b].append({k:z[k].copy() for k in z if k!='metadata'})
    for dataset,names in sorted(population['native_positions'].items()):
        cases=[c for c in population['cases'] if c['dataset']==dataset]
        fit=sorted({c['subject'] for c in cases if c['fold']!=0});scope[dataset]=fit
        sources[dataset]=load_source(run/'component_source_transfer/transient_gain_v1'/f'{dataset}_fold0_transient.npz')
        if set(sources[dataset].metadata['fit_subjects'])!=set(fit):raise ValueError('fold0 source fit scope changed')
        with np.load(run/'component_source_transfer/adaptive_band_v1'/dataset/'fold_0_model.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['names'],names)
            if set(z['fit_people'])!=set(fit):raise ValueError('fold0 covariance fit scope changed')
            cov[dataset]={b:z[b].copy() for b in 'SO'}
        table={k.upper():v for k,v in scales['positions']['eeg'][dataset].items()}
        weights[dataset]={b:np.array([1/float(np.ravel(table[n.upper()]['target_variance'][b.lower()])[0]) for n in names]) for b in 'SO'}
        held=[c for c in cases if c['fold']==0]
        c=held[1]  # fixed next held crop, disjoint from the 16 prior first-crop checks
        row=rows[c['crop']['row']];start=c['crop']['start_s']
        if (row['split'],row['modality'],row['subject'])!=('train','eeg',c['subject']):raise ValueError('train boundary before X read')
        fields,seen,_=base_tokenizer_joint_eeg(rows,dict(c['crop'],positions=[]),scales,names,allowed_splits=('train',))
        source=np.load(Path(row['folder'])/'x.npy',mmap_mode='r')[start*200:(start+30)*200,:,0]
        raw=np.zeros((len(names),6000,1),np.float32);lookup={n.upper():i for i,n in enumerate(names)}
        for j,n in enumerate(row['positions']):raw[lookup[n.upper()],:,0]=source[:,j]
        # Missing native sites remain present; the integer0 transmits absence.
        raw[~seen]=0
        normalized=np.stack([raw[j,:,0]/table[n.upper()]['scale'] for j,n in enumerate(names)]).astype(np.float32)
        np.testing.assert_array_equal(normalized,fields['x'].T.astype(np.float32))
        offset=0
        for prior in cases:
            if prior is c:break
            offset+=prior['count']
        held_offset=sum(p['count'] for p in held[:1]);native=np.asarray(c['native']);inv=native.argsort()
        with np.load(parent/dataset/'fold_0_ids.npz',allow_pickle=False) as z:
            for b in 'SO':expected.setdefault('eeg_'+b,[]).append(z['exchange_balanced_'+b][offset:offset+c['count']][inv])
        with np.load(parent/'replay_review_v1'/dataset/'fold_0_exchange_balanced_energy_prediction.npz',allow_pickle=False) as z:
            factors=np.array([table[n.upper()]['scale'] for n in names])
            for b in ('S','O','sum'):
                expected.setdefault('wave_eeg_'+b,[]).append(z[b][held_offset:held_offset+c['count']][inv].reshape(len(names),6000,1)*factors[:,None,None])
        for key,value in dict(eeg_x=raw,eeg_dataset=np.repeat(dataset,len(names)),eeg_position=np.array(names),
                              eeg_crop=np.repeat(len(identities),len(names)),eeg_observed=seen).items():observed.setdefault(key,[]).append(value)
        identities.append(dict(modality='eeg',dataset=dataset,subject=row['subject'],record=row['record'],row=c['crop']['row'],
            start_s=start,native_positions=names,observed=seen.tolist(),source_unit=row.get('source_unit'),split='train',
            rule='second original fold0 held crop; first was in prior16 checks',input_float32_X_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    metadata=dict(schema='band_exchange_eeg_v1',modality='eeg',positions=population['native_positions'],layout='exchange',
        assignment='balanced',synthesis='energy',codebook_size=5116,context_seconds=30,patch_seconds=2,held_fold=0,
        input_identity=manifest['cfg_identity'],scales_identity=manifest['scales_identity'],fit_subjects_by_dataset=scope,
        scientific_status='both_single_ID_pattern_passed_waveform_failed_research_reference_only')
    model=BandExchangeEEG(metadata,books,cov,sources,weights);model.save(models/'eeg_fold0.npz')
    # Four existing, explicitly train-only Hb examples plus a retained quiet case.
    for dataset in sorted(population['native_positions']):
        with np.load(run/'multimodal_hb'/dataset/'prepared.npz',allow_pickle=False) as z:ex=z['examples'].copy()
        with np.load(run/'multimodal_hb/local_geometry_v1'/dataset/'reference.npz',allow_pickle=False) as z:
            selected=[min(range(len(ex)),key=lambda i:tuple(ex[i]))]
            quiet=np.flatnonzero(z['quiet'])
            if len(quiet) and int(quiet[0]) not in selected:selected.append(int(quiet[0]))
            for index in selected:
                i,start,site=map(int,ex[index]);row=rows[i]
                if (row['split'],row['modality'])!=('train','hb'):raise ValueError('Hb train boundary before X read')
                name=row['positions'][site];factor=scales['positions']['hb'][dataset][name]['scale']
                raw=np.asarray(np.load(Path(row['folder'])/'x.npy',mmap_mode='r')[start*10:(start+30)*10,site],np.float32)
                if not np.isfinite(raw).all():raise ValueError('finite registered Hb observation required')
                observed.setdefault('hb_x',[]).append(raw[None]);observed.setdefault('hb_dataset',[]).append(np.array([dataset]));observed.setdefault('hb_position',[]).append(np.array([name]))
                for b in 'SO':expected.setdefault('hb_'+b,[]).append(z['ids_'+b][index:index+1])
                for b in ('S','O','sum'):expected.setdefault('wave_hb_'+b,[]).append(z[b][index:index+1].reshape(1,300,2)*factor)
                identities.append(dict(modality='hb',dataset=dataset,subject=row['subject'],record=row['record'],row=i,start_s=start,
                    position=name,source_unit=row.get('source_unit'),quiet=bool(z['quiet'][index]),split='train',rule='old train-panel lexicographic first and first quiet'))
    observed={k:np.concatenate(v) for k,v in observed.items()};expected={k:np.concatenate(v) for k,v in expected.items()}
    np.savez_compressed(package/'observations.npz',**observed)
    hb=run/'hb_native_id_readout/v1'
    for n in ('native_readout.pt','current.pt','model.pt'):shutil.copyfile(hb/n,models/n)
    scale_subset=dict(positions={m:{d:{n:dict(scale=v['scale']) for n,v in sites.items()} for d,sites in scales['positions'][m].items()} for m in ('eeg','hb')},source_scales_identity=manifest['scales_identity'])
    write(package/'scales.json',scale_subset)
    bundle=dict(schema='joint_tokenizer_inference_v1',experimental_status='fold0_EEG_pattern_passed_waveform_failed_Hb111_different_fit_scope_research_pair',
        eeg=dict(kind='band_exchange_eeg',**entry(models/'eeg_fold0.npz',package)),
        hb=dict(kind='hb_native_id_readout',**entry(models/'native_readout.pt',package)),scales=entry(package/'scales.json',package))
    write(package/'bundle.json',bundle);write(package/'sample_identity.json',identities)
    ids=apply(package/'bundle.json','encode',observed)
    waves=apply(package/'bundle.json','decode',ids)
    # Hb111's float64 BLAS reduction has a last-bit dependence on batch size.
    # Keep that comparison with the historical full panel, then require exact
    # portable equality against the unchanged source model on this same batch.
    from src.tokenizers.hb_native_id_readout import load_model as load_hb
    original_hb=load_hb(hb/'native_readout.pt')
    original_hb_wave=original_hb.decode_ids({b:ids['hb_'+b] for b in 'SO'})
    factors=np.array([scales['positions']['hb'][str(d)][str(p)]['scale']
        for d,p in zip(observed['hb_dataset'],observed['hb_position'])])
    historical_hb={}
    for b in ('S','O','sum'):
        key='wave_hb_'+b
        historical_hb[b]=float(np.max(np.abs(waves['hb_'+b]-expected[key])))
        expected[key]=original_hb_wave[b].reshape(len(factors),300,2)*factors[:,None,None]
    write(out/'historical_Hb_batch_arithmetic.json',dict(max_original_unit_difference=historical_hb,
        historical_reference_batch='whole3088-panel dataset batches',current_batch=len(factors),
        decoder_unchanged=True,portable_required_endpoint='original frozen Hb111 decoded on identical5 example batch',
        no_tolerance_relaxation_for_portable_checks=True))
    for modality in ('eeg','hb'):
        for b in 'SO':np.testing.assert_array_equal(ids[modality+'_'+b],expected[modality+'_'+b])
        for b in ('S','O','sum'):np.testing.assert_array_equal(waves[modality+'_'+b],expected['wave_'+modality+'_'+b])
    np.savez_compressed(package/'ids.npz',**ids);np.savez_compressed(package/'voltage_and_Hb.npz',**waves)
    write(package/'training_scope.json',dict(EEG=metadata,Hb='original frozen111 head plus unchanged S129/O257 quiet encoder, fitted on all permitted training people; historic G1 ordinary-validation exposure inherited',
        joint_qualification=False,no_new_fitting=True,no_new_validation_signal_reads=True,EEG_CV_macro_nmse=decision['summary']['metrics']['nmse']['macro'],
        IDs_per30seconds_per_position={'EEG_S':15,'EEG_O':15,'Hb_S':15,'Hb_O':15},nominal_bits_per_second_per_position={'EEG_S':6.5,'EEG_O':6.5,'Hb_S':4,'Hb_O':4.5}))
    return package


def portable(args,package):
    out=args.output
    # Reuse the previous exporter strategy: collect exactly the imported repository
    # inference modules in a fresh process, then pin them inside the archive.
    code="""import sys,json,numpy as np
from pathlib import Path
from experiments.scripts.apply_joint_tokenizer import apply
root=Path.cwd();p=Path(sys.argv[1])
with np.load(p/'observations.npz') as x:y=apply(p/'bundle.json','encode',x)
apply(p/'bundle.json','decode',y)
files=[]
for m in list(sys.modules.values()):
 f=getattr(m,'__file__',None)
 if isinstance(f,str):
  q=Path(f).resolve()
  if q.suffix=='.py' and q.is_relative_to(root) and (q.is_relative_to(root/'src') or q==root/'experiments/scripts/apply_joint_tokenizer.py'):files.append(str(q.relative_to(root)))
(p/'inference_modules.json').write_text(json.dumps(sorted(set(files)),indent=2))
"""
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','PYTHONPATH':''}
    subprocess.run([sys.executable,'-c',code,str(package.resolve())],cwd=ROOT,check=True,env=env)
    files=json.loads((package/'inference_modules.json').read_text())
    for name in files:
        dest=package/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        for parent in Path(name).parents:
            init=ROOT/parent/'__init__.py'
            if str(parent)!='.' and init.is_file():
                d=package/parent/'__init__.py';d.parent.mkdir(parents=True,exist_ok=True)
                if not d.exists():shutil.copyfile(init,d)
    (package/'README.zh.md').write_text('''# fold0 四分支研究复现包

EEG 为新 exchange_balanced_energy 冻结词典：两支单码局部信息已通过原训练门槛，波形验收失败。该包用于复现研究参照。EEG 只含原四折中的 fold0：词典、空间协方差及瞬态来源用其他三折训练人拟合；Hb 为原冻结111读出及S129/O257静默整数，来自所有训练人，历史G1来源曾暴露普通验证。两个训练范围不同，未经过联合达标验收。本次没有重新拟合。

输入原单位float32 X，EEG为200Hz电压、Hb为10Hz双色（HbO/HbR）。每个位置30秒上下文，每支15个整数。EEG词表5116、13bit/整数，两支13bit/s/位置；Hb两支8.5bit/s/位置。Hb保留subject_normalized_Hb，EEG保留各库发布单位身份。固定训练尺度随包保存；没有未编码的逐窗尺度或连续latent。完整阵列位置、缺位置与裁片身份是合法固定元数据。解码只使用整数和冻结状态；源估计仅在编码时接收X。

在具有numpy、scipy、scikit-learn、torch和PyYAML的项目Python环境中，解压进入本目录执行（输出需为新文件）：

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/scripts/apply_joint_tokenizer.py encode --bundle bundle.json --input observations.npz --output my_ids.npz
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/scripts/apply_joint_tokenizer.py decode --bundle bundle.json --input my_ids.npz --output my_waves.npz
```

随包样例全部来自原训练面板：四库各取先前16项追溯之外的下一个fold0完整裁片；Hb沿用已核验训练前例并保留首个静默例。sample_identity.json给出被试、记录、位置、起点及单位。ids.npz和voltage_and_Hb.npz是逐值复现端点；SHA256SUMS固定包内资产。源代码和冻结数值状态均随包携带，不依赖原工作区或数据路径。
''')
    import importlib.metadata
    (package/'requirements.txt').write_text(''.join(f'{n}=={importlib.metadata.version(n)}\n' for n in ('numpy','scipy','scikit-learn','torch','PyYAML')))
    files=sorted(p for p in package.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    (package/'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(package)}\n' for p in files))
    archive=out/'fold0_four_branch_research_reference.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        for p in files+[package/'SHA256SUMS']:z.write(p,str(p.relative_to(package)))
    extracted=out/'independent_unpacked';extracted.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('archive integrity failure')
        z.extractall(extracted)
    for mode,source,dest in [('encode','observations.npz','checked_ids.npz'),('decode','checked_ids.npz','checked_waves.npz')]:
        subprocess.run([sys.executable,'experiments/scripts/apply_joint_tokenizer.py',mode,'--bundle','bundle.json',
            '--input',source,'--output',dest],cwd=extracted,env=env,check=True)
    checks={}
    for got,wanted in [('checked_ids.npz','ids.npz'),('checked_waves.npz','voltage_and_Hb.npz')]:
        with np.load(extracted/got,allow_pickle=False) as a,np.load(package/wanted,allow_pickle=False) as b:
            if set(a)!=set(b):raise ValueError('portable output field mismatch')
            for k in b:np.testing.assert_array_equal(a[k],b[k]);checks[got+'/'+k]='exact'
    write(out/'portable_verification.json',dict(status='passed',archive=archive.name,sha256=sha(archive),bytes=archive.stat().st_size,
        independent_directory=str(extracted.resolve()),new_CLI_processes=2,PYTHONPATH='',checks=checks,
        EEG_four_libraries_original_X_IDs_and_original_integer_audit_waveforms_exact=True,
        Hb_original_IDs_and_original_frozen111_same_batch_waveforms_exact=True,
        historical_full_batch_Hb_arithmetic='separately retained in historical_Hb_batch_arithmetic.json',no_fitting=True,
        no_new_validation_reads=True,no_confirmation_reads=True,no_test_reads=True))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--package-name',default='package')
    args=p.parse_args();args.run=args.run.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    package=prepare(args);portable(args,package)


if __name__=='__main__':main()
