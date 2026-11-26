import sys
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
import run_aligned_tokenizer_optimization as run


def test_inner_subjects_disjoint_stable_and_reserved_excluded():
    rows=[dict(dataset=str(d),subject=str(s),split='train',inputs={'score':[{'positions':[0]}]})
          for d in range(4) for s in range(15)]
    rows.append(dict(dataset='0',subject='reserved',split='validation',inputs={'score':[{'positions':[0]}]}))
    fit,held=run.inner_subjects(rows)
    assert len(fit)==48 and len(held)==12 and fit.isdisjoint(held)
    assert ('0','reserved') not in fit|held
    assert run.inner_subjects(rows[::-1])==(fit,held)


def test_decoder_training_freezes_assignments_and_integer_path():
    cfg,task=run.settings('hb')
    model=run._base_training_model(cfg,task).train()
    for b in model.branches.values():
        b['quantizer'].initialized.fill_(True)
    run.frozen_decoder(model)
    x=torch.randn(2,15,40)
    with torch.no_grad():
        previous=model(x)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-5)
    model(x)['sum'].square().mean().backward()
    opt.step()
    model.eval()
    after=model(x)
    for name in ['S','O']:
        assert torch.equal(previous['ids'][name],after['ids'][name])
        assert torch.equal(model.decode_ids(after['ids'])[name],after[name])
        assert model.branches[name]['quantizer'].update_count==0
    assert any(p.grad is not None for p in model.parameters())


def test_qc_moments_read_fit_record_and_exclude_invalid_positions(tmp_path):
    import json
    import numpy as np
    folder=tmp_path/'eeg'
    folder.mkdir()
    r=dict(folder=str(folder),parent='p',subject='person',split='train',modality='eeg',rate=1,
           positions=['A','B'],channels=1,inputs={'score':[dict(start_s=0,positions=[0])]},
           unavailable_inputs={'score':[{'new_qc_exclusions':1}]})
    arr=np.arange(60).reshape(30,2,1)
    for k in 'xso':
        np.save(folder/(k+'.npy'),arr)
    np.save(folder/'mask.npy',np.ones_like(arr,dtype=bool))
    moments=run.qc_moments(r)
    assert moments['x']['count']==[[30.],[0.]]
    assert moments['dx']['count']==[[29.],[0.]]
    assert moments['dx']['second']==[[116.],[0.]]
    (tmp_path/'record.json').write_text(json.dumps({'rows':[dict(r,moments=moments)]}))
    r['unavailable_inputs']['score'][0]['new_qc_exclusions']=0
    assert run.qc_moments(r)==moments


def test_review_pairs_real_holdout_members_with_matched_alternatives(tmp_path,monkeypatch):
    import numpy as np
    import score_aligned_optimization as review
    rows=[];groups={};scales={'positions':{'hb':{'study':{'P':dict(scale=1,target_variance={k:[1,1] for k in 'xso'},difference_variance={k:[1,1] for k in 'xso'})}}}}
    for split in ('train','validation'):
        triples=[]
        for i in range(4):
            for index in range(2):
                folder=tmp_path/f'{split}_{i}_{index}'
                folder.mkdir()
                a=np.ones((300,1,2),np.float32)*(1+index*.1)
                for k in 'xso':
                    np.save(folder/(k+'.npy'),a)
                np.save(folder/'mask.npy',np.ones_like(a,dtype=bool))
                triples.append((len(rows),0,0))
                rows.append(dict(dataset='study',subject=f'{split}{i}',record=str(index),parent=str(folder),
                                 positions=['P'],modality='hb',folder=str(folder),split=split,rate=10,channels=2,
                                 inputs={'score':[dict(start_s=0,positions=[0])]}))
        groups[split]=triples
    def fake(model,raw,device):
        code=(raw['x'][:,:,0]>1.05).astype(int)
        return None,np.stack([code,code],axis=-1)
    monkeypatch.setattr(review,'infer',fake)
    review.patternfig('hb','synthetic',rows,scales,None,groups,'cpu',tmp_path)
    outcomes=run.read_json(tmp_path/'word_support.json')
    assert len(outcomes)==4 and all(r['people']==4 and r['pairs']==60 for r in outcomes)
    assert len(list(tmp_path.glob('word_*.png')))==4


def test_fixed_integer_partners_selected_on_training_support(tmp_path):
    import numpy as np
    import pandas as pd
    from score_aligned_optimization import partner_figures
    meta=pd.DataFrame(dict(dataset=['d']*6,subject=['A','B','C']*2))
    codes=np.array([[1,4]]*3+[[1,5]]*3)
    waves=np.ones((6,3,40),dtype=np.float32)
    parts={'train':(meta,waves,codes),'validation':(meta,waves,codes)}
    partner_figures('hb','synthetic',parts,tmp_path)
    entries=run.read_json(tmp_path/'partner_support.json')
    assert {r['partner_word'] for r in entries if r['fixed_branch']=='S'}=={4,5}
    assert all(r['inner_people']==3 for r in entries if r['fixed_branch']=='S')
    assert (tmp_path/'fixed_S_partners.png').is_file()


def test_shift_review_uses_only_exact_common_samples(tmp_path,monkeypatch):
    import numpy as np
    import pandas as pd
    import score_aligned_optimization as review
    scales={'positions':{'hb':{'d':{'P':dict(scale=1,target_variance={k:[1,1] for k in 'xso'},difference_variance={k:[1,1] for k in 'xso'})}}}}
    row=dict(modality='hb',dataset='d',subject='held',split='validation',folder=str(tmp_path),rate=10,
             channels=2,positions=['P'],inputs={'train':[dict(start_s=j,positions=[0]) for j in [0,2]]})
    a=np.arange(640,dtype=np.float32).reshape(320,1,2)
    for k in 'xso':np.save(tmp_path/(k+'.npy'),a)
    np.save(tmp_path/'mask.npy',np.ones_like(a,dtype=bool))
    def fake(model,raw,device):
        x=raw['x'][:,:,0].astype(int)
        return None,np.stack([x,x],axis=-1)
    monkeypatch.setattr(review,'infer',fake)
    review.shift_review([row],[(0,0,0)],scales,None,'cpu',tmp_path)
    people=pd.read_csv(tmp_path/'natural_shift.csv')
    assert set(people.patch_pairs)=={2,12} and (people.id_match==1).all()
