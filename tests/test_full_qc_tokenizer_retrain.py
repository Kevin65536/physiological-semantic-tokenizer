"""Exercise the four-array EEG training and reload path without measured inputs."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
import retrain_qc_eeg_innovations as fit


def test_source_books_calibration_and_reloaded_words(tmp_path, monkeypatch):
    rng=np.random.default_rng(120)
    names=[f'P{i}' for i in range(12)]
    datasets=['synthetic']
    group=[]
    for _ in range(3):
        x=rng.normal(size=(6000,len(names)))
        group.append(({'weight':1/3},{'x':x,'s':x*.4,'o':x*.6},np.ones(12,bool),np.ones(12,bool)))
    settings=dict(fit.CFG['eeg'],sample_patches_per_dataset_role_branch=64,pilot_capacity=8,residual_capacity=16)
    monkeypatch.setitem(fit.CFG,'eeg',settings)
    monkeypatch.setattr(fit,'DIR',tmp_path)
    monkeypatch.setattr(fit,'DATASETS',datasets)
    monkeypatch.setattr(fit,'data',lambda ds,stage:iter(group))
    refs={k:{'target_variance':{'s':[1.0],'o':[1.0]}} for k in names}
    monkeypatch.setattr(fit,'STATE',dict(names={'synthetic':names},scales={'positions':{'eeg':{'synthetic':refs}}}))
    fit.save(tmp_path/'frequency.npz',S=np.full(3001,.35),O=np.full(3001,.65))
    fit.source('synthetic')
    for stage in ['pilot','calibration','residual','calibrate']:
        fit.fit_arrays(('synthetic',stage))
        if stage in ('pilot','residual'):
            for b in 'SO':
                for j in range(5):
                    fit.fit_book((stage,b,j))
    fit.export()
    assert (tmp_path/'fit_complete.json').is_file()
