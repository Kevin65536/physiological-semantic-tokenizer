"""Fit on training people and score each frozen representation on the retained panel."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
import multiprocessing
import numpy as np

import run_single_vq_readout as run
from src.analysis.frozen_token_readout import Conditional, ResidualRidge
from src.analysis.tokenizer_pattern_audit import feature_means
from src.analysis.dual_component_tokenizer import base_training_aggregate
from run_dual_component_tokenizer import write_json, read_json


def load_part(task, branch, seed, embeddings):
    mod, ds, role, start, end = task
    rows = run.capacity.PANEL['rows']
    with np.load(run.destination(task)) as z:
        ex, valid = z['examples'], z['valid']
        np.testing.assert_array_equal(ex, run.capacity.PANEL['examples'][mod, role, ds][start:end])
        key = f'{branch}_{seed}_'
        ids = z[key+'id']
        if valid.shape != (len(ex), 15) or ids.shape != (valid.sum(),):
            raise ValueError('patch support differs from the frozen grid')
        person = [run.PEOPLE[(ds, rows[r]['subject'])] for r in ex[:,0]]
        a = dict(y=z[branch+'_target'], id=ids, bg=z['bg'], person=np.repeat(person, valid.sum(1)),
                 hard_z=z[key+'z'], embedding=embeddings[ids])
        if seed == 17:
            a['continuous_z'] = z[branch+'_continuous_z']
        if role == 'validation':
            a['hard_wave'] = z[key+'wave']
            if seed == 17:
                a['continuous_wave'] = z[branch+'_continuous_wave']
    return a, ex, valid


def selection_weights(people, dataset):
    identities, inv, count = np.unique(people, return_inverse=True, return_counts=True)
    ds = np.array([dataset[p] for p in identities])
    _, ds_idx, ds_count = np.unique(ds, return_inverse=True, return_counts=True)
    return 1. / (count[inv] * ds_count[ds_idx[inv]] * len(ds_count))


def train_models(a, k, reps):
    raw, balanced = [Conditional(a['bg'], a['id'], a['person'], a['y'], k, flag) for flag in (False, True)]
    ridge = {rep: ResidualRidge(a['bg'], a[rep], a['y'], raw.background) for rep in reps}
    return raw, balanced, ridge


def fit(a, k, reps, fields, dataset):
    folds = np.full(max(dataset)+1, -1)
    for ds in sorted(set(dataset.values())):
        group = sorted(p for p in np.unique(a['person']) if dataset[p] == ds)
        if len(group) < 3:
            raise ValueError('too few training people for grouped folds')
        folds[group] = np.arange(len(group)) % 3
    strengths = run.PLAN['shrinkage_pseudocounts']
    penalties = run.PLAN['ridge_penalties']
    choices = {name: strengths for name in ('shrunk', 'balanced_shrunk')}
    choices.update({name: penalties for name in reps})
    losses = {name: np.zeros((len(grid), len(fields))) for name, grid in choices.items()}
    # Every person receives the same weight within their dataset, including across folds.
    weights = selection_weights(a['person'], dataset)
    for fold in range(3):
        held = folds[a['person']] == fold
        tr = {key: value[~held] for key,value in a.items()}
        te = {key: value[held] for key,value in a.items()}
        raw, bal, ridge = train_models(tr, k, reps)
        for name, model in [('shrunk', raw), ('balanced_shrunk', bal)]:
            parts = model.read(te['bg'], te['id'])
            for i, strength in enumerate(strengths):
                err = feature_means((model.shrink(*parts, strength)-te['y'])**2, fields)
                losses[name][i] += weights[held] @ err
        for name, model in ridge.items():
            for i, penalty in enumerate(penalties):
                pred = model.predict(te['bg'], te[name], model.coefficients(penalty))
                losses[name][i] += weights[held] @ feature_means((pred-te['y'])**2, fields)
    choice = {name: np.asarray(grid)[losses[name].argmin(0)] for name, grid in choices.items()}
    params = {name: np.repeat(v, [end-start for start,end in fields.values()]) for name,v in choice.items()}
    raw, bal, ridge = train_models(a, k, reps)
    coefs = {}
    for name, model in ridge.items():
        c = np.zeros_like(model.cross)
        for p in np.unique(params[name]):
            columns = params[name] == p
            c[:, columns] = model.coefficients(p)[:, columns]
        coefs[name] = c
    return raw, bal, ridge, coefs, params, dict(selected=choice, cv_loss=losses, folds={p:int(folds[p]) for p in np.unique(a['person'])})


def write_csv(path, rows):
    with path.with_suffix('.tmp').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.tmp').replace(path)


def score(job):
    mod, seed, b, tasks = job
    folder = run.OUT/f'{mod}_{seed}_{b}'
    folder.mkdir(exist_ok=True)
    if (folder/'summary.json').is_file():
        return
    spec = run.PLAN['models'][mod]
    k = spec['size']
    book = run.network(mod,k,seed).branches[b]['quantizer'].codebook.detach().numpy()
    fields = run.attributes(np.zeros((1,400 if mod=='eeg' else 40)),mod)[1]
    data = defaultdict(list)
    for t in tasks:
        if t[2] == 'train':
            part, _, _ = load_part(t,b,seed,book)
            for key, v in part.items():
                data[key].append(v)
    train = {key: np.concatenate(values) for key, values in data.items()}
    del data
    reps = [name for name in ('hard_z','embedding','continuous_z') if name in train]
    ds_of = {code: ds for (ds, subject), code in run.PEOPLE.items()}
    raw, bal, ridge, coefs, params, selection = fit(train,k,reps,fields,ds_of)
    selection['folds'] = [dict(dataset=ds,subject=sub,fold=selection['folds'][p])
                          for (ds,sub),p in run.PEOPLE.items() if p in selection['folds']]
    write_json(folder/'training_selection.json', selection)
    sites = {i:p for p,i in run.SITES[mod].items()}
    cells = []
    np.testing.assert_array_equal(raw.joint.keys,bal.joint.keys)
    for joint, n, people in zip(raw.joint.keys,raw.joint.count,bal.joint.count):
        bg, word = divmod(int(joint), k)
        site, rms = divmod(bg,4)
        ds, location = sites[site]
        cells.append(dict(dataset=ds,position=location,rms_bin=rms,id=word,train_patches=n,train_people=people))
    write_csv(folder/'training_support.csv', cells)
    del cells, train
    examples, errors = [], []
    coverage = defaultdict(lambda: np.zeros(3, dtype=int))
    total_grid = defaultdict(int)
    for t in tasks:
        if t[2] != 'validation':
            continue
        a, ex, valid = load_part(t,b,seed,book)
        parts = raw.read(a['bg'], a['id'])
        balanced = bal.read(a['bg'], a['id'])
        methods = dict(background=parts[0], original=parts[0]+parts[1],
                       shrunk=raw.shrink(*parts,params['shrunk']), balanced_background=balanced[0],
                       balanced_shrunk=bal.shrink(*balanced,params['balanced_shrunk']))
        methods.update({name: model.predict(a['bg'],a[name],coefs[name]) for name,model in ridge.items()})
        methods.update({name:a[name] for name in ('hard_wave','continuous_wave') if name in a})
        losses = np.stack([feature_means((p-a['y'])**2,fields) for p in methods.values()],axis=1)
        if not np.isfinite(losses).all():
            raise ValueError('invalid error on complete patch')
        v = np.zeros((*valid.shape,*losses.shape[1:]))
        v[valid] = losses
        counts = valid.sum(1)[:,None,None]
        errors.append(np.divide(v.sum(1),counts,out=np.full_like(v[:,0],np.nan),where=counts>0))
        examples.append(ex)
        support = run.PLAN['support']
        enough = (parts[2]>=support['min_train_patches']) & (balanced[2]>=support['min_train_people'])
        coverage[t[1]] += [np.count_nonzero(enough), np.count_nonzero(~enough & (parts[2]>0)), np.count_nonzero(parts[2]==0)]
        total_grid[t[1]] += valid.size
    examples = np.concatenate(examples)
    # The hierarchy owner requires its NMSE slot. Here the contents are attribute MSE.
    agg = base_training_aggregate(dict(nmse=np.concatenate(errors)),examples,run.capacity.PANEL['rows'])
    stats = agg['metrics']['nmse']
    src = run.capacity.summary(mod,spec['dimension'],k,seed)
    names = list(methods)
    for scope in ('macro','by_dataset'):
        scores = stats[scope]
        old = src['metrics'][b+'_word'][scope]
        if scope == 'macro':
            scores,old = {'macro':scores},{'macro':old}
        for ds in scores:
            np.testing.assert_allclose(np.array(scores[ds])[:2,:len(old[ds][0])],old[ds],rtol=2e-5,atol=1e-10)
    if agg['subjects']!=16 or agg['inputs']!=run.capacity.PLAN['support'][mod+'_positions']:
        raise ValueError('incomplete retained panel')
    records = []
    scores = [('macro','',stats['macro']), *((d,'',v) for d,v in stats['by_dataset'].items()),
              *((s['dataset'],s['subject'],s['value']) for s in stats['by_subject'])]
    for ds, subject, values in scores:
        values = np.array(values)
        for i, method in enumerate(names):
            base = names.index('balanced_background') if method.startswith('balanced_') else 0
            for j, attr in enumerate(fields):
                mse, baseline = values[i,j], values[base,j]
                records.append(dict(dataset=ds,subject=subject,method=method,attribute=attr,mse=mse,
                                    background_mse=baseline,gain=1-mse/baseline if baseline>0 else None,
                                    mse_minus_original=mse-values[names.index('original'),j]))
    write_csv(folder/'scores.csv', records)
    write_json(folder/'summary.json', dict(status='completed',branch=b,seed=seed,modality=mod,
               positions=agg['inputs'],people=agg['subjects'],coverage_order=['adequate','sparse','unseen'],
               coverage=coverage,nominal_patch_slots=total_grid,attributes=list(fields),methods=names,
               macro_mse=stats['macro'],training_people=len(selection['folds']),historical_readout_replayed=True))


def score_all(tasks, workers):
    jobs = [(m,s,b,[t for t in tasks if t[0]==m]) for m,spec in run.PLAN['models'].items()
            for s in spec['seeds'] for b in 'SO']
    with ProcessPoolExecutor(min(workers,len(jobs)),mp_context=multiprocessing.get_context('fork')) as pool:
        list(pool.map(score, jobs))
    results = [read_json(run.OUT/f'{m}_{s}_{b}/summary.json') for m,s,b,_ in jobs]
    write_json(run.OUT/'summary.json',dict(status='completed',experiment_id=run.PLAN['experiment_id'],
               endpoint='held-out attribute MSE on the ordinary development panel',results=results))
    summarize_scores()


def summarize_scores():
    """Collect the macro gains and descriptive paired intervals from retained person scores."""
    import pandas as pd

    gain_rows, paired_rows, support_rows = [], [], []
    comparisons = [('original','shrunk'), ('balanced_background','balanced_shrunk'),
                   ('hard_z','embedding'), ('embedding','hard_wave'), ('continuous_wave','hard_wave')]
    results = read_json(run.OUT/'summary.json')['results']
    expected = {(m,s,b) for m,spec in run.PLAN['models'].items() for s in spec['seeds'] for b in 'SO'}
    actual = {(r['modality'], r['seed'], r['branch']) for r in results if r['status']=='completed'}
    if actual != expected or len(results) != len(expected):
        raise ValueError('incomplete model/seed/branch matrix')
    for result in results:
        m, s, b = result['modality'], result['seed'], result['branch']
        identity = dict(modality=m,seed=s,branch=b)
        frame = pd.read_csv(run.OUT/f'{m}_{s}_{b}'/'scores.csv',keep_default_na=False)
        for attr, group in frame.groupby('attribute'):
            macro = group[group.dataset=='macro'].set_index('method')
            gain_rows.extend(dict(identity,attribute=attr,method=method,gain=row.gain,
                                  mse=row.mse,background_mse=row.background_mse)
                             for method,row in macro.iterrows())
            people = group[group.subject!=''].pivot(index=['dataset','subject'],columns='method',values='mse')
            if len(people)!=16 or people.isna().any().any():
                raise ValueError('expected 16 paired ordinary development people')
            for left,right in comparisons:
                if left not in people:
                    continue
                delta = people[left] - people[right]
                rng = np.random.default_rng(712)
                draws = [rng.choice(group.values,(10000,len(group)),replace=True).mean(1)
                         for _, group in delta.groupby(level='dataset')]
                low, high = np.quantile(np.mean(draws,axis=0),[.025,.975])
                point = delta.groupby(level='dataset').mean().mean()
                np.testing.assert_allclose(point,macro.loc[left,'mse']-macro.loc[right,'mse'],atol=1e-12)
                paired_rows.append(dict(identity,attribute=attr,left=left,right=right,mse_left_minus_right=point,
                                        ci_low=low,ci_high=high,right_better_people=int((delta>0).sum()),people=len(delta)))
        cov = result['coverage']
        for ds, values in [*cov.items(), ('pooled', np.sum(list(cov.values()),axis=0))]:
            total = int(sum(values))
            nominal = result['nominal_patch_slots']
            support_rows.append(dict(identity,dataset=ds,nominal=sum(nominal.values()) if ds=='pooled' else nominal[ds],
                                     valid=total,adequate=values[0],sparse=values[1],unseen=values[2],
                                     adequate_fraction=values[0]/total,sparse_fraction=values[1]/total,unseen_fraction=values[2]/total))
    write_csv(run.OUT/'macro_gains.csv', gain_rows)
    write_csv(run.OUT/'paired_differences.csv', paired_rows)
    write_csv(run.OUT/'support.csv', support_rows)
    write_json(run.OUT/'verification.json',dict(status='verified',historical_readout='replayed by dataset and macro',
               expected_readouts=len(expected),completed_readouts=len(results),
               paired_interval='Descriptive post-run 95% percentile intervals; 10000 paired person resamples within dataset, equal datasets, fixed models and readouts, no multiplicity adjustment. Positive MSE left minus right favors right.'))
