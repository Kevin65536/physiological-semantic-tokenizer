"""Perturb training subjects while preserving the original hierarchical patch draw."""
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.analysis.observation_tokenizer_evaluation import KEYS, hierarchy_weights, paired_contrast


def training_draw(frame, weights, condition, index, cfg):
    """Use one shared index vector for both arms and all initializations."""
    p = weights.copy()
    seed = cfg['full_draw_seed']
    if condition == 'bootstrap':
        rng = np.random.default_rng(cfg['bootstrap_seed'] + int(index))
        for _, block in frame.groupby('dataset', sort=True):
            subjects = np.sort(block.subject.unique())
            counts = np.bincount(rng.integers(len(subjects), size=len(subjects)), minlength=len(subjects))
            p[block.index] *= block.subject.map(dict(zip(subjects, counts))).to_numpy()
        seed = cfg['bootstrap_seed'] + 1000 + int(index)
    elif condition == 'omit':
        if index not in set(frame.dataset):
            raise ValueError('unknown omitted dataset')
        p[frame.dataset == index] = 0
    elif condition != 'full':
        raise ValueError('unknown perturbation')
    if condition != 'full':
        p /= p.sum()
    return np.random.default_rng(seed).choice(len(p), cfg['draw_size'], p=p, replace=True)


def word_support(frame, ids):
    weights = hierarchy_weights(frame)
    rows = []
    for word in range(16):
        mask = ids == word
        subset = frame.loc[mask]
        rows.append(dict(word=word, patches=int(mask.sum()), mass=float(weights[mask].sum()),
                         subjects=len(subset[KEYS[:2]].drop_duplicates()), datasets=subset.dataset.nunique()))
    return rows


def subject_agreement(groups, reference, current):
    rows = []
    for (ds, subject), ix in groups.items():
        a, b = reference[ix], current[ix]
        rows.append(dict(dataset=ds, subject=subject, patches=len(ix), ari=adjusted_rand_score(a, b),
                         ref_words=len(np.unique(a)), refit_words=len(np.unique(b))))
    return rows


def summarize(scores, probes, support, cfg):
    decision = {}
    for modality, table in scores.groupby('modality'):
        # Refit repetitions are dependent observations for each validation person.
        people = (table[table.condition == 'bootstrap']
                  .groupby(['arm', 'dataset', 'subject'], as_index=False).ari.mean()
                  .assign(level='subject', bits_per_patch=4).rename(columns={'arm': 'method'}))
        dif = paired_contrast(people, 'ari', ('standard', 4), ('rank', 4))
        def avg(rows):
            return rows.groupby(['arm', 'dataset', 'subject']).ari.mean().groupby(
                ['arm', 'dataset']).mean().groupby('arm').mean().to_dict()
        s = support[support.modality == modality]
        c = cfg['selection']
        usage = {}
        for arm, group in s.groupby('arm'):
            p = group.loc[group.mass > 0, 'mass'].to_numpy()
            usage[arm] = dict(nonempty=len(p), effective_words=float(np.exp(-(p*np.log(p)).sum())),
                              supported=int(((group.subjects >= c['min_support_subjects'])
                                             & (group.datasets >= c['min_support_datasets'])).sum()))
        p = probes[(probes.modality == modality) & (probes.level == 'macro')].set_index(['arm', 'lag_patches'])
        gains = {arm: p.loc[(arm, 0), 'pattern_gain'] for arm in ('standard', 'rank')}
        checks = dict(stability_gain=dif['candidate_minus_reference'] >= c['min_mean_ari_gain'],
                      positive_interval=dif['ci95'][0] > c['min_ari_difference_lower_bound'],
                      all_words_used=usage['rank']['nonempty'] == 16,
                      cross_subject_coverage=usage['rank']['supported'] >= usage['standard']['supported'],
                      positive_readout=gains['rank'] > 0,
                      retains_readout=gains['rank'] >= gains['standard']-c['max_current_attribute_gain_loss'],
                      beats_rotated=p.loc[('rank', 0), 'pattern'] < p.loc[('rank', 0), 'rolled_pattern'])
        only_boot = table[table.condition == 'bootstrap']
        omissions = {ds: avg(sub) for ds, sub in table[table.condition == 'omit'].groupby('dataset')}
        decision[modality] = dict(passed=all(checks.values()), checks=checks, bootstrap_ari=avg(only_boot),
                                  paired_ari=dif, validation_coverage=usage, current_attribute_gains=gains,
                                  current_blind_gain=p.loc[('rank', 0), 'wave_kmeans_gain'],
                                  next_attribute_gains={arm: p.loc[(arm, 1), 'pattern_gain'] for arm in gains},
                                  initialization_only_ari=avg(table[(table.condition == 'full') & (table.seed != 17)]),
                                  omit_training_dataset_ari=omissions,
                                  degenerate_bootstrap_subject_comparisons=int(((only_boot.ref_words == 1)
                                                                              | (only_boot.refit_words == 1)).sum()))
    return decision
