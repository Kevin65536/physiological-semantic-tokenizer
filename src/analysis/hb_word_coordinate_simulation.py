"""Generated two-color trajectories with an identified time-linear coefficient."""
import numpy as np
from sklearn.metrics import adjusted_rand_score

from src.tokenizers.factorized_observation import FactorizedObservationTokenizer
from src.tokenizers.partial_rank_hb_words import PartialRankHbWords
from src.tokenizers.rank_observation_words import RankObservationWords


def population(seed, replicate, cohort, subjects, patches, regime, *, noise_sd=0.12):
    """The three cohorts use independent draws. Four regimes share their random variates."""
    if regime not in ('ordinary', 'level_tails', 'slope_tails', 'both_tails'):
        raise ValueError('unknown generator condition')
    rng = np.random.default_rng(np.random.SeedSequence([seed, replicate, cohort]))
    n = subjects*patches
    person = np.repeat(np.arange(subjects), patches)
    offset = rng.normal(size=(subjects, 2))
    level_tail = rng.random(subjects) < 0.12
    if regime in ('level_tails', 'both_tails'):
        offset[level_tail] *= 8
    levels = offset[person] + rng.normal(scale=.65, size=(n, 2))
    levels[:, 1] += .4*levels[:, 0]
    gain = np.exp(rng.normal(0, .3, subjects))
    slopes = rng.normal(size=(n, 2))
    slopes[:, 1] = -.45*slopes[:, 0] + .8*slopes[:, 1]
    slopes *= gain[person, None]
    slope_tail = rng.random(n) < .04
    if regime in ('slope_tails', 'both_tails'):
        slopes[slope_tail] *= 6
    t = (np.arange(20)-9.5)/10
    curve = .2*rng.normal(size=(n, 2))
    wave = levels[:,:,None] + slopes[:,:,None]*t + curve[:,:,None]*(t**2-np.mean(t**2))
    wave += rng.normal(scale=noise_sd, size=wave.shape)
    return wave.astype(np.float32), slopes, person


def codebook(arm, seed):
    if arm == 'standard':
        return FactorizedObservationTokenizer('hb', detail_count=0, seed=seed)
    if arm == 'rank_all':
        return RankObservationWords('hb', seed=seed)
    return PartialRankHbWords(group={'rank_levels':'levels', 'rank_slopes':'slopes'}[arm], seed=seed)


def evaluate_cell(cfg, rep, regime):
    s, p = cfg['scoring_subjects'], cfg['scoring_patches_per_subject']
    xv, yv, _ = population(cfg['generator_seed'], rep, 2, s, p, regime)
    cohorts = [population(cfg['generator_seed'], rep, i, cfg['training_subjects_per_population'],
                          cfg['training_patches_per_subject'], regime)[:2] for i in (0, 1)]
    tv = (np.arange(20)-9.5)/10
    ols = (xv @ tv)/(tv@tv)
    rows = []
    for arm in cfg['arms']:
        assigned = []
        means = []
        for xt, yt in cohorts:
            variance = yt.var(axis=0)
            center = yt.mean(axis=0)
            def score(prediction):
                return float(np.mean((prediction-yv)**2/variance))
            by_seed = []
            for seed in cfg['fitting_seeds']:
                model = codebook(arm, seed).fit(xt, partition='train')
                a = model.encode(xt, 'prototype')['pattern']
                b = model.encode(xv, 'prototype')['pattern']
                if model.rate('prototype')['bits_per_item'] != 4:
                    raise ValueError('mismatched word budget')
                counts = np.bincount(a, minlength=16)
                total = np.zeros((16, 2))
                np.add.at(total, a, yt)
                values = np.divide(total, counts[:,None], where=counts[:,None]>0,
                                   out=np.tile(center, (16,1)))
                means.append([score(values[b]), score(center), score(ols), len(np.unique(b))])
                by_seed.append(b.reshape(s, p))
            assigned.append(by_seed)
        ari = [adjusted_rand_score(a, b) for i, ids in enumerate(assigned[0])
               for a, b in zip(ids, assigned[1][(i+1) % len(assigned[1])])]
        error, constant, observed, coverage = np.mean(means, axis=0)
        rows.append(dict(replicate=rep, regime=regime, arm=arm, ari=float(np.mean(ari)),
                         slope_nmse=error, mean_slope_nmse=constant, continuous_ols_nmse=observed,
                         mean_occupied_words=coverage))
    return rows


def summarize(frame, cfg):
    """The same population draw under all four tail regimes is one independent simulation unit."""
    expected = {(rep, regime, arm) for rep in range(cfg['replicates'])
                for regime in cfg['regimes'] for arm in cfg['arms']}
    actual = set(frame[['replicate','regime','arm']].itertuples(index=False, name=None))
    if expected != actual or len(frame) != len(expected) or not np.isfinite(frame.select_dtypes('number')).all().all():
        raise ValueError('missing, duplicate or nonfinite simulation outcomes')
    rng = np.random.default_rng(28117)
    boot = rng.integers(cfg['replicates'], size=(4000,cfg['replicates']))
    measures = ['ari','slope_nmse','mean_slope_nmse','continuous_ols_nmse','mean_occupied_words']
    def contrast(block):
        group = block.groupby(['replicate','arm'])[measures].mean().unstack('arm')
        contrasts = {}
        for arm in cfg['arms'][1:]:
            stats = {}
            for metric in ('ari','slope_nmse'):
                d = (group[metric][arm]-group[metric]['standard']).to_numpy()
                stats[metric] = dict(difference=float(d.mean()), ci95=np.quantile(d[boot].mean(axis=1),[.025,.975]).tolist())
            contrasts[arm] = stats
        return dict(means=block.groupby('arm')[measures].mean().to_dict('index'), paired=contrasts)
    overall = contrast(frame)
    candidate = cfg['primary_candidate']
    estimates = overall['paired'][candidate]
    c = cfg['selection']
    checks = dict(stability=estimates['ari']['difference'] >= c['min_stability_gain'] and estimates['ari']['ci95'][0] > 0,
                  slope_retention=estimates['slope_nmse']['ci95'][1] <= c['max_slope_nmse_increase'],
                  informative=overall['means'][candidate]['slope_nmse'] < overall['means'][candidate]['mean_slope_nmse'])
    return dict(overall=overall, by_regime={r:contrast(g) for r,g in frame.groupby('regime')},
                checks=checks, passed=all(checks.values()))
