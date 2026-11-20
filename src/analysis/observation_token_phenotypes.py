"""Can a frozen *observation* word be given a more specific, falsifiable name?

Pure in-memory diagnostic on complete 2-s G0-scaled X patches. No paths, fit,
encoding, task labels, protected split, or reconstruction. The two predictions
and all cutoffs below are fixed, not chosen against the validation outcomes.
Names based on the dictionary describe its construction, not neural sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.observation_tokenizer_evaluation import KEYS, _pairs
from src.tokenizers.factorized_observation import _properties  # exact physical routing coordinates; no fit/encode


_FEATURES = {"eeg": ("0.5-4", "4-8", "alpha", "beta", "30-45", "45-100"),
             "hb": ("hbo_level", "hbr_level", "so", "sr")}
_SHAPE = {"eeg": (1, 400), "hb": (2, 20)}
_SPLITS = ("train", "validation")
_HF = ("little_high_power", "line50_like", "line60_like", "both_lines_like", "diffuse_like", "other_or_mixed")
_HB = ("unresolved", "opposed_o_up", "opposed_o_down", "common_up", "common_down")
_PEOPLE = KEYS[:2]


def _dict_state(codec, mod):
    if getattr(codec, "modality", mod) != mod or getattr(codec, "routing", "pattern") != "pattern":
        raise ValueError("expected the matching frozen physical-pattern dictionary")
    c = np.asarray(codec.pattern_centers_, dtype=float)
    n = np.asarray(codec.pattern_counts_)
    if (n.ndim != 1 or n.dtype.kind not in "iu" or (n < 0).any() or not (n > 0).any()
            or c.shape != (len(n), len(_FEATURES[mod])) or not np.isfinite(c).all()):
        raise ValueError("invalid frozen physical centers/fit counts")
    return c, n


def _panel(frame, wave, mod, split, k):
    cols = KEYS + ["patch", "pattern"]
    if not isinstance(frame, pd.DataFrame) or not set(cols) <= set(frame):
        raise ValueError("need a DataFrame of complete patch identities and pattern IDs")
    for key, value in (("split", split), ("modality", mod)):
        if key in frame and not (frame[key].notna() & frame[key].eq(value)).all():
            raise ValueError("row is outside the requested split/modality")
    z = frame[cols].reset_index(drop=True).copy()
    if z.isna().any().any() or not len(z):
        raise ValueError("empty panel or missing identity")
    for col in ("dataset", "subject", "record", "parent", "position"):
        z[col] = z[col].astype(str)
    for col, bound in (("patch", 15), ("pattern", k)):
        v = z[col].to_numpy()
        if v.dtype.kind not in "iu" or (v < 0).any() or (v >= bound).any():
            raise ValueError(f"{col} must be an in-range integer (0..{bound-1})")
    if z.duplicated(KEYS + ["patch"]).any():
        raise ValueError("duplicate physical patch identity")
    wave = np.asarray(wave)
    if wave.shape != (len(z), *_SHAPE[mod]) or wave.dtype.kind not in "fiu":
        raise ValueError("wave/metadata shape mismatch: expected ordered 2-second observations")
    # Bounded float64 temporaries, never a full-panel time x frequency tensor.
    pieces = []
    for i in range(0, len(z), 512):
        x = np.asarray(wave[i:i+512], dtype=float)
        if not np.isfinite(x).all() or np.abs(x).max() > 1e120:
            raise ValueError("nonfinite or out-of-range G0 observation")
        direct = _properties(x, mod)
        rms = np.sqrt(np.mean((x - x.mean(-1, keepdims=True))**2, axis=(1, 2)))
        part = pd.DataFrame(direct, columns=_FEATURES[mod]).assign(rms=rms)
        extra = _eeg_shape(x) if mod == "eeg" else {"interior": _hb_shape(x)}
        pieces.append(part.assign(**extra))
    z = pd.concat([z, pd.concat(pieces, ignore_index=True)], axis=1)
    # Optional redundant caller columns also detect misordered waves. AC RMS
    # removes each color's own mean. The blind ID does not define these tests.
    for col in ("rms", "alpha", "beta") if mod == "eeg" else ("rms", "so", "sr"):
        if col in frame and not np.allclose(z[col], frame[col], atol=2e-6, rtol=5e-5):
            raise ValueError(f"frame {col} disagrees with same-order wave")
    if mod == "hb":
        i, j = _pairs(z, 1)  # next *actual* patch, all six series identity fields fixed
        y = np.full(len(z), np.nan)
        y[i] = np.isin(z.interior.to_numpy()[j], [1, 2])
        z["next_opposed"] = y
    return z


def _eeg_shape(x):
    a = x[:, 0] - x[:, 0].mean(axis=1, keepdims=True)
    scale = np.abs(a).max(axis=1, keepdims=True)
    a = a / np.where(scale > 0, scale, 1.)
    p = np.abs(np.fft.rfft(a * np.hanning(400)))**2
    # Double non-DC, non-Nyquist power. Bin spacing 0.5 Hz, no zero padding.
    p[:, 1:-1] *= 2
    f = np.arange(201) / 2
    # Mild, fixed three-bin smoothing; four flank medians protect the local
    # log-log slope/intercept from domination by a few flank periodogram bins.
    sm = (p[:, :-2] + 2*p[:, 1:-1] + p[:, 2:])/4
    db = 10*np.log10(np.maximum(sm, np.maximum(sm.max(1, keepdims=True)*1e-12, 1e-300)))
    freq = f[1:-1]
    sides = [(freq >= lo) & (freq <= hi) for lo, hi in [(4., 5.5), (6., 7.5), (14.5, 17.), (17.5, 21.)]]
    # Center at the four median log frequencies, not the mean of all bins
    # (the two upper flanks contain more frequencies).
    lx = np.array([np.median(np.log(freq[s])) for s in sides]); xbar = lx.mean(); lx -= xbar
    ly = np.column_stack([np.median(db[:, s], axis=1) for s in sides])
    slope = (ly * lx).sum(axis=1) / (lx @ lx)
    band = np.flatnonzero((freq >= 8.) & (freq < 13.))
    # Only 0.5-Hz bins from 6.5 to 14 Hz are needed for the shoulders.
    ii = np.arange(band[0]-3, band[-1]+4)
    residual = db[:, ii] - ly.mean(1)[:, None] - slope[:, None]*(np.log(freq[ii])-xbar)
    j = band - ii[0]
    top = residual[:, j]
    local = (top > residual[:, j-1]) & (top >= residual[:, j+1])
    left = np.minimum.reduce([residual[:, j-h] for h in (1, 2, 3)])
    right = np.minimum.reduce([residual[:, j+h] for h in (1, 2, 3)])
    prom = top - np.maximum(left, right)
    hit = local & (top >= 6.) & (prom >= 3.)
    any_hit = hit.any(1)
    which = np.where(hit, top, -np.inf).argmax(1)
    ac = np.ptp(x[:, 0], axis=1) > 0

    h = p[:, 90:]  # 45..100 inclusive: 111 of 200 non-DC bins
    total = h.sum(1)
    share = np.divide(h, total[:, None], out=np.zeros_like(h), where=total[:, None] > 0)
    amount = p[:, 1:].sum(1)
    substantial = ac & (amount > 0) & (total >= .05*amount)
    lines, kinds = [], []
    for line in (50., 60.):
        distance = np.abs(f[90:] - line)
        center = distance <= 1.
        flank = (distance >= 2.) & (distance <= 4.)
        line_share = share[:, center].sum(1)
        # A concentrated peak AND a local excess, not just high total band power.
        lines.append(line_share)
        kinds.append((line_share >= .25) & (share[:, center].max(1) > 4*np.median(share[:, flank], axis=1)))
    concentration = (share**2).sum(1)
    effective = np.divide(1., concentration, out=np.zeros_like(total), where=concentration > 0)
    kind = np.full(len(x), 5, dtype=int)
    kind[(effective >= 20.) & (lines[0]+lines[1] <= .20)] = 4
    a50, a60 = kinds
    kind[a50 & ~a60] = 1; kind[a60 & ~a50] = 2; kind[a50 & a60] = 3
    kind[~substantial] = 0
    return dict(peak=np.where(ac, any_hit.astype(float), np.nan),
                excess_db=np.where(ac, top.max(1), np.nan), peak_hz=np.where(any_hit & ac, freq[band[which]], np.nan),
                peak_prom_db=np.where(any_hit & ac, prom[np.arange(len(x)), which], np.nan),
                high_class=kind, line_share=np.where(substantial, lines[0]+lines[1], np.nan),
                effective_bins=np.where(substantial, effective, np.nan))


def _hb_shape(x):
    """Two interior steps in BOTH colors. Samples 0,1,18,19 have no vote."""
    left, mid, right = (np.median(x[:, :, a:b], axis=2) for a, b in ((2, 7), (8, 12), (13, 18)))
    inc = np.diff(x[:, :, 2:18], axis=2)
    tol = np.maximum(.01, 1.4826*np.median(np.abs(inc-np.median(inc, axis=2, keepdims=True)), axis=2))
    a, b = mid-left, right-mid
    sign = ((a > tol) & (b > tol)).astype(int) - ((a < -tol) & (b < -tol)).astype(int)
    o, r = sign.T
    return np.select([(o == 1) & (r == -1), (o == -1) & (r == 1),
                      (o == 1) & (r == 1), (o == -1) & (r == -1)], [1, 2, 3, 4], default=0)


def _q(values):
    v = np.asarray(values, dtype=float); v = v[np.isfinite(v)]
    return np.quantile(v, [.25, .5, .75]).tolist() if len(v) else None


def _counts(codes, labels):
    return dict(zip(labels, np.bincount(codes, minlength=len(labels)).astype(int).tolist()))


def _support(z, mod, split, centers, counts):
    rows = []
    for p, center in enumerate(centers):
        g = z[z.pattern == p]
        by = {}
        for d, s in g.groupby("dataset", sort=True):
            freq = s.subject.value_counts()
            by[d] = dict(n=len(s), subjects=len(freq), subjects_with_two=int((freq >= 2).sum()),
                         positions={str(k): int(v) for k, v in s.position.value_counts().items()})
        row = dict(modality=mod, split=split, pattern_id=p, fit_n=int(counts[p]),
                   fit_center=dict(zip(_FEATURES[mod], center.tolist())), n=len(g),
                   n_subjects=sum(b["subjects"] for b in by.values()), n_datasets=len(by), by_dataset=by,
                   max_subject_fraction=float(g.groupby(_PEOPLE).size().max()/len(g)) if len(g) else None,
                   direct_q={c: _q(g[c]) for c in (*_FEATURES[mod], "rms")})
        if mod == "eeg":
            row.update(peak_hits=int(g.peak.sum()), peak_evaluable=int(g.peak.notna().sum()),
                       max_excess_db_q=_q(g.excess_db), hit_peak_hz_q=_q(g.peak_hz), hit_prominence_db_q=_q(g.peak_prom_db),
                       high_classes=_counts(g.high_class, _HF),
                       line50_60_share_q=_q(g.line_share), effective_high_bins_q=_q(g.effective_bins))
        else:
            agrees = (np.sign(g.so) == np.sign(center[2])) & (np.sign(g.sr) == np.sign(center[3]))
            row.update(ols_both_center_signs=int(agrees.sum()), interior_classes=_counts(g.interior, _HB),
                       next_opposed_hits=int(g.next_opposed.sum()), next_evaluable=int(g.next_opposed.notna().sum()))
        rows.append(row)
    return rows


def _select(mod, rows, centers, counts):
    """Only dictionary state and TRAIN direct features/occupancy enter selection."""
    fit_mean = np.average(centers, weights=counts, axis=0)
    panel_d = set().union(*(r["by_dataset"] for r in rows))
    roles = ("eeg_alpha", "eeg_high") if mod == "eeg" else ("hb_opposed", "hb_common")
    options = {key: [] for key in roles}
    for r, center in zip(rows, centers):
        p = r["pattern_id"]; by = list(r["by_dataset"].values())
        enough = (r["fit_n"] >= 8 and r["n"] >= 12 and sum(b["subjects_with_two"] for b in by) >= 3
                  and sum(b["subjects_with_two"] > 0 for b in by) >= min(2, len(panel_d)))
        r["train_supported"] = enough
        r["construction_label"] = "unlabelled_sparse_train"
        if not enough:
            continue
        if mod == "eeg":
            diff = center-fit_mean; j = int(np.argmax(diff))
            r["construction_label"] = ("enriched_" + _FEATURES[mod][j] + "_fraction" if diff[j] >= .05
                                       else "mixed_band_fractions")
            for key, i in zip(roles, (2, 5)):
                if diff[i] >= .05 and r["direct_q"][_FEATURES[mod][i]][1] > fit_mean[i]:
                    options[key].append((float(diff[i]), p))
        else:
            lo = ["positive" if r["direct_q"][c][0] > 0 else "negative" if r["direct_q"][c][2] < 0 else "mixed"
                  for c in _FEATURES[mod][:2]]
            so, sr = center[2:]; strength = float(min(abs(so), abs(sr)))
            trend = "no_consistent_strong_ols_pair"
            if strength >= .01 and r["ols_both_center_signs"] >= .6*r["n"]:
                key = "hb_opposed" if np.sign(so) != np.sign(sr) else "hb_common"
                trend = key[3:]+("_o_up" if so > 0 else "_o_down")
                options[key].append((strength, p))
            r["construction_label"] = f"levels_o_{lo[0]}_r_{lo[1]};{trend}"
    rep = {}
    for role, choices in options.items():
        score, code = min(choices, key=lambda v: (-v[0], v[1])) if choices else (None, None)
        rep[role] = dict(modality=mod, pattern_id=code, status="train_selected" if choices else "no_train_representative",
                         selection_score=score, n_eligible_candidates=len(choices))
        if code is not None:
            r = rows[code]
            basis = {k: r[k] for k in ("fit_n", "n", "n_subjects", "n_datasets", "fit_center")}
            if mod == "eeg":
                i = 2 if role == "eeg_alpha" else 5
                basis.update(fit_mean_fraction=float(fit_mean[i]), panel_median_fraction=r["direct_q"][_FEATURES[mod][i]][1])
            else:
                basis["both_ols_signs_fraction"] = r["ols_both_center_signs"]/r["n"]
            rep[role]["train_basis"] = basis
    return rep


def _subject_profiles(z, p, mod):
    # Only validating people carrying a train-selected ID, not a patch dump or
    # an exhaustive ID x feature x subject table. Absence means zero members.
    result = []
    for (d, s), g in z[z.pattern == p].groupby(_PEOPLE):
        row = dict(dataset=d, subject=s, n=len(g))
        if mod == "eeg":
            row.update(peak_hits=int(g.peak.sum()), peak_evaluable=int(g.peak.notna().sum()),
                       high_classes=_counts(g.high_class, _HF))
        else:
            row.update(interior_classes=_counts(g.interior, _HB), next_opposed_hits=int(g.next_opposed.sum()),
                       next_evaluable=int(g.next_opposed.notna().sum()))
        result.append(row)
    return result


def _compare(mod, z, reps):
    eeg = mod == "eeg"
    name = "eeg_alpha_peak" if eeg else "hb_next_opposed"
    p = reps["eeg_alpha" if eeg else "hb_opposed"]["pattern_id"]
    ref = None if eeg else reps["hb_common"]["pattern_id"]
    out = dict(target_id=p, reference="all_other_ids" if eeg else "hb_common", reference_id=ref,
               status="no_train_representative", target_rate=None, reference_rate=None, delta=None, ci95=None,
               overlap_only=None, support=None)
    if p is None or (not eeg and ref is None):
        return name, out, []
    y = z["peak" if eeg else "next_opposed"].to_numpy()
    t = z.pattern.to_numpy() == p; r = ~t if eeg else z.pattern.to_numpy() == ref
    tv, rv = t & np.isfinite(y), r & np.isfinite(y)
    if eeg:
        a = z.alpha.to_numpy(); amp = z.rms.to_numpy(); b = np.log(np.where(amp > 0, amp, 1.))
        u, v = np.floor(a/.05+.5), np.where(amp > 0, np.floor(b/np.log(1.25)+.5), -np.inf)
    else:
        a, b = z.hbo_level.to_numpy(), z.hbr_level.to_numpy()
        u, v = np.floor(a/.25+.5), np.floor(b/.25+.5)
    cells = z[_PEOPLE + ["position"]].assign(u=u, v=v, target_n=t, reference_n=r, target_evaluable=tv,
        reference_evaluable=rv, target_hits=tv & (y == 1), reference_hits=rv & (y == 1),
        ta=tv*a, ra=rv*a, tb=tv*b, rb=rv*b).groupby(_PEOPLE+["position", "u", "v"], observed=True).sum()
    nt, nr = cells.target_evaluable, cells.reference_evaluable
    w = np.minimum(nt, nr); sn, rn = nt.clip(lower=1), nr.clip(lower=1)
    cells = cells.assign(pair_mass=w, shared_cells=(w > 0), target_shared=nt*(w > 0), reference_shared=nr*(w > 0),
        matched_target_hits=w*cells.target_hits/sn, matched_reference_hits=w*cells.reference_hits/rn,
        gap_a=w*(cells.ta/sn-cells.ra/rn).abs(), gap_b=w*(cells.tb/sn-cells.rb/rn).abs())
    people = cells.drop(columns=["ta", "ra", "tb", "rb"]).groupby(level=_PEOPLE).sum()
    ints = ["target_n", "reference_n", "target_evaluable", "reference_evaluable", "target_hits", "reference_hits",
            "shared_cells", "pair_mass", "target_shared", "reference_shared"]
    subjects = []
    for (d, s), row in people.iterrows():
        w = int(row.pair_mass)
        h, b = float(row.matched_target_hits), float(row.matched_reference_hits)
        subjects.append(dict(comparison=name, dataset=d, subject=s, **{k: int(row[k]) for k in ints},
            matched_target_hits=h, matched_reference_hits=b, overlap_delta=(h-b)/w if w else None,
            mean_abs_within_cell_gaps=[float(row.gap_a/w), float(row.gap_b/w)] if w else None,
            qualified=bool(w >= 2 and w >= .5*row.target_n)))

    def coverage(rows):
        good = [s for s in rows if s["qualified"]]
        return dict(**{k: sum(s[k] for s in rows) for k in ints}, panel_subjects=len(rows),
                    target_subjects=sum(s["target_n"] > 0 for s in rows), qualified_subjects=len(good),
                    qualified_pair_mass=sum(s["pair_mass"] for s in good))

    by = {d: coverage([s for s in subjects if s["dataset"] == d]) for d in sorted(set(z.dataset))}
    support = coverage(subjects)
    used = [s for s in subjects if s["qualified"]]
    scoped = {s["dataset"] for s in used}
    support.update(by_dataset=by, qualified_datasets=len(scoped))
    out.update(status="insufficient_common_support", support=support)
    if not used:
        return name, out, subjects
    ds_rates = [np.array([[s["matched_target_hits"]/s["pair_mass"], s["matched_reference_hits"]/s["pair_mass"]]
                          for s in used if s["dataset"] == d]) for d in sorted(scoped)]
    ht, hr = np.mean([q.mean(0) for q in ds_rates], axis=0).tolist()
    out["overlap_only"] = dict(target_rate=ht, reference_rate=hr, delta=ht-hr)
    # No extrapolation: losing most target patches OR most target people in a
    # library vetoes the primary direction (not just a footnote on a regression).
    enough = (len(used) >= 3 and len(scoped) >= min(2, len(by)) and all(
        b["qualified_pair_mass"] >= .5*b["target_n"] and b["qualified_subjects"] >= .5*b["target_subjects"]
        for b in by.values()))
    if enough:
        out.update(**out["overlap_only"], status="direction_observed" if ht > hr else "direction_not_observed")
        if all(len(q) >= 2 for q in ds_rates):
            rng = np.random.default_rng(1731)
            draws = [(q[:, 0]-q[:, 1])[rng.integers(len(q), size=(2000, len(q)))].mean(1) for q in ds_rates]
            out["ci95"] = np.quantile(np.mean(draws, axis=0), [.025, .975]).tolist()
    return name, out, subjects


def run_phenotypes(frames, waves, codecs):
    """Return a finite JSON-compatible dictionary, without any IO or mutations.

    Supply one/both modalities with BOTH train and validation panels, keyed by
    (modality, split), with same-order finite X and DataFrames of KEYS + patch
    (0..14) + frozen pattern. Optional rms/alpha/beta or so/sr are verified.
    A codec needs only its frozen physical pattern_centers_ and pattern_counts_.
    Missing patch rows are allowed; adjacency is a physical-key join, not shift.
    """
    mods = set(codecs)
    if not mods or not mods <= set(_FEATURES):
        raise ValueError("expected eeg and/or hb frozen dictionaries")
    expected = {(m, s) for m in mods for s in _SPLITS}
    if set(frames) != expected or set(waves) != expected:
        raise ValueError("only exact train/validation keys allowed; no test/protected access")
    state = {m: _dict_state(codecs[m], m) for m in mods}
    support, subjects, representatives, comparisons = [], [], {}, {}
    owners = {s: set() for s in _SPLITS}; named = {}
    # Freeze all choices before examining validation outcomes. No extra shape
    # measure, classifier, validation prevalence or performance ranks candidates.
    for split in _SPLITS:
        for m in sorted(mods):
            c, n = state[m]; key = (m, split)
            z = _panel(frames[key], waves[key], m, split, len(n))
            owners[split].update(z[_PEOPLE].drop_duplicates().itertuples(index=False, name=None))
            if owners["train"] & owners["validation"]:
                raise ValueError("dataset-qualified subject crosses train/validation")
            rows = _support(z, m, split, c, n)
            if split == "train":
                representatives.update(_select(m, rows, c, n))
                named[m] = [(r["construction_label"], r["train_supported"]) for r in rows]
            else:
                selected = {v["pattern_id"] for v in representatives.values() if v["modality"] == m}
                for r, (label, ok) in zip(rows, named[m]):
                    r.update(construction_label=label, train_supported=ok)
                    if r["pattern_id"] in selected:
                        r["subject_profiles"] = _subject_profiles(z, r["pattern_id"], m)
                name, result, who = _compare(m, z, representatives)
                comparisons[name] = result; subjects.extend(who)
            support.extend(rows)
    return dict(schema="frozen_observation_token_phenotypes_v1", representatives=representatives, support=support,
                comparisons=comparisons, subjects=subjects, contract=dict(
        selection="Train physical-center labels only. Require >=8 dictionary draws, >=12 panel patches, >=3 dataset-qualified subjects each with >=2 patches, in >=min(2,train dataset count) datasets. EEG rank by center fraction minus occupancy-weighted fit mean (>=0.05 and TRAIN member median above fit mean); Hb rank by smaller absolute center OLS slope (>=0.01 scaled X/s; both signs match on >=60% TRAIN members). Ties: lowest ID. Never substitute a better validating word.",
        distribution="fit_n counts dictionary draws, not different patches/people. direct_q and other _q fields: unweighted patch [Q25,median,Q75], null if undefined. by_dataset positions are patch counts. Only actual panel support, never a population estimate.",
        alpha="H1: alpha word > other IDs for same-patch narrow peak rate. Per 2s: demean, Hann, one-sided 0.5-Hz grid, smooth power [1,2,1]/4. Linear fit in dB vs log(f) to four flank medians at 4–5.5,6–7.5,14.5–17,17.5–21 Hz (relative -120 dB floor). A local max centered in [8,13) must be >=6 dB above local fitted background and drop >=3 dB on EACH side within 1.5 Hz. Excess is max residual in band, not simply alpha fraction. Constant X has no spectrum.",
        high="45–100 includes 111/200 non-DC bins; even flat noise can allocate ~55% of non-DC energy here (Nyquist single). On unsmoothed Hann PSD require >=5% AC power there. 50/60 line-like: +/-1 Hz has >=25% high-band power and peak >4x respective +/-2–4 Hz flank median. Diffuse-like: neither line, combined neighborhoods <=20%, inverse spectral concentration >=20 effective bins. Other/mixed and little high power are not labelled broad. Line-like is not proof of mains; diffuse is not proof of EMG or a brain rhythm.",
        hb="H2: opposed vs common OLS word > for next-patch robust opposed fraction, at similar current signed levels. Within each color, medians of samples [2:7],[8:12],[13:18] must step twice in same direction, each step >max(0.01 training-position-scaled X,1.4826*MAD of adjacent increments on [2:18]). Both colors required; unresolved/flat counts as not opposed. Same-patch interior classes test endpoint sensitivity but are not independent of construction. Next-patch label has disjoint samples within exact KEYS, patch+1<=14; adjacent preprocessed Hb is autocorrelated, not an independent subject or proof of neural activation.",
        matching="Exact dataset, subject and named position; bins floor(v/width+0.5). EEG: rectangular router alpha width 0.05 and log(linear AC RMS) width log(1.25), zeros separate. Hb: both current signed levels width 0.25 G0-scaled X. Within common cells weight both rates by min(n_target_evaluable,n_reference_evaluable); fractional matched_*_hits are sums of weight x cell rate. Gaps are weighted absolute cell-mean differences in binning variables. Equal qualified people per dataset, then equal represented datasets. No control is borrowed across subjects or sites; other-ID rate only refers to these shared strata.",
        support_gate="Qualified person has >=2 pair mass and >=50% of ALL their target patches (incl missing spectrum/successor). Primary requires >=3 such people in >=min(2,validation dataset count) datasets, and >=50% target mass AND target people covered in each dataset with targets. Raw/shared counts include failures. If insufficient, primary rates/delta are null: overlap_only is descriptive and cannot rescue naming. No extrapolation to absent sites/datasets.",
        inference="Direction is a falsifiable descriptive validation prediction: delta=target-reference <=0 is not the predicted direction. CI when defined: stratified paired subject percentile bootstrap (2000, seed=1731), datasets fixed; withheld with <2 qualified people per contributing dataset. No tuning, multiplicity correction, statistical confirmation, artifact attribution or cross-modal coupling claim."))
