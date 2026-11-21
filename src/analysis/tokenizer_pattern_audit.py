"""Training-only categorical readouts and validation conditional distributions for G0."""
from collections import defaultdict

import numpy as np


def attributes(y, modality):
    """Rows are 2-s patches in time-major, color-minor G0 input coordinates."""
    if modality == 'eeg':
        centered = y - y.mean(-1, keepdims=True)
        window = np.hanning(400)
        power = abs(np.fft.rfft(centered * window, axis=-1))**2 * 2 / (400 * (window**2).sum())
        hz = np.fft.rfftfreq(400, 1/200)
        bands = [(1,4), (4,8), (8,13), (13,30), (30,45)]
        result = dict(voltage=y, centered=centered)
        for lo,hi in bands:
            band = (hz >= lo) & ((hz <= hi) if hi == 45 else (hz < hi))
            result[f'power_{lo}_{hi}'] = power[...,band].sum(-1, keepdims=True)
        result['rms_envelope'] = np.sqrt(np.mean(y.reshape(*y.shape[:-1], 8, 50)**2, axis=-1))
    else:
        wave = y.reshape(*y.shape[:-1], 20, 2)
        t = (np.arange(20) - 9.5) / 10
        # Project onto the orthogonalized quadratic, then convert its coefficient to d²y/dt².
        quad = t**2 - np.mean(t**2)
        slope = np.einsum('...tc,t->...c', wave, t / (t@t))
        curvature = np.einsum('...tc,t->...c', wave, 2*quad / (quad@quad))
        result = {}
        for c,color in enumerate(('HbO','HbR')):
            result.update({color+'_wave':wave[...,c],color+'_level':wave[...,c].mean(-1,keepdims=True),
                           color+'_slope':slope[...,c:c+1],color+'_curvature':curvature[...,c:c+1]})
    offsets = np.cumsum([0]+[a.shape[-1] for a in result.values()])
    return np.concatenate(list(result.values()), axis=-1), dict(zip(result,zip(offsets[:-1],offsets[1:])))


def feature_means(values, fields):
    return np.stack([values[...,a:b].mean(axis=-1) for a,b in fields.values()], axis=-1)


class GroupTotals:
    """Accumulate wide feature vectors once per observed categorical group in a batch."""
    def __init__(self, groups, width):
        self.count = np.zeros(groups, dtype=np.int64)
        self.total = np.zeros((groups,width),dtype=np.float64)

    def add(self, keys, y):
        if not len(keys):
            return
        order = np.argsort(keys)
        labels, first, counts = np.unique(keys[order], return_index=True, return_counts=True)
        self.count[labels] += counts
        self.total[labels] += np.add.reduceat(y[order], first, axis=0, dtype=np.float64)

    def mean(self, keys):
        return self.total[keys] / self.count[keys,None].clip(1)


class PatternAudit:
    def __init__(self, codes, backgrounds, datasets, width, fields):
        self.k, self.fields = codes, fields
        self.local = GroupTotals(codes,width)
        self.background = GroupTotals(backgrounds,width)
        self.joint = GroupTotals(backgrounds*codes,width)
        self.valid = GroupTotals(len(datasets)*codes,width)
        self.second = GroupTotals(len(datasets)*codes,len(fields))
        self.id_error = GroupTotals(len(datasets)*codes,len(fields))
        self.support = defaultdict(lambda: np.zeros(codes,dtype=np.int64))

    def fit(self, ids, bg, y):
        self.local.add(ids,y)
        self.background.add(bg,y)
        self.joint.add(bg*self.k+ids,y)

    def read(self, ids, bg):
        global_mean = self.local.total.sum(0)/self.local.count.sum()
        local = np.where(self.local.count[ids,None]>0,self.local.mean(ids),global_mean)
        background = np.where(self.background.count[bg,None]>0,self.background.mean(bg),global_mean)
        joint = bg*self.k+ids
        conditional = np.where(self.joint.count[joint,None]>0,self.joint.mean(joint),background)
        return background,local,conditional

    def score(self, ids, bg, dataset, y, decoded):
        background,local,joint = self.read(ids,bg)
        errors = np.stack([feature_means((p-y)**2,self.fields)
                           for p in (background,local,joint,decoded)],axis=1)
        keys = dataset*self.k + ids
        self.valid.add(keys,y)
        self.second.add(keys,feature_means(y*y,self.fields))
        self.id_error.add(keys,errors[:,1])
        return errors

    def support_batch(self, ids, valid, units, split):
        for unit in set(units):
            matches = np.array([u == unit for u in units])
            codes = ids[matches][valid[matches]]
            self.support[(split,*unit)] += np.bincount(codes,minlength=self.k)

    def atlas(self, datasets, cfg):
        support = {}
        for split in ('train','validation'):
            counts = [v for (s,*_),v in self.support.items() if s==split]
            by_subject = defaultdict(lambda:np.zeros(self.k,dtype=np.int64))
            for (s,d,sub,_),count in self.support.items():
                if s==split:
                    by_subject[d,sub] += count
            support[split] = dict(patches=np.sum(counts,axis=0),records=np.sum(np.array(counts)>0,axis=0),
                                  subjects=np.sum(np.array(list(by_subject.values()))>0,axis=0),
                                  per_subject=[dict(dataset=d,subject=s,patches=v) for (d,s),v in by_subject.items()])
        low = ((support['train']['subjects']<cfg['minimum_train_subjects']) |
               (self.local.count<cfg['minimum_train_patches']))
        keys = np.arange(len(datasets)*self.k)
        variance = self.second.mean(keys)-feature_means(self.valid.mean(keys)**2,self.fields)
        values = {}
        for name,arr in dict(conditional_variance=np.maximum(0,variance),frozen_ID_mse=self.id_error.mean(keys)).items():
            values[name] = np.where(self.valid.count[:,None]>0,arr,np.nan).reshape(len(datasets),self.k,-1)
        patches = support['validation']['patches']
        count = self.valid.count.reshape(len(datasets),self.k).sum(0)
        sums = self.valid.total.reshape(len(datasets),self.k,-1).sum(0)
        second = self.second.total.reshape(len(datasets),self.k,-1).sum(0)
        pooled = np.maximum(0,second/count[:,None].clip(1)-feature_means((sums/count[:,None].clip(1))**2,self.fields))
        return dict(support=support,low_training_support=low,low_support_codes=int(low.sum()),
                    low_support_validation_patches=int(patches[low].sum()),complete_validation_patches=int(patches.sum()),
                    pooled_validation_conditional_variance=np.where(count[:,None]>0,pooled,np.nan),
                    by_dataset=dict(dataset_order=datasets,patches=self.valid.count.reshape(len(datasets),self.k),**values))


def match_training_patterns(left, right, fields):
    """One-to-one assignment by training-conditional shape, independent of labels."""
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    cost = np.zeros((len(left),len(right)))
    for a,b in fields.values():
        x,y = left[:,a:b],right[:,a:b]
        scale = np.var(np.concatenate((x,y)),axis=0).mean()
        if scale>0:
            cost += cdist(x,y,'sqeuclidean') / ((b-a)*scale)
    i,j = linear_sum_assignment(cost)
    return i,j,cost[i,j]
