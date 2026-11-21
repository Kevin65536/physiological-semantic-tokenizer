"""Hb words with train-quantile routing restricted to levels or slopes."""
from src.tokenizers.rank_observation_words import RankObservationWords


class PartialRankHbWords(RankObservationWords):
    def __init__(self, *, group, seed):
        if group not in ('levels', 'slopes'):
            raise ValueError('specify levels or slopes')
        super().__init__('hb', seed=seed)
        self.group = group

    def _route_features(self, x, physical=None):
        features = self.pattern_features(x) if physical is None else physical
        ranked = super()._route_features(x, features)
        preserve = slice(2, 4) if self.group == 'levels' else slice(0, 2)
        ranked[:, preserve] = features[:, preserve]
        return ranked
