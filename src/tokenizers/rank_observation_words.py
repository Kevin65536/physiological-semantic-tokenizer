"""Four-bit observation words with frozen marginal training quantiles."""
from sklearn.preprocessing import QuantileTransformer

from src.tokenizers.factorized_observation import FactorizedObservationTokenizer


class RankObservationWords(FactorizedObservationTokenizer):
    """Use the original physical features, Lloyd fit and integer interface."""

    def __init__(self, modality, *, seed):
        super().__init__(modality, n_patterns=16, detail_count=0, seed=seed)

    def fit(self, x_train, *, partition='train'):
        if partition != 'train':
            raise ValueError('quantile fitting requires partition=train')
        features = self.pattern_features(x_train)
        self._quantiles = QuantileTransformer(
            n_quantiles=min(257, len(features)), output_distribution='uniform',
            subsample=None, copy=True,
        ).fit(features)
        return super().fit(x_train, partition=partition)

    def _route_features(self, x, physical=None):
        features = self.pattern_features(x) if physical is None else physical
        return self._quantiles.transform(features)
