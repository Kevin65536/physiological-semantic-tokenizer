"""Training-only sparse categorical readouts and background-adjusted linear probes."""
import numpy as np


class Means:
    def __init__(self, keys, values):
        order = keys.argsort(kind='stable')
        self.keys, first, self.count = np.unique(keys[order], return_index=True, return_counts=True)
        self.values = np.add.reduceat(values[order], first, axis=0, dtype=np.float64) / self.count[:, None]
        self.global_mean = np.average(self.values, weights=self.count, axis=0)

    def lookup(self, keys, fallback=None):
        idx = np.searchsorted(self.keys, keys).clip(max=len(self.keys)-1)
        seen = self.keys[idx] == keys
        fallback = self.global_mean if fallback is None else fallback
        return np.where(seen[:, None], self.values[idx], fallback), np.where(seen, self.count[idx], 0)


class Conditional:
    def __init__(self, bg, ids, people, y, k, balanced):
        self.k = k
        joint = bg*k + ids
        span = int(people.max()) + 1
        def table(keys):
            if not balanced:
                return Means(keys, y)
            person_cells = Means(keys*span + people, y)
            result = Means(person_cells.keys//span, person_cells.values)
            result.global_mean = Means(people, y).values.mean(0)
            return result
        self.background = table(bg)
        self.joint = table(joint)

    def read(self, bg, ids):
        base = self.background.lookup(bg)[0]
        value, count = self.joint.lookup(bg*self.k+ids, base)
        return base, value-base, count

    @staticmethod
    def shrink(base, delta, count, strength):
        n = count[:, None]
        return base + delta * n / np.maximum(n + strength, 1)


class ResidualRidge:
    """Ridge on within-background residuals, standardized on fit subjects."""
    def __init__(self, bg, x, y, background):
        self.background = background
        self.centers = Means(bg, x)
        residual = x - self.centers.lookup(bg)[0]
        self.scale = residual.std(0)
        self.scale[self.scale < 1e-10] = 1
        residual /= self.scale
        self.gram = residual.T @ residual / len(y)
        self.cross = residual.T @ (y-background.lookup(bg)[0]) / len(y)

    def coefficients(self, penalty):
        return np.linalg.solve(self.gram + np.eye(len(self.gram))*penalty, self.cross)

    def predict(self, bg, x, coefficients):
        z = (x-self.centers.lookup(bg)[0])/self.scale
        return self.background.lookup(bg)[0] + z @ coefficients
