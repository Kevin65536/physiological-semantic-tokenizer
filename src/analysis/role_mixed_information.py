"""Exact fixed-role stitching of already cross-fitted categorical readout errors."""

from collections import defaultdict

import numpy as np

FIELDS = ("power_4_8", "power_8_13", "rms_envelope")


def mix_by_role(high, low, roles, mask):
    high, low, roles = np.asarray(high), np.asarray(low), np.asarray(roles)
    if (
        high.shape != low.shape
        or high.shape[0] != len(roles)
        or not 0 <= int(mask) < 32
    ):
        raise ValueError("same held population and five-bit fixed role mask required")
    use_high = ((int(mask) >> roles) & 1).astype(bool)
    return np.where(use_high.reshape((-1,) + (1,) * (high.ndim - 1)), high, low)


class RoleInformationPanel:
    """Linear hierarchy, retaining each original finite position and independent person."""

    def __init__(self, high, low, high_mass, low_mass, bank, rows, selection=None):
        high, low = np.asarray(high), np.asarray(low)
        n = len(bank["examples"])
        if high.shape != (n, 2, 3) or low.shape != high.shape:
            raise ValueError("per-window background and full ID errors required")
        # Unseen backgrounds use a global mean summed in dictionary-ID order.
        # Different K can change its float64 reduction order, never its targets.
        np.testing.assert_allclose(
            high[:, 0], low[:, 0], rtol=1e-12, atol=1e-14, equal_nan=True
        )
        np.testing.assert_array_equal(np.isfinite(high), np.isfinite(low))
        finite = np.isfinite(high).all((1, 2))
        if np.any(np.isfinite(high).any((1, 2)) != finite):
            raise ValueError(
                "partial attribute finiteness cannot use one shared hierarchy"
            )
        selected = (
            np.ones(n, bool) if selection is None else np.asarray(selection, bool)
        )
        self.roles = np.asarray(bank["fixed_band"])
        self.examples = np.asarray(bank["examples"])
        self.selected = selected
        groups = defaultdict(list)
        for j in np.flatnonzero(selected & finite):
            i, start, _ = self.examples[j]
            row = rows[i]
            groups[
                row["dataset"], row["subject"], row["record"], row["parent"], int(start)
            ].append(j)
        records = defaultdict(list)
        for key, positions in groups.items():
            records[key[:3]].append(positions)
        subjects = defaultdict(list)
        for key, crops in records.items():
            subjects[key[:2]].append(crops)
        self.people = sorted(subjects)
        if not self.people:
            raise ValueError("no finite held training people")
        weights = np.zeros((len(self.people), n))
        for s, identity in enumerate(self.people):
            for crops in subjects[identity]:
                for positions in crops:
                    weights[s, positions] = 1 / (
                        len(subjects[identity]) * len(crops) * len(positions)
                    )
        datasets = sorted({d for d, _ in self.people})
        self.macro_weights = np.array(
            [
                1 / (len(datasets) * sum(d == dataset for d, _ in self.people))
                for dataset, _ in self.people
            ]
        )
        self.parts = np.zeros((2, 5, len(self.people), 2, 3))
        for family, value in enumerate([low, high]):
            for role in range(5):
                take = self.roles == role
                self.parts[family, role] = (
                    weights[:, take] @ np.nan_to_num(value[take]).reshape(take.sum(), 6)
                ).reshape(len(self.people), 2, 3)
        complete = bank["valid"].sum(1)
        self.complete = int(complete[selected].sum())
        self.support_parts = np.zeros((2, 5))
        for family, mass in enumerate([low_mass, high_mass]):
            for role in range(5):
                take = selected & (self.roles == role)
                self.support_parts[family, role] = np.sum(
                    np.nan_to_num(np.asarray(mass)[take]) * complete[take]
                )

    def evaluate(self, mask):
        if not 0 <= int(mask) < 32:
            raise ValueError("five fixed dictionary choices required")
        choices = [(int(mask) >> role) & 1 for role in range(5)]
        values = sum(self.parts[choice, role] for role, choice in enumerate(choices))
        macro = np.einsum("s,sac->ac", self.macro_weights, values)
        gain = 1 - np.divide(
            macro[1], macro[0], out=np.full(3, np.nan), where=macro[0] > 0
        )
        fraction = (values[:, 1] < values[:, 0]).mean(0)
        mass = sum(
            self.support_parts[choice, role] for role, choice in enumerate(choices)
        ) / max(self.complete, 1)
        checks = dict(
            theta_alpha=bool(np.all(gain[:2] > 0)),
            one_gain=bool(np.max(gain) >= 0.05),
            people=bool(np.all(fraction[:2] >= 0.5)),
            support=bool(mass >= 0.9),
        )
        return dict(
            mask=int(mask),
            high_roles=np.flatnonzero(choices).tolist(),
            hierarchy=dict(
                macro=macro,
                by_subject=[
                    dict(dataset=d, subject=s, value=value)
                    for (d, s), value in zip(self.people, values)
                ],
                by_dataset={
                    d: values[
                        [i for i, person in enumerate(self.people) if person[0] == d]
                    ].mean(0)
                    for d, _ in self.people
                },
            ),
            gain=dict(zip(FIELDS, gain)),
            positive_subject_fraction=dict(zip(FIELDS, fraction)),
            complete_patches=self.complete,
            qualified_mass=float(mass),
            checks=checks,
            passed=all(checks.values()),
        )


def select_fixed_mask(records, priority=(0, 3, 4, 1, 2)):
    valid = [mask for mask, result in records.items() if result["passed"]]
    return (
        max(valid, key=lambda mask: tuple((int(mask) >> role) & 1 for role in priority))
        if valid
        else None
    )
