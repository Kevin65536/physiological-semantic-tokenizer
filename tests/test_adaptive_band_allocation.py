import numpy as np
from src.tokenizers.adaptive_band_allocation import (
    choose_roles,
    synthesize,
    extract_coordinates,
    regular_covariance,
)


def test_role_budget_missing_and_replay():
    rng = np.random.default_rng(471)
    cov = np.stack([regular_covariance(np.eye(12))] * 5)
    fields = rng.normal(size=(5, 12, 15, 400))
    seen = np.arange(12) != 2
    expected = np.bincount(np.flatnonzero(seen) % 5, minlength=5)
    for strategy in ("fixed", "energy", "innovation"):
        roles = choose_roles(fields, cov, seen, strategy)
        assert np.all(roles[~seen] == -1)
        for t in range(15):
            np.testing.assert_array_equal(np.bincount(roles[seen, t], minlength=5), expected)
        values = extract_coordinates(fields, roles)
        pred = synthesize(values, roles, cov)
        np.testing.assert_array_equal(pred[~seen], 0)
        np.testing.assert_array_equal(synthesize(values.copy(), roles.copy(), cov), pred)


def test_spatially_isolated_event_acquires_its_dominant_band():
    fields = np.zeros((5, 5, 15, 400))
    fields[0, 3, 7] = 5
    cov = np.repeat(np.eye(5)[None], 5, axis=0)
    roles = choose_roles(fields, cov, np.ones(5, bool), "innovation")
    assert roles[3, 7] == 0


def test_exchange_cost_matches_full_source_voltage_objective():
    from src.tokenizers.adaptive_band_allocation import exchange_costs, choose_exchanges

    rng = np.random.default_rng(53)
    n = 8
    fields = rng.normal(size=(5, n, 15, 40))
    mats = rng.normal(size=(5, n, n))
    cov = np.stack([regular_covariance(a @ a.T) for a in mats])
    weight = np.arange(n) + 1.0
    seen = np.ones(n, bool)
    seen[-1] = False
    weight[~seen] = 0
    roles, delta = choose_exchanges(fields, seen, exchange_costs(cov, seen, weight))
    fixed = np.where(seen, np.arange(n) % 5, -1)

    def cost(tags, t):
        loss = 0.0
        for b in range(5):
            idx = np.flatnonzero(tags == b)
            pred = np.zeros_like(fields[b, :, t])
            if len(idx):
                mat = np.linalg.solve(cov[b][np.ix_(idx, idx)], cov[b][idx]).T
                pred = mat @ fields[b, idx, t]
            loss += np.sum(weight[:, None] * (pred - fields[b, :, t]) ** 2)
        return loss

    for t in range(15):
        np.testing.assert_allclose(cost(roles[:, t], t) - cost(fixed, t), delta[t], atol=1e-10)
        assert delta[t] <= 0
        np.testing.assert_array_equal(np.sort(roles[:, t]), np.sort(fixed))
