import numpy as np
from src.tokenizers.eeg_component_transfer import (
    component_residual,
    frequency_groups,
    transfer_coefficients,
    transfer_waveform,
)


def test_complex_transfer_recovers_phase_shift_from_only_the_other_native_position():
    t = np.arange(6000) / 200
    x = np.column_stack([np.sin(2 * np.pi * 6 * t), np.zeros(6000)])
    target = np.cos(2 * np.pi * 6 * t)
    f = np.fft.rfft(x, axis=0)
    gram = np.zeros((5, 2, 2, 2), complex)
    cross = np.zeros((5, 2, 2), complex)
    indices = frequency_groups()[1]
    observed = f[indices]
    gram[1, 1] = observed.conj().T @ observed
    cross[1, 1] = observed.conj().T @ np.fft.rfft(target)[indices]
    local, full = transfer_coefficients(gram, cross, 0.001)
    np.testing.assert_allclose(
        transfer_waveform(f, full)[:, 1], target / 1.001, atol=1e-10
    )
    np.testing.assert_array_equal(transfer_waveform(f, local)[:, 1], np.zeros(6000))


def test_weighted_transfer_minimizes_two_component_error_with_sum_conservation():
    truth_s, truth_o = np.array([1.0, 7.0]), np.array([4.0, 2.0])
    s, o = np.array([2.0, 1.0]), np.array([5.0, 4.0])
    vs, vo = np.array([1.0, 3.0]), np.array([4.0, 1.0])
    delta = component_residual(truth_s, truth_o, s, o, [vs, vo])
    np.testing.assert_allclose((s + delta - truth_s) / vs, (o - delta - truth_o) / vo)
    np.testing.assert_allclose((s + delta) + (o - delta), s + o)


def test_frozen_integer_allocator_conserves_total_and_uses_only_sent_streams():
    from types import SimpleNamespace
    from src.tokenizers.eeg_component_transfer import IntegerSumSourceDecoder
    import pytest

    layout = {"positions": {"tiny": ["a", "b"]}}

    def decode(ids, dataset):
        s = np.repeat(np.asarray(ids["S"], float)[..., None], 400, axis=-1)
        return dict(S=s, O=2 * s, sum=3 * s)

    raw = SimpleNamespace(metadata=layout, decode_ids=decode)
    source = SimpleNamespace(
        metadata=layout, predict_continuous=lambda x, d, m: {"S": 0.25 * x}
    )
    ids = {b: np.array([[[1] * 15, [0] * 15]], np.int32) for b in "SO"}
    model = IntegerSumSourceDecoder(raw, source)
    result = model.decode_ids(ids, "tiny")
    np.testing.assert_array_equal(result["sum"], raw.decode_ids(ids, "tiny")["sum"])
    np.testing.assert_array_equal(result["S"], 0.25 * result["sum"])
    np.testing.assert_array_equal(result["S"] + result["O"], result["sum"])
    with pytest.raises(ValueError):
        model.decode_ids(dict(ids, X=np.zeros(100)), "tiny")
