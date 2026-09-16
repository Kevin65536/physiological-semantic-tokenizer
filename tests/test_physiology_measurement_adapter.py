import numpy as np
import pytest

from src.data.physiology_measurement_adapter import (
    ADAPTER_SCHEMA,
    MeasurementAdapterSpec,
    PhysiologyMeasurementAdapter,
    measurement_unit_conversion,
    PAIRED_ADAPTER_SCHEMA,
    measurement_baseline,
)


def _fit(records, transform="center"):
    return PhysiologyMeasurementAdapter.fit(
        records,
        dataset="example",
        modality="fnirs",
        original_semantics="relative HbO/HbR",
        original_unit="mmol/L",
        canonical_semantics="baseline-relative paired optical measurement",
        transform=transform,
        channel_names=("primary", "secondary"),
        fit_subjects=("01", "02"),
    )


def test_adapter_round_trip_and_schema():
    record = np.asarray([[10.0, 20.0], [11.0, 18.0], [9.0, 22.0], [12.0, 17.0]])
    adapter = _fit([record])
    baseline = adapter.record_baseline(record)
    canonical = adapter.transform(record, baseline=baseline)

    np.testing.assert_allclose(adapter.inverse_transform(canonical, baseline=baseline), record)
    assert adapter.spec.schema == ADAPTER_SCHEMA
    restored = MeasurementAdapterSpec.from_dict(adapter.spec.to_dict())
    assert restored == adapter.spec


def test_adapter_is_crop_position_invariant_when_record_baseline_is_reused():
    record = np.column_stack((np.linspace(2.0, 4.0, 100), np.linspace(5.0, 1.0, 100)))
    adapter = _fit([record])
    baseline = adapter.record_baseline(record)
    full = adapter.transform(record, baseline=baseline)

    np.testing.assert_allclose(
        adapter.transform(record[20:40], baseline=baseline),
        full[20:40],
    )


def test_shared_pair_scale_preserves_relative_amplitude_ratio():
    time = np.linspace(0.0, 2.0 * np.pi, 200)
    record = np.column_stack((2.0 * np.sin(time), 0.5 * np.sin(time)))
    adapter = _fit([record])
    canonical = adapter.transform(record)

    raw_ratio = np.ptp(record[:, 0]) / np.ptp(record[:, 1])
    canonical_ratio = np.ptp(canonical[:, 0]) / np.ptp(canonical[:, 1])
    np.testing.assert_allclose(canonical_ratio, raw_ratio)


def test_relative_change_handles_zero_baseline_without_nonfinite_values():
    record = np.asarray([[0.0, 10.0], [1.0, 11.0], [-1.0, 9.0], [0.5, 10.5]])
    adapter = _fit([record], transform="relative_change")
    canonical = adapter.transform(record)

    assert np.all(np.isfinite(canonical))


def test_unit_evidence_and_quantity_are_required_for_conversion():
    resolve = measurement_unit_conversion
    volts = resolve("V", quantity="electric_potential", evidence="header", group="device")
    np.testing.assert_allclose(np.array([1e-6, -2e-6]) * volts["factor"], [1, -2])
    hb = resolve("mmol/L", quantity="concentration_change", evidence="yUnit", group="device")
    assert hb["factor"] == 1000 and hb["output_unit"] == "uM"
    for unit, evidence in [("V", ""), ("unknown", "paper reviewed"), ("mM mm", "header")]:
        quantity = "electric_potential" if unit == "V" else "concentration_change"
        state = resolve(unit, quantity=quantity, evidence=evidence, group="device")
        assert state["unit_status"] == "unknown" and state["factor"] == 1
    detector = resolve("V", quantity="detector_voltage", evidence="yUnit", group="device")
    assert detector["factor"] == 1 and detector["output_unit"] == "V"


def test_paired_fit_freezes_scale_and_preserves_hbt_and_missing_support():
    t = np.arange(100)/10
    clean = np.column_stack([4*np.sin(t), -np.sin(t)])
    damaged = clean.copy()
    damaged[30:40] = 1e8
    mask = np.ones_like(clean,dtype=bool)
    mask[30:40] = False
    kwargs = dict(dataset='device',modality='fnirs',original_semantics='published_Hb',
                  original_unit='relative',canonical_semantics='paired_Hb',transform='center',
                  channel_names=['CH1_HbO','CH1_HbR'],fit_subjects=['train'],
                  baselines=[np.zeros(2)],valid_masks=[mask],fit_record_ids=['train/record'])
    first = PhysiologyMeasurementAdapter.fit([clean],**kwargs)
    second = PhysiologyMeasurementAdapter.fit([damaged],**kwargs)
    assert first.spec == second.spec
    assert first.spec.schema == PAIRED_ADAPTER_SCHEMA
    baseline = np.array([.5,.2])
    y = first.transform(clean,baseline=baseline)
    np.testing.assert_allclose(y.sum(axis=1), (clean.sum(axis=1)-baseline.sum())/first.spec.shared_scale)
    np.testing.assert_allclose(np.std(y[:,0])/np.std(y[:,1]),4.)
    np.testing.assert_allclose(first.inverse_transform(y,baseline=baseline),clean,atol=1e-15)
    frozen = first.spec.shared_scale
    first.transform(clean*1e5,baseline=baseline)
    assert first.spec.shared_scale == frozen
    with pytest.raises(ValueError,match='explicit declared baseline'):
        first.transform(clean)


def test_declared_baseline_uses_common_real_support_and_retains_noise_rank():
    t = np.arange(24)/4-2
    x = np.column_stack([np.sin(t),-.3*np.sin(t)])
    mask = np.ones_like(x,dtype=bool);mask[2,1] = False
    b,w,state = measurement_baseline(x,t,mask,interval_s=(-2,0),
                                    role='pre_event',evidence='fixture event clock',minimum_samples=6)
    assert state['sample_count'] == 7 and w[2] == 0
    c = np.eye(len(t))-np.ones((len(t),1))*w
    np.testing.assert_allclose(c@x,x-b,atol=1e-15)
    covariance = c@c.T
    assert np.linalg.matrix_rank(covariance) == len(t)-1
    assert abs(covariance[12,13]) > .01
    missing,_,state = measurement_baseline(x,t,mask,interval_s=(-5,-3),
                                          role='pre_event',evidence='fixture')
    assert missing is None and state['status']=='insufficient_support'
