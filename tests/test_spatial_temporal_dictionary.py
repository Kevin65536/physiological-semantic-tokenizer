"""Synthetic checks for complementary time supports and integer-only decoding."""
import json

import numpy as np

from experiments.scripts.run_joint_spatial_temporal import average_statistics, synthetic


def test_complementary_slot_assignment_missing_channel_and_saved_decode(tmp_path):
    synthetic(tmp_path)
    record = json.loads((tmp_path/'synthetic.json').read_text())
    assert record['status'] == 'passed'
    assert record['roundtrip_exact']


def test_statistics_keep_equal_record_weight_and_exclude_unavailable_target():
    statistics = [{('continuous', 'S', 0): np.array([[2., 4.], [8., 10.]])},
                  {('continuous', 'S', 0): np.array([[6., 8.], [0., 0.]])}]
    averaged, available = average_statistics(statistics, [np.array([True, True]), np.array([True, False])])
    np.testing.assert_array_equal(averaged['continuous', 'S', 0], [[4., 6.], [8., 10.]])
    np.testing.assert_array_equal(available, [True, True])
