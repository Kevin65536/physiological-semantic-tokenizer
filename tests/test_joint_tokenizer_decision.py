from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

from experiments.scripts.review_joint_tokenizer import score


def test_joint_waveform_gates_keep_each_hb_branch_and_color():
    config = Path(__file__).resolve().parents[1] / 'experiments/configs/physiology_semantic_tokenizer/joint_tokenizer_acceptance_v1.yaml'
    contract = yaml.safe_load(config.read_text())
    def summary(nmse):
        metrics = {}
        for key, val in [('nmse', nmse), ('rms_gain', 1.), ('correlation', .9),
                         ('slope_correlation', .8), ('curvature_correlation', .6),
                         ('boundary_jump_difference_nmse', .2)]:
            metrics[key] = {'macro': np.full((3, 2), val).tolist()}
        metrics['nmse']['by_dataset'] = {str(d): metrics['nmse']['macro'] for d in range(4)}
        return dict(modality='hb', role='screen', checkpoint='fixture', metrics=metrics,
                    integer_replay={'status': 'passed'}, support={'nonfinite_prediction_item_color_cells': 0})

    reference, continuous, good = summary(.4), summary(.3), summary(.1)
    assert score(good, reference, continuous, contract)['reconstruction_passed']
    missing_gain = deepcopy(good)
    missing_gain['metrics']['rms_gain']['macro'][2][0] = float('nan')
    result = score(missing_gain, reference, continuous, contract)
    assert not result['reconstruction_passed']
    assert result['checks']['gain_min']['values'][2][0] is None
    assert not result['checks']['gain_min']['component_color_pass'][2][0]
