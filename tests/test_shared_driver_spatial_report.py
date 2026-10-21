"""Report audits use temporary, unprotected numeric fixtures."""
import hashlib
import json

import numpy as np
import pytest

from experiments.scripts.render_shared_driver_spatial import _experiment_evidence, _figure_text_audit


def _fixture(root, mode):
    root.mkdir()
    (root / "prepared").mkdir()
    (root / "cells").mkdir()
    (root / "calibration").mkdir()
    identity = "fixture_subject_record"
    spec = dict(id="cell", record_id=identity, subject="s1", arm="K1_true_H4_E12",
                mode=mode, pairing="paired", method="joint_ridge", scenario="fixture")
    target = np.column_stack((np.linspace(0, 1, 120), np.linspace(1, -1, 120)))
    hb = np.zeros((120, 4, 2))
    hb[:, 0] = target * 2.
    prediction = target + [.5, -.25]
    sd = np.ones(68)
    sd[60:62] = [2., 4.]
    endpoint = slice(48, 72) if mode == "center_pair" else slice(20, 120)
    nmse = float(np.mean(((prediction[endpoint] - target[endpoint]) / sd[60:62]) ** 2))
    key = hashlib.sha256(identity.encode()).hexdigest()[:24]
    np.savez(root / "prepared" / (key + ".npz"), hb=hb)
    np.savez(root / "cells/cell.npz", target_prediction=prediction)
    row = dict(spec, status="completed", success=True, nmse=nmse, nrmse=np.sqrt(nmse))
    group = "|".join(spec[k] for k in ("scenario", "arm", "mode", "pairing", "method"))
    values = {
        "plan.json": dict(planned_cells=1, rows=[dict(id=identity)], cells=[spec]),
        "coordinate.json": dict(hb_scale=2., sd=sd.tolist()),
        "cells/cell.json": row,
        "summary.json": dict(execution="completed", terminal=1, status_counts={"completed": 1},
            groups={group: dict(subject_nmse_with_failure_penalty={"s1": nmse})}, comparisons=[]),
    }
    for name, value in values.items():
        (root / name).write_text(json.dumps(value))
    return prediction


@pytest.mark.parametrize("mode", ["heldout_pair", "center_pair"])
def test_report_recomputes_normalization_from_retained_arrays(tmp_path, mode):
    root = tmp_path / "run"
    prediction = _fixture(root, mode)
    result = _experiment_evidence(root)
    assert result["checks"]["rescored_arrays"] == 1
    assert result["checks"]["max_nmse_error"] < 1e-14
    prediction[50, 0] += 3.
    np.savez(root / "cells/cell.npz", target_prediction=prediction)
    with pytest.raises(ValueError, match="scoring mismatch"):
        _experiment_evidence(root)


def test_report_rejects_incomplete_terminal_denominator(tmp_path):
    root = tmp_path / "run"
    _fixture(root, "heldout_pair")
    path = root / "summary.json"
    summary = json.loads(path.read_text())
    summary["execution"] = "incomplete"
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete planned denominator"):
        _experiment_evidence(root)


def test_figure_audit_distinguishes_undrawn_ticks_from_clipped_text():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3))
    try:
        ax.set_ylim(0, 1.22)
        ax.set_yticks(np.arange(0, 1.6, .2))
        ax.set_ylim(0, 1.22)
        assert _figure_text_audit(fig)["text_artists_outside_figure"] == []
        fig.text(1.01, .5, "This label really is clipped")
        with pytest.raises(RuntimeError, match="outside its image"):
            _figure_text_audit(fig)
    finally:
        plt.close(fig)
