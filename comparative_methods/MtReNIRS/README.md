# MtReNIRS fNIRS-only comparison

[Official code](https://github.com/microsoft/MtReNIRS) and bundled weights are
pinned by [the source manifest](sources/method_manifest.yaml). The local
`upstream/` checkout is ignored. `rebench_v1.py` uses the unmodified encoder and
adds recorded-input masks, sample-local normalization and registered template
geometry. It does not load measured data or fit a task head.

The development comparison uses
[`downstream_fnirs_probe_v1.yaml`](../downstream_fnirs_probe_v1.yaml) through the
existing `performance_analysis/audit_downstream_inputs_v1.py --stage probe-plan`
and subsequent probe stages. Subject folds, fitting, saved predictions and
independent metric verification remain owned there. Targeted adapter and
traditional-fNIRS tests are `tests/test_downstream_fnirs_probe.py`.

The input remains the common 10 Hz, 0.01–0.2 Hz HbO/HbR measurement. The
sample-local normalization differs from the upstream full-run normalization;
source preprocessing is not silently substituted. Missing channels are masked
in the official attention. Thirty-second inputs preserve three ordered tokens;
DSR uses its actual two seconds followed by explicit zero padding. REFED has no
validated sequence adapter in this comparison.

Simultaneous is a declared upstream pretraining source. This overlap is retained
as provenance and, at the user's explicit request, does not exclude its tasks.
Geometry is template metadata, not measured participant anatomy; named EEG
template locations can establish the coordinate frame, but no EEG signal enters
the model. Current execution and verdicts remain in the project registry.
