# Comparator performance-degradation analyses

> **Historical P0: STOPPED.** The P0 synthesis is complete and retained as exploratory
> evidence. Unexecuted mechanism follow-ups are **ABANDONED**; this directory is
> an evidence index, not a current analysis queue or launch entrypoint.

The separately requested [MA 30-second adaptation audit](../ma_adaptation_audit_v1.yaml)
uses [`audit_ma_adaptation_v1.py`](audit_ma_adaptation_v1.py) and its own versioned
run under `comparative_methods/runs/performance_analysis/`. It does not resume P0;
its input scope and interpretation are in the owning experiment design.

This package contains post-campaign mechanism analyses defined by
`docs/comparisons/PERFORMANCE_DEGRADATION_ANALYSIS_PLAN_20260816.md`.
The completed first-round synthesis is
`docs/comparisons/PERFORMANCE_DEGRADATION_P0_RESULTS_20260816.md`; generated
evidence bundles are indexed under
`comparative_methods/runs/performance_analysis/20260816_p0/`.

Development and model-selection analyses must use public development data or
outer-train-only resampling. Existing protected aggregates may be used only for
the frozen descriptive global result surface. No script in this package may
retune a method against protected predictions or test labels.

Generated artifacts belong under
`comparative_methods/runs/performance_analysis/20260816_p0/` and must retain
source locators, configuration, seeds, sample/subject support, uncertainty
definitions, and an explicit exploratory/confirmatory label.
