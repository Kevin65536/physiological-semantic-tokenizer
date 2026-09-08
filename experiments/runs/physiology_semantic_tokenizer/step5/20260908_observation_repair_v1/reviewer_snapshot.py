"""Append-only numerical review of this run; does not rewrite run outcomes."""
from pathlib import Path
import hashlib
import importlib.util
import json
import subprocess
import sys
from decimal import Decimal, localcontext

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from experiments import evaluate_step5_observation_repair as repair
from experiments.scripts.review_step5_observation_diagnostic import replay_failure
import numpy as np

run_dir = Path(__file__).resolve().parent
output = run_dir/'verification_review.json'
if output.exists():
    raise FileExistsError('the dated review already exists; append a new version instead')
manifest = json.loads((run_dir/'manifest.json').read_text())
for path, digest in manifest['source_sha256'].items():
    if repair.diagnostic.digest(ROOT/path) != digest:
        raise ValueError('source differs from the completed run: '+path)
cfg, dc, base, _, _ = repair.load_config(run_dir/'resolved_config.yaml')
legacy_source = subprocess.check_output(['git', 'show', '5ea9245:src/inference/t3a_balloon_robust_ssm.py'], cwd=ROOT)
legacy_sha = hashlib.sha256(legacy_source).hexdigest()
assert legacy_sha == '4221b4a53e9b2041d6db5e0274e4d5b509ab8108291e004c01ab67d9054326e4'
legacy_path = run_dir/'legacy_core_snapshot.py'
with legacy_path.open('xb') as stream:
    stream.write(legacy_source)
module_spec = importlib.util.spec_from_file_location('legacy_balloon_review', legacy_path)
legacy = importlib.util.module_from_spec(module_spec)
sys.modules[module_spec.name] = legacy
module_spec.loader.exec_module(legacy)

normal_errors = []
for flow in np.geomspace(.03, 30., 61):
    before, after = np.array(legacy._extraction(flow, .32)), np.array(repair.core._extraction(flow, .32))
    normal_errors.append(np.abs(before-after))
p, c = repair.step5.localization.model(base, 'W', 0.)
rng = np.random.default_rng(20260908)
rhs_errors, jacobian_errors, transition_errors = [], [], []
for _ in range(64):
    state = np.r_[rng.normal(0, .03, 2), rng.uniform(.7, 1.3, 4)]
    z = repair.core.physical_to_transformed(state)
    rhs_errors.append(np.max(abs(legacy.balloon_rhs(z, p)-repair.core.balloon_rhs(z, p))))
    jacobian_errors.append(np.max(abs(legacy.balloon_rhs_jacobian(z, p)-repair.core.balloon_rhs_jacobian(z, p))))
    before = legacy.rk4_transition_with_jacobian(z, p, c)
    after = repair.core.rk4_transition_with_jacobian(z, p, c)
    transition_errors.append(max(np.max(abs(a-b)) for a, b in zip(before, after)))

scalar = []
for value in ('.003209156', '.01', '.2', '1', '3', '100', '1000000000000'):
    with localcontext() as ctx:
        ctx.prec = 100
        flow, e0 = Decimal(value), Decimal('.32')
        a = (1-e0).ln()/flow
        complement = a.exp()
        e = 1-complement
        derivative = complement*(1-e0).ln()/(flow*flow)
        flux = flow*e/e0
        flux_derivative = flow*(e+flow*derivative)/e0
        expected = [float(v) for v in (e, derivative, flux)]
    actual = repair.core._extraction(float(flow), .32)
    scalar.append(dict(flow=float(flow), expected=expected, actual=actual,
        log_one_minus_E=float(a), one_minus_E=float(complement),
        flux_log_flow_derivative_reference=float(flux_derivative),
        flux_log_flow_derivative=repair.core._flow_extraction_log_derivative(float(flow), .32, *actual[:2])))

summary = json.loads((run_dir/'summary.json').read_text())
old_summary = json.loads((ROOT/cfg['previous_run']/'summary.json').read_text())
replays = []
for subject, trial in summary['replay']['saturation_unique_trials']:
    arrays, identity = repair.load_replay_inputs(cfg, dc, base, subject)
    assert identity['input_sha256'] == json.loads((run_dir/'replay_inputs.json').read_text())[subject]['input_sha256']
    config = json.loads(json.dumps(base))
    config['model']['observation_scale'] = old_summary['preparation'][subject]['broadband_pca_noise_scale']
    y = arrays['broadband_pca'][trial].copy(); y[:, 0] = np.nan
    for w in (0., -.5):
        for substeps in (2, 8, 32):
            result = replay_failure(y, config, w, substeps)
            pre = result['diagnostic'].get('pretransition_transformed_state')
            if pre is not None:
                flow = float(np.exp(pre[2])); velocity = float(pre[1])
                result['last_pretransition_flow'] = flow
                result['last_pretransition_flow_velocity'] = velocity
                result['local_linear_time_to_zero_s'] = -flow/velocity if velocity < 0 else None
            replays.append(dict(subject=subject, training_trial_index=trial, w=w, result=result))

report = dict(source_summary_sha256=repair.diagnostic.digest(run_dir/'summary.json'),
    reviewer_sha256=repair.diagnostic.digest(Path(__file__)), legacy_core_sha256=legacy_sha,
    normal_range=dict(flow_range=[.03, 30.], scalar_cases=61, state_cases=64,
        maximum_extraction_tuple_errors=np.max(normal_errors, axis=0),
        maximum_rhs_error=max(rhs_errors), maximum_jacobian_error=max(jacobian_errors),
        maximum_transition_or_jacobian_error=max(transition_errors)),
    high_precision_scalar=scalar,
    refined_substep_replay=replays,
    interpretation='Same-input post-result numerical checks only. A negative flow velocity near zero can still violate positive-flow dynamics after valid extraction saturation; the local linear zero time is descriptive, not an alternative integrator or clipping rule.')
repair.step5.write_json(output, report)
print(json.dumps(repair.step5.localization.jsonable(report['normal_range'])))
print([(r['subject'], r['training_trial_index'], r['w'], r['result']['substeps'], r['result']['execution'],
        r['result'].get('error')) for r in replays])
