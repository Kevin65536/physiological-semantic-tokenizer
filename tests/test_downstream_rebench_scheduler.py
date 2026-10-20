"""Scheduler checks use tiny local processes, never model/data artifacts."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "comparative_methods/run_downstream_rebench_v1.py"
spec = importlib.util.spec_from_file_location("downstream_rebench_scheduler", SCRIPT)
scheduler = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(scheduler)


@pytest.fixture
def fixture_worker(tmp_path):
    worker = tmp_path / "worker.py"
    worker.write_text('''import json, os, pathlib, signal, sys, time
output = pathlib.Path(sys.argv[1]); output.mkdir(parents=True, exist_ok=True)
seconds = float(sys.argv[2]); code = int(sys.argv[3])
def write(status):
    temporary = output / ("status.tmp." + str(os.getpid()))
    temporary.write_text(json.dumps({"status": status, "pid": os.getpid()}))
    temporary.replace(output / "status.json")
def interrupted(signum, frame):
    (output / "checkpoint").write_text("retained")
    write("interrupted")
    sys.exit(75)
signal.signal(signal.SIGTERM, interrupted)
with (output / "starts.jsonl").open("a") as stream:
    stream.write(json.dumps({"pid": os.getpid(), "time": time.time()}) + "\\n")
write("running")
print("worker stdout", flush=True); print("worker stderr", file=sys.stderr, flush=True)
time.sleep(seconds)
write("completed" if code == 0 else "failed")
(output / "ended").write_text(str(time.time()))
sys.exit(code)
''')
    return worker


def write_manifest(tmp_path, fixture_worker, *, specs=None, cpu=2):
    run = tmp_path / "run"
    jobs = []
    for job_id, seconds, code, deps in (specs or [("a", .3, 0, []), ("b", .3, 0, [])]):
        output = run / job_id
        jobs.append({"id": job_id, "kind": "fixture", "command": [sys.executable, str(fixture_worker), str(output), str(seconds), str(code)],
                     "cwd": str(tmp_path), "output_dir": str(output), "dependencies": deps,
                     "resources": {"gpu": False, "gpu_memory_mib": 0, "ram_gib": .01, "cpu_threads": 1},
                     "timeout_seconds": 10})
    manifest = {"schema": "downstream_rebench_schedule_v1", "run_id": "fixture", "run_root": str(run),
                "limits": {"cpu_threads": cpu, "ram_gib": 1, "gpu_ids": [], "gpu_reserve_mib": {}, "max_gpu_jobs_per_device": 2}, "jobs": jobs}
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps(manifest))
    return path, run, manifest


def launch(path, **kwargs):
    return subprocess.Popen([sys.executable, str(SCRIPT), "--manifest", str(path), "--poll-seconds", ".03", "--stop-grace-seconds", "1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs)


def wait_for(predicate, timeout=8):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        result = predicate()
        if result:
            return result
        time.sleep(.02)
    raise AssertionError("fixture did not reach expected state")


def status(run):
    return scheduler.read_json(run / "scheduler_status.json") or {}


def test_real_cpu_parallel_and_logs(tmp_path, fixture_worker):
    path, run, _ = write_manifest(tmp_path, fixture_worker)
    process = launch(path)
    assert process.wait(timeout=10) == 0, process.communicate()
    retained = status(run)
    assert retained["status"] == "completed"
    assert retained["max_active_jobs"] == 2
    starts = {name: json.loads((run / name / "starts.jsonl").read_text()) for name in ("a", "b")}
    assert max(x["time"] for x in starts.values()) < min(float((run / name / "ended").read_text()) for name in starts)
    assert (run / "a/scheduler_stdout.log").read_text().strip() == "worker stdout"
    assert (run / "a/scheduler_stderr.log").read_text().strip() == "worker stderr"
    assert scheduler.read_json(run / "progress.json")["progress"]["completed_fraction"] == 1
    # A completed controller run is idempotent and does not launch workers again.
    assert launch(path).wait(timeout=10) == 0
    assert len((run / "a/starts.jsonl").read_text().splitlines()) == 1


def test_failure_blocks_dependencies_but_independent_jobs_continue(tmp_path, fixture_worker):
    path, run, _ = write_manifest(tmp_path, fixture_worker, specs=[("bad", .05, 4, []), ("dependent", .05, 0, ["bad"]), ("independent", .1, 0, [])])
    assert launch(path).wait(timeout=10) == 1
    records = status(run)["jobs"]
    assert records["bad"]["status"] == "failed"
    assert records["bad"]["returncode"] == 4
    assert records["dependent"]["status"] == "blocked"
    assert records["dependent"]["blocked_by"] == ["bad"]
    assert records["independent"]["status"] == "completed"
    assert not (run / "dependent/starts.jsonl").exists()
    assert launch(path).wait(timeout=10) == 1
    assert len((run / "bad/starts.jsonl").read_text().splitlines()) == 1


def test_aggregate_runs_after_failed_and_blocked_dependencies_without_hiding_failure(tmp_path, fixture_worker):
    path, run, manifest = write_manifest(tmp_path, fixture_worker, specs=[
        ("bad", .05, 4, []), ("blocked", .01, 0, ["bad"]),
        ("independent", .2, 0, []), ("aggregate", .4, 0, ["bad", "blocked", "independent"])])
    aggregate = manifest["jobs"][-1]
    aggregate.update(kind="aggregate", dependency_policy="terminal")
    report = run / "aggregate/aggregate.json"
    aggregate["command"] = [sys.executable, "-c", (
        "import json,pathlib,time; time.sleep(.4); "
        f"s=json.loads(pathlib.Path({str(run / 'scheduler_status.json')!r}).read_text()); "
        f"pathlib.Path({str(report)!r}).write_text(json.dumps({{k:v['status'] for k,v in s['jobs'].items()}}))"
    )]
    path.write_text(json.dumps(manifest))
    process = launch(path)
    try:
        running = wait_for(lambda: (s if (s := status(run)).get("jobs", {}).get("aggregate", {}).get("status") == "running" else None))
        assert running["status"] == "running"
        assert running["jobs"]["bad"]["status"] == "failed"
        assert running["jobs"]["blocked"]["status"] == "blocked"
        assert process.wait(timeout=10) == 1
        observed = json.loads(report.read_text())
        assert observed["bad"] == "failed"
        assert observed["blocked"] == "blocked"
        assert observed["independent"] == "completed"
        retained = status(run)
        assert retained["status"] == "failed"
        assert retained["jobs"]["aggregate"]["status"] == "completed"
        assert retained["jobs"]["bad"]["returncode"] == 4
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)


def test_aggregate_waits_for_interrupted_dependency_and_does_not_start_on_shutdown(tmp_path, fixture_worker):
    path, run, manifest = write_manifest(tmp_path, fixture_worker, specs=[("interrupted", .05, 75, []), ("aggregate", .01, 0, ["interrupted"])])
    manifest["jobs"][-1].update(kind="aggregate", dependency_policy="terminal")
    path.write_text(json.dumps(manifest))
    process = launch(path)
    try:
        wait_for(lambda: status(run).get("jobs", {}).get("interrupted", {}).get("status") == "interrupted")
        assert status(run)["jobs"]["aggregate"]["status"] == "pending"
        assert not (run / "aggregate/starts.jsonl").exists()
    finally:
        process.terminate()
        assert process.wait(timeout=10) == 143
    assert not (run / "aggregate/starts.jsonl").exists()
    assert status(run)["jobs"]["aggregate"]["status"] == "pending"


@pytest.mark.parametrize("change,match", [
    ({"dependency_policy": "terminal"}, "only allowed for aggregate"),
    ({"dependency_policy": "unknown"}, "must be completed or terminal"),
    ({"dependency_policy": "terminal", "kind": "aggregate", "gpu": True}, "must use CPU"),
])
def test_dependency_policy_contract(tmp_path, fixture_worker, change, match):
    path, _, manifest = write_manifest(tmp_path, fixture_worker)
    if change.get("gpu"):
        change = dict(change)
        change.pop("gpu")
        manifest["limits"]["gpu_ids"] = [0]
        manifest["jobs"][0]["resources"].update(gpu=True, gpu_memory_mib=1000)
    manifest["jobs"][0].update(change)
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=match):
        scheduler.load_manifest(path)


def test_graceful_stop_and_checkpoint_resume(tmp_path, fixture_worker):
    path, run, _ = write_manifest(tmp_path, fixture_worker, specs=[("a", .5, 0, [])])
    process = launch(path)
    try:
        wait_for(lambda: (run / "a/status.json").exists())
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=10) == 143
        assert status(run)["status"] == "interrupted"
        assert (run / "a/checkpoint").exists()
        assert status(run)["jobs"]["a"]["status"] == "interrupted"
        assert launch(path).wait(timeout=10) == 0
        assert status(run)["jobs"]["a"]["attempt"] == 2
        assert len((run / "a/starts.jsonl").read_text().splitlines()) == 2
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_sigkill_controller_adopts_live_worker_without_duplicate(tmp_path, fixture_worker):
    path, run, _ = write_manifest(tmp_path, fixture_worker, specs=[("a", .6, 0, [])])
    process = launch(path)
    try:
        wait_for(lambda: (run / "a/status.json").exists())
        wait_for(lambda: status(run).get("jobs", {}).get("a", {}).get("status") == "running")
        process.kill()
        process.wait(timeout=10)
        resumed = launch(path)
        assert resumed.wait(timeout=10) == 0, resumed.communicate()
        assert status(run)["jobs"]["a"]["attempt"] == 1
        assert len((run / "a/starts.jsonl").read_text().splitlines()) == 1
        events = [json.loads(line)["event"] for line in (run / "scheduler_events.jsonl").read_text().splitlines()]
        assert "job_adopted" in events
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_single_controller_lock_and_readonly_status(tmp_path, fixture_worker):
    path, run, _ = write_manifest(tmp_path, fixture_worker, specs=[("a", 3, 0, [])])
    process = launch(path)
    try:
        wait_for(lambda: status(run).get("jobs", {}).get("a", {}).get("status") == "running")
        duplicate = launch(path)
        stdout, stderr = duplicate.communicate(timeout=10)
        assert duplicate.returncode == 2
        assert "controller lock" in stderr
        observed = subprocess.run([sys.executable, str(SCRIPT), "--manifest", str(path), "--status"], capture_output=True, text=True, timeout=10)
        assert observed.returncode == 0
        assert json.loads(observed.stdout)["status"] == "running"
    finally:
        process.terminate()
        process.wait(timeout=10)


def test_manifest_unique_ids_cycle_and_immutable_resume(tmp_path, fixture_worker):
    path, run, manifest = write_manifest(tmp_path, fixture_worker, specs=[("a", .01, 0, [])])
    assert launch(path).wait(timeout=10) == 0
    manifest["jobs"][0]["timeout_seconds"] = 11
    path.write_text(json.dumps(manifest))
    rejected = launch(path)
    assert rejected.wait(timeout=10) == 2
    assert "manifest changed" in rejected.communicate()[1]
    manifest["jobs"].append(dict(manifest["jobs"][0]))
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="duplicate job id"):
        scheduler.load_manifest(path)
    manifest["jobs"] = manifest["jobs"][:1]
    manifest["jobs"][0]["dependencies"] = ["a"]
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dependency"):
        scheduler.load_manifest(path)


def test_timeout_is_failure_and_blocks_descendants(tmp_path, fixture_worker):
    path, run, manifest = write_manifest(tmp_path, fixture_worker, specs=[("a", 3, 0, []), ("b", .01, 0, ["a"])])
    manifest["jobs"][0]["timeout_seconds"] = .15
    path.write_text(json.dumps(manifest))
    assert launch(path).wait(timeout=10) == 1
    assert status(run)["jobs"]["a"]["reason"] == "timeout"
    assert status(run)["jobs"]["b"]["status"] == "blocked"


@pytest.mark.parametrize("retained_status,exit_code", [("interrupted", 1), ("failed", 0)])
def test_failure_is_not_hidden_by_worker_exit_or_stale_status(tmp_path, fixture_worker, retained_status, exit_code):
    path, run, manifest = write_manifest(tmp_path, fixture_worker, specs=[("a", .01, 0, [])])
    manifest["jobs"][0]["command"] = [sys.executable, "-c", f"raise SystemExit({exit_code})"]
    path.write_text(json.dumps(manifest))
    (run / "a").mkdir(parents=True)
    (run / "a/status.json").write_text(json.dumps({"status": retained_status}))
    assert launch(path).wait(timeout=10) == 1
    assert status(run)["jobs"]["a"]["status"] == "failed"


def test_cpu_ram_and_gpu_reservations_use_actual_free_and_unallocated_peak(tmp_path, fixture_worker, monkeypatch):
    path, _, manifest = write_manifest(tmp_path, fixture_worker)
    manifest["limits"].update(gpu_ids=[0, 1], gpu_reserve_mib={"0": 6000, "1": 2000})
    path.write_text(json.dumps(manifest))
    controller = scheduler.Scheduler(path)
    monkeypatch.setattr(scheduler.psutil, "virtual_memory", lambda: type("Memory", (), {"available": 100 * scheduler.GIB})())
    controller.snapshots = {0: {"free_mib": 12000}, 1: {"free_mib": 16000}}
    gpu_request = {"gpu": True, "gpu_memory_mib": 7000, "ram_gib": .1, "cpu_threads": 1}
    assert controller.fit(gpu_request)[:2] == (True, 1)
    controller.active = {"existing": {"record": {"gpu_id": 1, "reserved_resources": gpu_request,
                                                "rss_bytes": 0, "current_gpu_memory_mib": 2000}}}
    controller.snapshots[1]["free_mib"] = 13000
    # 13000 free - 2000 reserve - (7000 reserved - 2000 current) = 6000.
    assert controller.fit(gpu_request) == (False, None, "available_gpu_memory_or_slots")
    assert controller.fit({**gpu_request, "gpu": False, "cpu_threads": 2})[2] == "cpu_threads"
    assert controller.fit({**gpu_request, "gpu": False, "ram_gib": 1})[2] == "reserved_ram"
    controller.active = {}
    controller.snapshots = {}
    assert controller.fit(gpu_request)[0] is False


def test_smoke_resource_profile_and_missing_profile_fail_closed(tmp_path, fixture_worker):
    path, run, manifest = write_manifest(tmp_path, fixture_worker)
    manifest["limits"].update(gpu_ids=[0], gpu_reserve_mib={"0": 100})
    manifest["jobs"][1].update(dependencies=["a"], profile_from="a")
    manifest["jobs"][1]["resources"].update(gpu=True, gpu_memory_mib=1000)
    path.write_text(json.dumps(manifest))
    controller = scheduler.Scheduler(path)
    with pytest.raises(ValueError, match="no positive measured"):
        controller.request("b")
    (run / "a").mkdir(parents=True)
    (run / "a/resource_usage.json").write_text(json.dumps({"train": {"peak_cuda_memory_allocated_bytes": 2000 * scheduler.MIB}}))
    assert controller.request("b")["gpu_memory_mib"] == 3112
    manifest["jobs"][1]["dependencies"] = []
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="explicit dependency"):
        scheduler.load_manifest(path)


@pytest.fixture
def build_fixture(tmp_path):
    import shutil
    import yaml
    project = tmp_path / "project"
    config_path = project / "comparative_methods/downstream_rebench_v1.yaml"
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "comparative_methods/downstream_rebench_v1.yaml", config_path)
    config = yaml.safe_load(config_path.read_text())
    run = project / "comparative_methods/runs/downstream_rebench_v1/fixture_v1"
    run.mkdir(parents=True)
    source = run / "source"
    (source / "comparative_methods").mkdir(parents=True)
    directories = {"biot": "BIOT", "cbramod": "CBraMod", "reve": "REVE", "normwear": "NormWear",
                   "efrm": "EFRM-PyTorch", "sta_net": "STA-Net-PyTorch", "brainfusion": "BrainFusion-NVC-CSP-Stacking"}
    # Every source would raise if imported or run: the builder only reads metadata.
    (source / "comparative_methods/downstream_rebench_worker_v1.py").write_text(f"METHOD_DIRS = {directories!r}\nraise AssertionError('must not import worker')\n")
    (source / "comparative_methods/aggregate_downstream_rebench_v1.py").write_text("raise AssertionError('must not run aggregate')\n")
    for method in config["methods"]:
        path = source / (config["methods"][method].get("source") or f"comparative_methods/{directories[method]}/rebench_v1.py")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("DEFAULT_SEARCH = ({'lr': 0.0001, 'weight_decay': 0.01},)\nraise AssertionError('must not import model or read weights')\n")
    dataset_map = {"motor_imagery": "eeg_fnirs_single_trial", "mental_arithmetic": "eeg_fnirs_single_trial",
                   "wg": "simultaneous_eeg_nirs", "nback": "simultaneous_eeg_nirs", "dsr": "simultaneous_eeg_nirs",
                   "visual": "visual_cognitive_motivation", "refed_regression": "refed"}
    tasks = {task: {"dataset_id": dataset, "sample_count": 1} for task, dataset in dataset_map.items()}
    scheduler.atomic_json(run / "inventory.json", {"schema": "downstream_rebench_inventory_v1", "tasks": tasks,
                         "cache_root": "/does-not-exist/no-signal-read", "samples": [{"path": "/does-not-exist/no-signal-read"}]})
    scheduler.atomic_json(run / "pretraining_inventory.json", {"schema": "downstream_rebench_inventory_v1", "inventory_kind": "pretraining", "tasks": tasks})
    scheduler.atomic_json(run / "splits.json", {"schema": "downstream_rebench_subject_folds_v1", "seed": config["split"]["seed"],
                         "outer_fold_count": 5, "inner_fold_count": 3,
                         "tasks": {task: {"folds": [{"inner_folds": [{}, {}, {}]} for _ in range(5)]} for task in tasks}})
    for task in tasks:
        scheduler.atomic_json(run / "input_audit_v2" / task / "result.json", {"status": "completed", "audit_pass": True,
                              "sample_count": 1, "full_support_sample_count": 1, "shape_failure_count": 0, "nonfinite_valid_point_count": 0,
                              "modalities": {"eeg": {"total_point_count": 10}, "fnirs": {"total_point_count": 10}}})
    resources = run / "resources.json"
    scheduler.atomic_json(resources, {"methods": {method: {"microbatch_size": 1, "gpu_memory_mib": 0 if method in {"signal_linear", "brainfusion"} else 1000,
                                                        "ram_gib": 1, "cpu_threads": 1} for method in config["methods"]},
                                     "pretraining": {"microbatch_size": 2, "gpu_memory_mib": 2000, "ram_gib": 2, "cpu_threads": 1}})
    return config_path, run, source, resources


def test_build_full_plan_is_metadata_only_and_preserves_dependency_contracts(build_fixture):
    config_path, run, source, resources = build_fixture
    manifest = scheduler.build_schedule(config_path, run, source, resources)
    assert len(manifest["jobs"]) == 988
    assert len([node for node in manifest["jobs"] if node["kind"] == "evaluate"]) == 975
    assert len([node for node in manifest["jobs"] if node["kind"] == "pretrain"]) == 12
    assert len(manifest["unsupported_cells"]) == 6
    ordered = manifest["jobs"]
    assert all(node["kind"] == "pretrain" for node in ordered[:12])
    assert all(node["track"] == "lp" and node["resources"]["gpu"] for node in ordered[12:237])
    assert all(node["method"] == "signal_linear" for node in ordered[237:342])
    assert all(node["track"] == "supervised" for node in ordered[342:522])
    assert all(node["track"] == "ft" for node in ordered[522:-1])
    positions = {node["id"]: index for index, node in enumerate(ordered)}
    assert all(positions[dependency] < positions[node["id"]] for node in ordered for dependency in node["dependencies"])
    loaded, _ = scheduler.load_manifest(run / "jobs.json")
    assert loaded == manifest
    nodes = {node["id"]: node for node in manifest["jobs"]}
    efrm = nodes["efrm-ft-visual-outer0-seed42"]
    job = scheduler.read_json(Path(efrm["job_path"]))
    assert efrm["dependencies"] == ["efrm-pretrain-exclude_visual_cognitive_motivation-seed42"]
    assert "reuse_selection_from" not in job
    assert job["method_profile"]["pretrained_checkpoint"].endswith("pretrain/exclude_visual_cognitive_motivation/seed42/pretrained_checkpoint.pt")
    assert job["microbatch_size"] == 1 and job["batch_size"] == 32
    official = nodes["biot-ft-visual-outer0-seed42"]
    assert nodes["brainfusion-supervised-visual-outer0-seed17"]["resources"]["gpu"] is False
    official_job = scheduler.read_json(Path(official["job_path"]))
    assert official["dependencies"] == ["biot-ft-visual-outer0-seed17"]
    assert official_job["reuse_selection_from"].endswith("ft/visual/outer0/seed17")
    assert len(official_job["search"]["candidates"]) == 6
    reve = scheduler.read_json(Path(nodes["reve-lp-motor_imagery-outer0-seed17"]["job_path"]))
    assert reve["target_corpus_overlap"] == "confirmed_Shin2017A_overlap"
    aggregate = nodes["aggregate"]
    assert aggregate["dependency_policy"] == "terminal" and not aggregate["resources"]["gpu"]
    assert set(aggregate["dependencies"]) == set(nodes) - {"aggregate"}
    assert all(node["timeout_seconds"] == 365 * 24 * 3600 for node in nodes.values())
    assert all(scheduler.read_json(Path(node["job_path"]))["inventory_path"] == str(run / "inventory.json") for node in nodes.values() if "job_path" in node)
    assert not (run / "scheduler_status.json").exists()
    assert not (run / "aggregate").exists()
    # Identical build is repeatable; a modified contract cannot overwrite a plan.
    assert scheduler.build_schedule(config_path, run, source, resources) == manifest
    retained = scheduler.read_json(Path(official["job_path"]))
    retained["seed"] = 999
    scheduler.atomic_json(Path(official["job_path"]), retained)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        scheduler.build_schedule(config_path, run, source, resources)


def test_build_refuses_eeg_only_audit_and_retains_partial_native_support(build_fixture):
    config_path, run, source, resources = build_fixture
    path = run / "input_audit_v2/visual/result.json"
    audit = scheduler.read_json(path)
    audit["modalities"]["fnirs"]["total_point_count"] = 0
    scheduler.atomic_json(path, audit)
    with pytest.raises(ValueError, match="nonempty EEG and fNIRS"):
        scheduler.build_schedule(config_path, run, source, resources)
    assert not (run / "jobs.json").exists()
    audit["modalities"]["fnirs"]["total_point_count"] = 10
    audit["full_support_sample_count"] = 0
    scheduler.atomic_json(path, audit)
    manifest = scheduler.build_schedule(config_path, run, source, resources)
    assert manifest["input_support"]["visual"]["partial_support_sample_count"] == 1
    assert manifest["input_support"]["visual"]["dropped_sample_count"] == 0
