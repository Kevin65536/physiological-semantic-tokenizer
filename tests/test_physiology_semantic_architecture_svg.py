import copy
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from experiments.scripts.render_physiology_semantic_architecture import render_svg


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.drawio"
)
SVG_PATH = PROJECT_ROOT / "docs/physiology_semantic_tokenizer/figures/physiology_semantic_architecture.svg"
RUNTIME_OVERVIEW_DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_runtime_overview.drawio"
)
RUNTIME_OVERVIEW_SVG_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/physiology_semantic_runtime_overview.svg"
)
SPEC_SOURCE = "docs/physiology_semantic_tokenizer/architecture/physiology_semantic_architecture.json"
EXPLORATION_SPEC_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json"
)
EXPLORATION_DRAWIO_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.drawio"
)
EXPLORATION_SVG_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/plans/observation_source_exploration_v2.svg"
)
EXPLORATION_SPEC_SOURCE = (
    "docs/physiology_semantic_tokenizer/architecture/observation_source_exploration_v2.json"
)
DISCOVERY_SPEC_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/architecture/pst_discovery_v1_experiment_plan.json"
)
DISCOVERY_SVG_PATH = PROJECT_ROOT / (
    "docs/physiology_semantic_tokenizer/figures/pst_discovery_v1_experiment_plan.svg"
)
DISCOVERY_SPEC_SOURCE = (
    "docs/physiology_semantic_tokenizer/architecture/pst_discovery_v1_experiment_plan.json"
)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _spec():
    return _load(SPEC_PATH)


def _exploration_spec():
    return _load(EXPLORATION_SPEC_PATH)


def _discovery_spec():
    return _load(DISCOVERY_SPEC_PATH)


def _xml(svg: str):
    return ET.fromstring(svg)


def _drawio_cells(root):
    return {
        cell.attrib["id"]: (cell.attrib.get("value", ""), cell.attrib.get("style", ""))
        for cell in root.iter("mxCell")
        if "id" in cell.attrib
    }


def test_runtime_renderer_preserves_baseline_content_and_exposes_visual_axes():
    root = _xml(render_svg(_spec(), spec_source=SPEC_SOURCE))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert root.find("svg:title", namespace).text == "Physiology-Semantic Tokenizer — Current Runtime Architecture"
    assert "Implemented E2-compatible runtime" in root.find("svg:desc", namespace).text
    assert "fixed K=128 health passed" in root.find("svg:desc", namespace).text
    assert "no E2 teacher semantic row admitted" in root.find("svg:desc", namespace).text

    required = {
        "loader",
        "teacher_adapter",
        "eeg_quantizer",
        "fnirs_quantizer",
        "eeg_context",
        "fnirs_context",
        "trainer_gate",
        "export",
        "consumers",
        "p6_coupling",
    }
    xml_ids = {element.attrib.get("id") for element in root.iter()}
    assert {f"node-{node_id}" for node_id in required}.issubset(xml_ids)
    assert root.find(".//*[@id='node-trainer_gate']").attrib["data-status"] == "guarded"
    assert root.find(".//*[@id='node-p6_coupling']").attrib["data-status"] == "blocked"


def test_discovery_plan_svg_is_current_and_preserves_the_gate_path():
    rendered = render_svg(_discovery_spec(), spec_source=DISCOVERY_SPEC_SOURCE)
    assert rendered == DISCOVERY_SVG_PATH.read_text(encoding="utf-8")
    root = _xml(rendered)
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace).text.startswith("PST-DISCOVERY-v1")
    assert "P0 SUITE READY" in rendered
    assert "measured/protected 数据保持关闭" in rendered
    assert "fNIRS observation residual" in rendered
    coupling = root.find(".//*[@id='node-c0']")
    assert "fNIRS observation residual" in " ".join(coupling.itertext())
    assert 'class="banner-box" fill="#FFF4E5"' in rendered
    for phrase in (
        "可辨识 r + s/f/v/p/q",
        "独立 EEG / fNIRS LDS",
        "T2a/b · Croce 参照",
        "T2a · 论文式 PF",
        "T2b · 旧 adaptive 基线",
        "T3a · 鲁棒 Balloon",
        "Tak Eq.3 · tau_v=0 · Student-t",
        "主晋级候选",
        "T3b · 残差/PPC 失败",
        "T3c · ID通过但跨人不稳",
        "T5 · T-G4通过",
        "逐个加入 · 重走 gate",
        "T-G2 SBC / profile / 可辨识",
        "T-G5 T5 空间 / 稳定性",
        "MSE / R² 仅描述",
        "HRF / DCM-lite / SLDS",
    ):
        assert phrase in rendered
    assert "Functional role" not in rendered
    assert "micro-badge" not in rendered
    required = {
        "node-p0",
        "node-teacher_qualified",
        "node-continuous_tokenizer",
        "node-conditional_vq",
        "node-coupling_screen",
        "node-freeze",
        "node-conditional_teacher_extensions",
        "node-diag_teacher",
        "node-diag_tokenizer",
        "node-diag_coupling",
    }
    assert required <= {element.attrib.get("id") for element in root.iter()}
    assert root.find(".//*[@id='node-p0']").attrib["data-implementation"] == "implemented"


def test_legacy_geometry_is_expanded_in_memory_without_mutating_design_source():
    spec = _spec()
    before = copy.deepcopy(spec)
    root = _xml(render_svg(spec))
    assert spec == before
    assert int(root.attrib["width"]) > int(spec["width"])
    assert int(root.attrib["height"]) > int(spec["height"])


def test_generic_v2_overlay_composes_after_state_without_updating_any_plan_file():
    changes = {
        "schema": "physiology_semantic_architecture_changes_v2",
        "plan_id": "rendering_test_only",
        "title": "Proposed After-State · Rendering Test",
        "subtitle": "in-memory test overlay",
        "summary": "test",
        "changes": [
            {
                "node_id": "eeg_quantizer",
                "kind": "modify",
                "note": "exercise visual replacement",
                "replace": {"label": "Proposed vocabulary", "implementation": "planned"},
            }
        ],
        "add_nodes": [],
        "edge_changes": [],
        "add_edges": [],
    }
    spec = _spec()
    before = copy.deepcopy(spec)
    root = _xml(render_svg(spec, changes))
    assert spec == before
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.find("svg:title", namespace).text == "Proposed After-State · Rendering Test"
    node = root.find(".//*[@id='node-eeg_quantizer']")
    assert node.attrib["data-change-kind"] == "modify"
    assert node.attrib["data-canonical-status"] == "implemented"
    assert node.attrib["data-implementation"] == "planned"


def test_edges_have_stable_accessible_ids_and_orthogonal_paths():
    root = _xml(render_svg(_spec(), spec_source=SPEC_SOURCE))
    edge_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("edge-")
    ]
    edge_ids = [element.attrib["id"] for element in edge_groups]
    assert edge_groups
    assert len(edge_ids) == len(set(edge_ids))
    for group in edge_groups:
        assert group.attrib["aria-labelledby"]
        path = next(child for child in group if child.tag.endswith("path") and child.attrib.get("marker-end"))
        assert " L " in path.attrib["d"]
        assert " C " not in path.attrib["d"]


def test_validation_rejects_bad_replacement_unknown_endpoint_and_duplicate_ids():
    bad_replace = {
        "schema": "physiology_semantic_architecture_changes_v2",
        "plan_id": "bad",
        "summary": "bad",
        "changes": [
            {
                "node_id": "eeg_quantizer",
                "kind": "modify",
                "note": "bad",
                "replace": {"id": "replacement_is_forbidden"},
            }
        ],
    }
    with pytest.raises(ValueError, match="Unsupported node replacement fields"):
        render_svg(_spec(), bad_replace)

    bad_endpoint = copy.deepcopy(_spec())
    bad_endpoint["edges"][0]["to"] = "missing"
    with pytest.raises(ValueError, match="Unknown edge endpoint"):
        render_svg(bad_endpoint)

    duplicate = copy.deepcopy(_spec())
    duplicate["nodes"][1]["id"] = duplicate["nodes"][0]["id"]
    with pytest.raises(ValueError, match="node ids must be unique"):
        render_svg(duplicate)


@pytest.mark.parametrize(
    ("drawio_path", "svg_path"),
    [
        (DRAWIO_PATH, SVG_PATH),
        (RUNTIME_OVERVIEW_DRAWIO_PATH, RUNTIME_OVERVIEW_SVG_PATH),
        (EXPLORATION_DRAWIO_PATH, EXPLORATION_SVG_PATH),
    ],
)
def test_drawio_owned_svgs_embed_current_source(drawio_path, svg_path):
    source_root = ET.parse(drawio_path).getroot()
    svg_root = _xml(svg_path.read_text(encoding="utf-8"))
    embedded_root = ET.fromstring(svg_root.attrib["content"])
    assert _drawio_cells(embedded_root) == _drawio_cells(source_root)
    assert svg_root.attrib["role"] == "img"
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert svg_root.find("svg:title", namespace) is not None
    assert svg_root.find("svg:desc", namespace) is not None
    styles = " ".join(style for _, style in _drawio_cells(source_root).values())
    assert "fontFamily=Helvetica" in styles
    assert "rounded=1" in styles
    assert "shadow=1" not in styles
    values = " ".join(value for value, _ in _drawio_cells(source_root).values())
    assert "PID-MCM" not in values
    assert "planned target" not in values.lower()
    assert "private" not in values.lower()
    if drawio_path == RUNTIME_OVERVIEW_DRAWIO_PATH:
        assert "semantic + residual reconstructions" in values
    elif drawio_path == EXPLORATION_DRAWIO_PATH:
        assert "EEG observation path" in values
        assert "fNIRS observation path" in values
    else:
        assert "EEG observation reconstruction" in values
        assert "fNIRS observation reconstruction" in values
    if drawio_path == DRAWIO_PATH:
        assert "Optional contribution probe" in values
        assert "architecture remains revisable" in values
        assert "fillColor=#F2F7FF;strokeColor=#3B73E8" in styles
        assert "fillColor=#FFF4F6;strokeColor=#FF5067" in styles
        assert "fillColor=#F0FBFA;strokeColor=#168C83" in styles
        assert "fillColor=#FFF7E8;strokeColor=#F28B2B" in styles
        assert "fillColor=#F5F0FF;strokeColor=#7B56D8" in styles


def test_renderer_mirrors_the_detailed_architecture_visual_language():
    svg = render_svg(_discovery_spec(), spec_source=DISCOVERY_SPEC_SOURCE)
    assert 'font-family: Helvetica' in svg
    assert '.banner-box { fill: #FFF4E5; stroke: #F28B2B;' in svg
    assert '.section-box { fill: #FAFBFD; stroke: #667085;' in svg
    assert 'class="section-box" fill="#FAFBFD"' in svg
    assert 'rx="16"' in svg
    assert '#F2F7FF' in svg
    assert '#F5F0FF' in svg
    assert '#FFF7E8' in svg
    assert '#EDF9EF' in svg
    assert "Functional role" not in svg
    assert "micro-badge" not in svg


def test_v2_exploration_visual_is_distinct_from_runtime():
    exploration = _exploration_spec()
    assert exploration["schema"] == "physiology_semantic_architecture_v2"
    assert EXPLORATION_DRAWIO_PATH.exists()
    assert EXPLORATION_SVG_PATH != SVG_PATH


def test_v2_exploration_keeps_paths_independent_without_freezing_decomposition():
    exploration = _exploration_spec()
    edges = {edge["id"]: edge for edge in exploration["edges"]}
    required_edges = {
        "measured-x-e--target-adapter-e",
        "measured-x-f--target-adapter-f",
        "target-adapter-e--self-teacher-e",
        "target-adapter-f--self-teacher-f",
        "croce-joint-candidate--source-inference-e",
        "croce-joint-candidate--source-inference-f",
        "self-teacher-e--source-inference-e",
        "self-teacher-e--observation-inference-e",
        "self-teacher-f--source-inference-f",
        "self-teacher-f--observation-inference-f",
        "fine-codebook-e--coarse-aggregator-e",
        "fine-codebook-f--coarse-aggregator-f",
        "coarse-aggregator-e--endpoint-aligned-g-tau",
        "coarse-aggregator-f--endpoint-aligned-g-tau",
        "endpoint-aligned-g-tau--conditional-probe",
        "observation-e--observation-objective",
        "observation-f--observation-objective",
        "observation-objective--conditional-probe",
        "endpoint-aligned-g-tau--evaluation-preregistration",
        "preregister--heldout-proper-score-null",
    }
    assert required_edges.issubset(edges)
    assert len(edges) == len(exploration["edges"])
    assert len(edges) == len({edge["id"] for edge in exploration["edges"]})
    assert all(edge.get("route") for edge in exploration["edges"])
    assert edges["measured-x-e--croce-joint-candidate"]["style"] == "guarded"
    assert edges["measured-x-f--croce-joint-candidate"]["style"] == "guarded"
    assert edges["croce-joint-candidate--source-inference-e"]["style"] == "guarded"
    assert edges["croce-joint-candidate--source-inference-f"]["style"] == "guarded"
    assert "no inference input" in edges[
        "croce-joint-candidate--source-inference-e"
    ]["label"]
    assert edges["preregister--heldout-proper-score-null"]["style"] == "evaluation"
    assert not any(
        edge["from"] == "conditional_contribution_probe"
        and edge["to"] == "evaluation_preregistration"
        for edge in exploration["edges"]
    )
    node_ids = {node["id"] for node in exploration["nodes"]}
    assert {
        "measured_x_e",
        "measured_x_f",
        "target_adapter_e",
        "target_adapter_f",
        "self_teacher_e",
        "self_teacher_f",
        "croce_joint_candidate",
        "source_inference_e",
        "observation_inference_e",
        "source_inference_f",
        "observation_inference_f",
        "fine_codebook_e",
        "coarse_aggregator_e",
        "fine_codebook_f",
        "coarse_aggregator_f",
        "endpoint_aligned_g_tau",
        "observation_objective",
        "conditional_contribution_probe",
        "evaluation_preregistration",
        "heldout_proper_score_null",
    }.issubset(node_ids)


def test_v2_exploration_status_axes_and_accessibility_are_explicit():
    exploration = _exploration_spec()
    assert all(node["implementation"] == "planned" for node in exploration["nodes"])
    croce = next(
        node for node in exploration["nodes"] if node["id"] == "croce_joint_candidate"
    )
    assert croce["scope"] == "training_only"
    assert croce["role"] == "teacher"
    probe = next(
        node
        for node in exploration["nodes"]
        if node["id"] == "conditional_contribution_probe"
    )
    assert probe["evidence"] == "n_a"
    assert "may be replaced or removed" in " ".join(probe["details"])
    assert exploration["title"] == "Observation–Source Candidate Exploration"
    root = _xml(render_svg(exploration, spec_source=EXPLORATION_SPEC_SOURCE))
    namespace = {"svg": "http://www.w3.org/2000/svg"}
    assert root.attrib["role"] == "img"
    assert "Replaceable candidates" in root.find("svg:desc", namespace).text
    boundary = root.find(".//*[@id='evidence-boundary']")
    assert boundary is not None
    assert "no method identity or decomposition is frozen" in "".join(boundary.itertext())
    node_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("node-")
    ]
    assert len(node_groups) == len(exploration["nodes"])
    for group in node_groups:
        assert group.attrib["aria-labelledby"]
        assert group.find("svg:title", namespace) is not None
        assert group.find("svg:desc", namespace) is not None
        assert group.attrib["data-implementation"] == "planned"
        assert group.attrib["data-evidence"] in {"guarded", "n_a"}
    edge_groups = [
        element
        for element in root.iter()
        if element.tag.endswith("g") and element.attrib.get("id", "").startswith("edge-")
    ]
    assert len(edge_groups) == len(exploration["edges"])
    assert len({group.attrib["id"] for group in edge_groups}) == len(edge_groups)
    for group in edge_groups:
        assert group.attrib["aria-labelledby"]
        assert group.find("svg:title", namespace) is not None
        assert group.find("svg:desc", namespace) is not None
        path = next(
            child
            for child in group
            if child.tag.endswith("path") and child.attrib.get("marker-end")
        )
        assert " L " in path.attrib["d"]
        assert " C " not in path.attrib["d"]
