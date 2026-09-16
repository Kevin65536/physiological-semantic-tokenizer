import numpy as np
import pytest

from experiments.scripts import analyze_dataset_scaling as report


def test_report_scope_rejects_protected_subject_before_read():
    c=report.config();c['datasets']['eeg_fnirs_single_trial']['subjects'][-1]='subject_24'
    with pytest.raises(ValueError,match='scope'):report.validate_scope(c)


def test_report_scaling_retains_canonical_prediction_and_negative_control():
    from experiments.evaluate_step5a_inference_consistency import load_config,generate_matched
    b=load_config();b['model']['steps']=32;g=generate_matched(b,'W',0.,20692500)
    task=dict(kind='synthetic',identity='fixture',subject='synthetic',base=b,arm='baseline',mode='center_fNIRS',
              y=g['observations'],target=g['clean'],truth=g['states'],sd=np.std(g['clean'],0),
              separate=[.8,2.],shared=.5,quadrature=7)
    a=report.fit_case(task);assert a['status']=='completed'
    for arm in ['shared_synchronized','separate_synchronized']:
        changed=report.fit_case(dict(task,arm=arm));assert changed['status']=='completed'
        np.testing.assert_allclose(changed['state'],a['state'],atol=1e-10)
        np.testing.assert_allclose(changed['prediction'],a['prediction'],atol=1e-10)
        assert abs(changed['log_likelihood']-a['log_likelihood'])<1e-8
    wrong=report.fit_case(dict(task,arm='separate_data_only'))
    assert wrong['status']!='completed' or np.max(abs(wrong['state']-a['state']))>1e-5


def test_report_amplitude_statistics_keep_missing_and_raw_maximum():
    x=np.array([[1.,np.nan],[-2.,np.nan],[3.,np.nan]])
    a,b=report.stats_rows(x,['a','b'],['HbO','HbR'],{})
    assert a['max_abs']==3 and a['peak_to_peak']==5
    assert b['finite_fraction']==0 and np.isnan(b['sd'])


@pytest.mark.parametrize('size', [(600, 200), (200, 600)])
def test_pdf_figure_is_bitmap_with_preserved_shape_and_searchable_caption(tmp_path, size):
    import fitz
    from PIL import Image

    png = tmp_path / 'figure.png'
    Image.new('RGB', size, 'white').save(png)
    with fitz.open() as doc:
        page, height = report.append_bitmap_figure(doc, png)
        page.insert_text((42, 60 + height), 'Searchable caption')
        with fitz.open(stream=doc.tobytes(), filetype='pdf') as saved:
            page = saved[0]
            assert len(page.get_images()) == 1
            assert not page.get_drawings()
            rect = page.get_image_rects(page.get_images()[0][0])[0]
            assert page.rect.contains(rect)
            assert rect.width / rect.height == pytest.approx(size[0] / size[1])
            assert 'Searchable caption' in page.get_text()
            page.get_pixmap()


def test_report_refuses_to_overwrite_retained_export(tmp_path):
    pdf = tmp_path / 'REPORT.pdf'
    pdf.write_bytes(b'retained evidence')
    with pytest.raises(RuntimeError, match='immutable'):
        report.render_report({}, tmp_path)
    assert pdf.read_bytes() == b'retained evidence'


def test_alignment_comparison_keeps_failures_and_pairs_by_identity():
    import pandas as pd

    old = pd.DataFrame([
        dict(sample_id='a', mode='full', solver='O2', status='completed'),
        dict(sample_id='b', mode='full', solver='O2', status='failed_numerical'),
    ])
    new = pd.DataFrame([
        dict(sample_id='b', mode='full', solver='O2', status='completed'),
        dict(sample_id='a', mode='full', solver='O2', status='failed_numerical'),
        dict(sample_id='c', mode='full', solver='O2', status='failed_domain'),
        dict(sample_id=None, mode=None, solver=None, status='diagnostic_summary'),
    ])
    candidate, reference, common, transitions = report.alignment_fit_comparison(new, old)
    assert len(candidate) == 3 and len(reference) == 2 and len(common) == 2
    assert common.set_index('sample_id').loc['a', 'status_candidate'] == 'failed_numerical'
    assert set(zip(transitions.status_reference, transitions.status_candidate)) == {
        ('completed', 'failed_numerical'), ('failed_numerical', 'completed')}
    with pytest.raises(ValueError, match='duplicate'):
        report.alignment_fit_comparison(pd.concat([new, new.iloc[:1]]), old)
    with pytest.raises(ValueError, match='missing a retained'):
        report.alignment_fit_comparison(new.iloc[1:], old)


def test_alignment_export_refuses_existing_directory_before_evidence_read(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'ROOT', tmp_path)
    out = tmp_path / 'experiments/runs/physiology_semantic_tokenizer/data_quality_audit/retained'
    out.mkdir(parents=True)
    (out / 'REPORT.md').write_text('retained report')
    with pytest.raises(RuntimeError, match='immutable'):
        report.render_alignment_report(out)
    assert (out / 'REPORT.md').read_text() == 'retained report'
