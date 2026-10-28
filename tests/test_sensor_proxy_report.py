"""Report-specific denominator and illustrative-selection checks, no run fixtures."""
import pytest
import fitz

from experiments.scripts.render_shared_driver_sensor_proxy import aggregate, select_examples, Document


def test_report_preserves_record_weight_and_failure_denominator():
    rows=[dict(subject='s1',native_record='r1',nmse=1.,success=True),
          dict(subject='s1',native_record='r1',nmse=3.,success=True),
          dict(subject='s1',native_record='r2',nmse=30.,success=False),
          dict(subject='s2',native_record='r3',nmse=4.,success=True)]
    full,subjects=aggregate(rows,penalty=100.)
    assert subjects=={'s1':51.,'s2':4.}
    assert full==27.5
    success,_=aggregate(rows)
    assert success==3.
    assert full != (1+3+100+4)/4


def test_examples_use_common_support_and_distinct_subject_rank():
    rows=[dict(id=str(i),subject='s'+str(i//2)) for i in range(8)]
    tasks={(r['id'],'pc1','prediction','paired'):dict(nmse=i) for i,r in enumerate(rows)}
    result=select_examples(rows,tasks,{str(i) for i in range(7)})
    assert [v['row']['id'] for v in result]==['0','3','6']
    assert {v['denominator'] for v in result}=={7}
    assert [v['rank'] for v in result]==[1,4,7]
    assert len({v['row']['subject'] for v in result})==3


def test_examples_do_not_silently_drop_insufficient_common_support():
    with pytest.raises(ValueError,match='three common-success'):
        select_examples([],{},set())


def test_pdf_wrapping_keeps_scores_and_percentages_intact():
    # Built-in font avoids requiring the report's external CJK font in unit tests.
    report=Document.__new__(Document)
    report.font=fitz.Font('helv')
    text='共同窗口的误差0.26621，相对改善12.73%，不能改写分母。'
    lines=report.wrap(text,75,10)
    assert ''.join(lines)==text
    assert any('0.26621' in line for line in lines)
    assert any('12.73%' in line for line in lines)
