import numpy as np
import pytest
from src.tokenizers.hb_parent_leaf import HbParentLeaf, fit_children, assign_leaf, synthesize_parent, emitted_error_polynomials


class Parent:
    def decode_ids(self, ids):
        s = np.repeat(ids["S"][:, :, None], 40, axis=2).astype(np.float32) * 0.01
        o = np.repeat(ids["O"][:, :, None], 40, axis=2).astype(np.float32) * 0.02
        return dict(S=s, O=o, sum=s + o)


def test_supervised_leaf_is_one_atom_and_returns_its_fixed_parent():
    n = 100
    ids = np.zeros(n, int)
    features = np.arange(n, dtype=float)[:, None]
    target = np.zeros((n, 42))
    target[50:] = 1.0
    settings = dict(
        minimum_leaf_complete_training_patches=20,
        minimum_leaf_training_subjects=2,
        max_leaves=2,
        minimum_parent_supervised_impurity_reduction=0.05,
    )
    fitted = fit_children(
        ids,
        features,
        target,
        np.zeros((n, 40)),
        np.array([str(i % 10) for i in range(n)]),
        settings,
        13,
    )
    leaf = assign_leaf(ids.reshape(10, 10), features.reshape(10, 10, 1), fitted)
    assert len(np.unique(leaf)) == 2
    np.testing.assert_array_equal(fitted["parent"][leaf], 0)


def test_parent_sum_is_exact_while_leaf_changes_component_and_quiet_is_explicit():
    fitted = dict(
        parent=np.array([3, 3]), residual=np.tile(np.sin(np.arange(40)), (2, 1))
    )
    metadata = dict(
        branch_codebook_sizes={"S": 129, "O": 3},
        quiet_threshold=0.01,
        quiet_level_S=[0.1, 0.2],
        color_gain=[1.05, 1.1],
        residual_strength=0.25,
        residual_lowpass_hz=0.4,
    )
    model = HbParentLeaf(Parent(), metadata, fitted)
    ids = dict(S=np.ones((1, 15), int), O=np.ones((1, 15), int))
    original = Parent().decode_ids(dict(S=ids["S"], O=np.full((1, 15), 3)))
    pred = model.decode_ids(ids)
    np.testing.assert_array_equal(pred["sum"], original["sum"])
    np.testing.assert_allclose(pred["S"] + pred["O"], pred["sum"], rtol=0, atol=1e-14)
    assert not np.array_equal(pred["O"], original["O"])
    quiet = model.decode_ids(dict(S=np.full((1, 15), 128), O=np.full((1, 15), 2)))
    np.testing.assert_array_equal(quiet["sum"], 0.0)
    np.testing.assert_array_equal(quiet["S"], -quiet["O"])
    with pytest.raises(ValueError, match="complete reserved pair"):
        model.decode_ids(dict(S=np.full((1, 15), 128), O=np.ones((1, 15), int)))


def test_train_fitted_multivariate_direction_uses_frozen_X_projection_at_assignment():
    rng=np.random.default_rng(9)
    features=rng.normal(size=(512,8))
    target=np.zeros((512,42));target[:,40]=features[:,0]+features[:,1]
    settings=dict(minimum_leaf_complete_training_patches=64,minimum_leaf_training_subjects=2,
                  max_leaves=4,minimum_parent_supervised_impurity_reduction=.05,
                  feature_mode='ridge',direction_ridge=1.,HbO_supervision_weight=4.)
    fitted=fit_children(np.zeros(512,int),features,target,np.zeros((512,40)),
                        np.array([str(i%8) for i in range(512)]),settings,17)
    assert fitted['directions'][0]['coefficient'].shape==(8,2)
    assert fitted['trees'][0].n_features_in_==10
    first=assign_leaf(np.zeros(512,int),features,fitted)
    # Inference receives no supervised target or per-window metadata.
    np.testing.assert_array_equal(first,assign_leaf(np.zeros(512,int),features.copy(),fitted))
    np.testing.assert_array_equal(fitted['parent'][first],0)
    assert len(np.unique(first))==4


def test_parent_sum_comparison_preserves_the_frozen_parent_batch_path():
    class BatchSensitiveParent(Parent):
        def decode_ids(self,ids):
            p=super().decode_ids(ids)
            p['sum']=p['sum']+np.float32(len(ids['S'])*1e-6)
            return p
    parent=BatchSensitiveParent()
    fitted=dict(parent=np.array([3]),residual=np.zeros((1,40)))
    metadata=dict(branch_codebook_sizes={'S':129,'O':2},quiet_level_S=[.1,.2],
                  color_gain=[1.05,1.1],residual_strength=.25,residual_lowpass_hz=.4)
    model=HbParentLeaf(parent,metadata,fitted)
    ids=dict(S=np.ones((2,15),int),O=np.zeros((2,15),int))
    expected=parent.decode_ids(dict(S=ids['S'],O=np.full((2,15),3)))
    np.testing.assert_array_equal(model.decode_ids(ids)['sum'],expected['sum'])
    one={b:v[:1] for b,v in ids.items()}
    assert not np.array_equal(model.decode_ids(one)['sum'],expected['sum'][:1])


def test_fit_expansion_preserves_original_people_windows_and_native_nonoverlap(monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    from run_hb_parent_leaf import expansion_plan
    rows=[]
    for person in ['fitted','unrequested']:
        for record in ['a','b']:
            rows.append(dict(dataset='d',subject=person,record=record,positions=['physical'],
                             native_start_sample=0,native_rate_hz=10,split='train',modality='hb',
                             inputs={'score':[dict(start_s=t,positions=[0]) for t in [0,30,60]]}))
    # This alternate cached parent overlaps the original physical interval.
    rows.append(dict(rows[0],native_start_sample=100,
                     inputs={'score':[dict(start_s=0,positions=[0])]}))
    old=np.array([[0,0,0]],np.int32);spec=dict(selection_seed=17,maximum_windows_per_subject=5)
    expanded,summary=expansion_plan(rows,old,spec)
    np.testing.assert_array_equal(expanded[:1],old)
    assert len(expanded)==5 and all(rows[i]['subject']=='fitted' for i,_,_ in expanded)
    assert not any(i==4 for i,_,_ in expanded)
    assert summary['new_overlapping_windows']==0
    assert summary['subjects'][0]['expanded_records']==2
    assert abs(summary['subjects'][0]['by_record']['a']-summary['subjects'][0]['by_record']['b'])<=1
    np.testing.assert_array_equal(expanded,expansion_plan(rows,old,spec)[0])


def test_actual_gain_quadratic_contains_residual_quiet_and_masked_coordinates():
    n=4;shape=(n,15,40);quiet=np.array([True,False,False,False])
    source=dict(S=np.full(shape,2.,np.float32),O=np.ones(shape,np.float32),sum=np.full(shape,3.,np.float32))
    fitted=dict(residual=np.sin(np.arange(40)[None]/4))
    leaf=np.zeros((n,15),int)
    meta=dict(color_gain=[1.,1.],residual_strength=.25,residual_lowpass_hz=.4,quiet_level_S=[.1,.2])
    at_one=synthesize_parent(source,leaf,fitted,meta,quiet)
    baseline={b:v.copy() for b,v in source.items()}
    for b in 'SO':baseline[b][quiet]=np.tile(np.array(meta['quiet_level_S'])*(1 if b=='S' else -1),20)
    baseline['sum'][quiet]=0
    targets={b:v.reshape(n,300,2)+.1 for b,v in source.items()}
    mask=np.ones(shape,bool);mask[1,-3:,:]=False
    variance=np.ones((n,3,2));coeff=emitted_error_polynomials(baseline,at_one,targets,mask,variance,quiet)
    np.testing.assert_array_equal(coeff[quiet,...,:2],0.)
    assert (coeff[quiet,...,2]>0).all()  # quiet error is scored, not filtered.
    for gains in [[.9,1.12],[1.075,1.03]]:
        emitted=synthesize_parent(source,leaf,fitted,dict(meta,color_gain=gains),quiet)
        errors=np.stack([((np.asarray(emitted[b],float).reshape(n,300,2)-targets[b])**2) for b in ['sum','S','O']],axis=1)
        m=mask.reshape(n,300,2)[:,None];direct=np.where(m,errors,0).sum(2)/m.sum(2)
        delta=np.array(gains)-1
        polynomial=coeff[...,0]*delta[None,None]**2+coeff[...,1]*delta[None,None]+coeff[...,2]
        np.testing.assert_allclose(polynomial,direct,rtol=1e-12,atol=1e-12)


def test_fixed_residual_can_make_every_gain_infeasible_when_parent_error_is_zero():
    shape=(2,15,40);source=dict(S=np.full(shape,2.),O=np.ones(shape),sum=np.full(shape,3.))
    quiet=np.zeros(2,bool);leaf=np.zeros((2,15),int);fitted=dict(residual=np.sin(np.arange(40)[None]/4))
    meta=dict(color_gain=[1.,1.],residual_strength=.25,residual_lowpass_hz=.4,quiet_level_S=[0.,0.])
    p=synthesize_parent(source,leaf,fitted,meta,quiet)
    c=emitted_error_polynomials(source,p,{b:v.reshape(2,300,2) for b,v in source.items()},np.ones(shape,bool),np.ones((2,3,2)),quiet)
    np.testing.assert_array_equal(c[...,3],0.)
    grid=np.linspace(.9,1.12,45)-1
    score=c[...,0,None]*grid**2+c[...,1,None]*grid+c[...,2,None]
    assert not (score.mean(0)[1:]<=1e-12).all((0,1)).any()


def test_emitted_projection_executes_independent_integer_grid_checks(tmp_path,monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'experiments/scripts'))
    from run_hb_parent_leaf import constrained_emitted_gain
    parent=Parent();ids=dict(S=np.ones((64,15),int),O=np.full((64,15),3));original=parent.decode_ids(ids)
    bank=dict(mask=np.ones((64,15,40),bool),id_S=ids['S'],id_O=ids['O'])
    for b,t in [('sum','x'),('S','s'),('O','o')]:
        bank['original_'+b]=original[b];bank[t]=original[b].copy()
        bank['v'+t]=np.ones((64,2));bank['dv'+t]=np.ones((64,2))
    rows=[dict(dataset='synthetic',subject='one',record='r',parent='p',modality='hb',positions=['site'])]
    examples=np.array([[0,30*j,0] for j in range(64)])
    scales={'positions':{'hb':{'synthetic':{'site':{'scale':1.}}}}}
    cfg=dict(model={'residual_lowpass_hz':.4},selection={'frozen_numeric_components_ratio_max':1.05,'frozen_dataset_components_ratio_max':1.1})
    gain,review=constrained_emitted_gain(bank,np.arange(64),examples,rows,scales,np.zeros(64,bool),cfg,
        np.zeros((64,15),int),dict(parent=np.array([3]),residual=np.zeros((1,40))),
        dict(level_S=[0.,0.],level_O=[0.,0.]),parent,tmp_path,'synthetic')
    np.testing.assert_array_equal(gain,[1.,1.])
    import json
    audit=json.loads((tmp_path/'emitted_gain_fit_synthetic.json').read_text())
    assert audit['max_abs_grid_check_difference']<1e-10
    assert len(audit['independent_grid_checks'])==4
