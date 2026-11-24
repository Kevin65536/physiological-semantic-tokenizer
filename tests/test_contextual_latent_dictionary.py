import numpy as np
import pytest
import torch

from src.tokenizers.contextual_latent_dictionary import ContextualLatentDictionary, load_model
from src.tokenizers.dual_component_tokenizer import GlobalLatentComponentTokenizer


def test_integer_only_saved_replay_and_input_validation(tmp_path):
    torch.set_num_threads(1)
    setting = dict(input_dim=40, channels=2, patches=15, width=8, latent_dim=4,
                   heads=2, encoder_layers=1, decoder_layers=1, kmeans_init=False,
                   waveform_io=False, dropout=0.)
    model = GlobalLatentComponentTokenizer(setting, codebook_size=8, hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    metadata = dict(schema='contextual_latent_dictionary_v1', modality='hb', codebook_size=8)
    wrapped = ContextualLatentDictionary(model, metadata)
    signal = np.random.default_rng(3).normal(size=(2, 15, 40)).astype(np.float32)
    original = wrapped(signal)
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata, model_config=setting, model_state=model.state_dict()), path)
    np.savez(tmp_path/'ids.npz', **original['ids'])
    with np.load(tmp_path/'ids.npz') as retained:
        replay = load_model(path).decode_ids(dict(retained))
    for key in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(original[key], replay[key])
    with pytest.raises(ValueError, match='integer'):
        wrapped.decode_ids({key: value.astype(float) for key, value in original['ids'].items()})
    with pytest.raises(ValueError, match='only S/O'):
        wrapped.decode_ids(dict(original['ids'], scale=np.ones(2)))


def test_semantic_assignment_uses_frozen_reference_after_decoder_changes(tmp_path):
    from copy import deepcopy
    from src.tokenizers.contextual_latent_dictionary import SemanticContextualDictionary, hb_semantic_features
    torch.set_num_threads(1)
    setting = dict(input_dim=40, channels=2, patches=15, width=8, latent_dim=4,
                   heads=2, encoder_layers=1, decoder_layers=1, waveform_io=False, dropout=0.)
    model = GlobalLatentComponentTokenizer(setting, codebook_size=8, hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    reference = deepcopy(model)
    signal = np.random.default_rng(11).normal(size=(2, 15, 40)).astype(np.float32)
    assignment = {}
    with torch.inference_mode():
        prediction = reference(torch.as_tensor(signal), quantize=False)
    for key in ('S', 'O'):
        features = hb_semantic_features(prediction[key].numpy()).reshape(-1, 46)
        assignment.update({key+'_centers':features[:8], key+'_mean':np.zeros(46, dtype=np.float32),
                           key+'_scale':np.ones(46, dtype=np.float32), key+'_weight':np.ones(46, dtype=np.float32)})
    metadata = dict(schema='contextual_latent_dictionary_v2', modality='hb', codebook_size=8)
    wrapped = SemanticContextualDictionary(model, metadata, reference, assignment)
    ids = wrapped.encode(signal)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if '.output.' in name:
                parameter.add_(1.)
    for key in ('S', 'O'):
        np.testing.assert_array_equal(ids[key], wrapped.encode(signal)[key])
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata, model_config=setting, model_state=model.state_dict(),
                    reference_state=reference.state_dict(), assignment=assignment), path)
    loaded = load_model(path)
    for key in ('S', 'O', 'sum'):
        np.testing.assert_array_equal(wrapped.decode_ids(ids)[key], loaded.decode_ids(ids)[key])


def test_fixed_synthesis_preserves_branch_sum_after_gain_and_filter():
    from scipy.signal import butter, sosfiltfilt
    wrapper = object.__new__(ContextualLatentDictionary)
    wrapper.metadata = dict(modality='hb', decoder_color_gain=dict(S=[1.2, .8], O=[.9, 1.1]),
                            decoder_lowpass_hz=.4)
    time = np.arange(300)/10
    wave = np.stack((np.sin(2*np.pi*.1*time)+.2*np.sin(2*np.pi*2*time),
                     np.cos(2*np.pi*.15*time)), axis=-1)[None]
    prediction = dict(S=wave.reshape(1, 15, 40), O=(.3*wave).reshape(1, 15, 40))
    result = wrapper.synthesize(prediction)
    sections = butter(4, .4, fs=10, output='sos')
    for key in ('S', 'O'):
        expected = sosfiltfilt(sections, prediction[key].reshape(1,300,2)*wrapper.metadata['decoder_color_gain'][key], axis=1)
        np.testing.assert_array_equal(result[key], expected.reshape(1,15,40))
    np.testing.assert_array_equal(result['sum'], result['S']+result['O'])


def test_component_transfer_conserves_X_and_uses_both_frozen_integer_branches():
    wrapper=object.__new__(ContextualLatentDictionary)
    wrapper.metadata=dict(modality='hb',decoder_component_transfer_gain=[1.1,.95])
    rng=np.random.default_rng(91)
    original=dict(S=rng.normal(size=(2,15,40)).astype(np.float32),O=rng.normal(size=(2,15,40)).astype(np.float32))
    original['sum']=original['S']+original['O']
    result=wrapper.synthesize(original)
    np.testing.assert_array_equal(result['sum'],original['sum'])
    np.testing.assert_allclose(result['S']+result['O'],original['sum'],rtol=0,atol=1e-15)
    expected=(original['O'].reshape(2,300,2).astype(float)*[1.1,.95]).reshape(2,15,40)
    np.testing.assert_array_equal(result['O'],expected)


def test_unrestricted_context_checkpoint_reloads_exact_integer_decoder(tmp_path):
    from src.tokenizers.contextual_latent_dictionary import UnrestrictedHbContextDecoder
    setting = dict(input_dim=40, channels=2, patches=15, width=8, latent_dim=4,
        heads=2, encoder_layers=1, decoder_layers=1, waveform_io=True, dropout=0.,
        sample_strides=[5], local_width=4, hb_local_shape=True, context_ratio=.25,
        context_norm_epsilon=1e-8)
    model = UnrestrictedHbContextDecoder(setting,codebook_size=8,hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    ids = {key:np.random.default_rng(seed).integers(0,8,size=(2,15)) for key,seed in [('S',4),('O',5)]}
    metadata = dict(schema='contextual_latent_dictionary_v1',modality='hb',codebook_size=8,
                    decoder_architecture='unrestricted_hb_context_v1')
    wrapper = ContextualLatentDictionary(model,metadata)
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata,model_config=setting,model_state=model.state_dict()),path)
    restored = load_model(path)
    assert isinstance(restored.model,UnrestrictedHbContextDecoder)
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(wrapper.decode_ids(ids)[key],restored.decode_ids(ids)[key])


def test_single_id_prototype_and_full_decode_replay_without_continuous_sidecars(tmp_path):
    from src.tokenizers.contextual_latent_dictionary import PrototypeSupervisedHbTokenizer
    setting = dict(input_dim=40,channels=2,patches=15,width=8,latent_dim=4,
        heads=2,encoder_layers=1,decoder_layers=1,waveform_io=True,dropout=0.,sample_strides=[5],local_width=4)
    model = PrototypeSupervisedHbTokenizer(setting,codebook_size=8,hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    signal = torch.randn(2,15,40)
    with torch.inference_mode():
        result = model(signal)
        decoded = model.decode_ids(result['ids'])
    for key in ('S','O','S_prototype','O_prototype'):
        torch.testing.assert_close(result[key],decoded[key],atol=0.,rtol=0.)
    for key in ('S_prototype','O_prototype'):
        torch.testing.assert_close(result[key].reshape(2,15,20,2).mean(2),torch.zeros(2,15,2),atol=1e-7,rtol=0.)
    metadata = dict(schema='contextual_latent_dictionary_v1',modality='hb',codebook_size=8,
                    decoder_architecture='prototype_supervised_hb_v1')
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata,model_config=setting,model_state=model.state_dict()),path)
    assert isinstance(load_model(path).model,PrototypeSupervisedHbTokenizer)
    code = {key:value.numpy() for key,value in result['ids'].items()}
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(load_model(path).decode_ids(code)[key],decoded[key].numpy())


def test_distinct_branch_vocabularies_replay_and_report_true_rates(tmp_path):
    from src.tokenizers.contextual_latent_dictionary import PrototypeSupervisedHbTokenizer
    from experiments.scripts.review_joint_tokenizer import integer_rate
    setting = dict(codebook_size=8,input_dim=40,channels=2,patches=15,width=8,latent_dim=4,
        heads=2,encoder_layers=1,decoder_layers=1,waveform_io=True,dropout=0.,sample_strides=[5],local_width=4,
        branch_codebook_sizes=dict(S=8,O=16))
    model = PrototypeSupervisedHbTokenizer(setting,codebook_size=8,hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    assert model.branches['S']['quantizer'].codebook.shape == (8,4)
    assert model.branches['O']['quantizer'].codebook.shape == (16,4)
    ids = dict(S=np.tile(np.arange(15)%8,(2,1)),O=np.tile(np.arange(15)%16,(2,1)))
    metadata = dict(schema='contextual_latent_dictionary_v1',modality='hb',codebook_size=16,
        branch_codebook_sizes=dict(S=8,O=16),decoder_architecture='prototype_supervised_hb_v1')
    path = tmp_path/'model.pt'
    torch.save(dict(metadata=metadata,model_config=setting,model_state=model.state_dict()),path)
    np.savez(tmp_path/'ids.npz',**ids)
    wrapper = ContextualLatentDictionary(model,metadata)
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(wrapper.decode_ids(ids)[key],load_model(path).decode_ids(ids)[key])
    rate = integer_rate(path,tmp_path/'ids.npz')
    assert rate['branches']['S']['fixed_width_bits_per_second'] == 1.5
    assert rate['branches']['O']['fixed_width_bits_per_second'] == 2.
    with pytest.raises(ValueError,match='in-range'):
        wrapper.decode_ids(dict(S=np.full((2,15),8),O=ids['O']))


def test_semantic_search_preserves_S_improves_observed_X_and_replays_saved_ids(tmp_path):
    import hashlib
    from src.tokenizers.contextual_latent_dictionary import PrototypeSupervisedHbTokenizer
    from src.tokenizers.hb_semantic_analysis_synthesis import HbSemanticAnalysisSynthesis
    from experiments.scripts.review_joint_tokenizer import integer_rate
    torch.set_num_threads(1)
    setting = dict(codebook_size=8,input_dim=40,channels=2,patches=15,width=8,latent_dim=4,
        heads=2,encoder_layers=1,decoder_layers=1,waveform_io=True,dropout=0.,sample_strides=[5],local_width=4,
        branch_codebook_sizes=dict(S=8,O=16))
    model = PrototypeSupervisedHbTokenizer(setting,codebook_size=8,hard=True).eval()
    for branch in model.branches.values():
        branch['quantizer'].initialized.fill_(True)
    source_metadata = dict(schema='contextual_latent_dictionary_v1',modality='hb',codebook_size=16,
        branch_codebook_sizes=dict(S=8,O=16),decoder_architecture='prototype_supervised_hb_v1')
    source = ContextualLatentDictionary(model,source_metadata)
    neighbors = np.c_[np.arange(16),(np.arange(16)+1)%16]
    metadata = dict(source_metadata,schema='hb_semantic_analysis_synthesis_v1',neighbor_budget=2,
        coordinate_passes=1,maximum_normalized_semantic_drift=.25,source_file='source.pt')
    search = HbSemanticAnalysisSynthesis(source,metadata,neighbors,np.ones(2),np.ones(2),np.ones(2))
    observed = np.random.default_rng(49).normal(size=(2,15,40)).astype(np.float32)
    before,after = source(observed),search(observed)
    np.testing.assert_array_equal(before['ids']['S'],after['ids']['S'])
    np.testing.assert_array_equal(before['S'],after['S'])
    assert (search.error(after['sum'],observed)<=search.error(before['sum'],observed)).all()
    assert (search.semantic_drift(after['O'],before['O'])<=.25).all()
    torch.save(dict(metadata=source_metadata,model_config=setting,model_state=model.state_dict()),tmp_path/'source.pt')
    metadata['source_sha256'] = hashlib.sha256((tmp_path/'source.pt').read_bytes()).hexdigest()
    torch.save(dict(metadata=metadata,neighbors=neighbors,x_variance=np.ones(2),
        shape_variance=np.ones(2),slope_variance=np.ones(2)),tmp_path/'model.pt')
    np.savez(tmp_path/'ids.npz',**after['ids'])
    restored = load_model(tmp_path/'model.pt')
    with np.load(tmp_path/'ids.npz') as stored:
        replay = restored.decode_ids(dict(stored))
    for key in ('S','O','sum'):
        np.testing.assert_array_equal(after[key],replay[key])
    rate = integer_rate(tmp_path/'model.pt',tmp_path/'ids.npz')
    assert rate['inference_package_bytes'] == (tmp_path/'model.pt').stat().st_size+(tmp_path/'source.pt').stat().st_size
    assert rate['model_size']['state_scalars_including_buffers'] > 0
    with (tmp_path/'source.pt').open('ab') as stream:
        stream.write(b'drift')
    with pytest.raises(ValueError,match='identity'):
        load_model(tmp_path/'model.pt')
