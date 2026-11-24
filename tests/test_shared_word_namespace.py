import numpy as np
from src.tokenizers.shared_word_namespace import WIDTH,compact,expand_native
from src.tokenizers.cross_branch_sampling import layouts
from src.tokenizers.cross_branch_innovations import pilot_positions


def test_contextual_bijection_recovers_full_words_after_unobserved_site():
    rng=np.random.default_rng(12);n=7;seen=np.array([1,1,0,1,1,1,1],bool)
    roles=layouts(seen)['staggered'];absent=~seen
    maps={b:np.stack([rng.permutation(WIDTH)+1 for _ in range(10)]) for b in 'SO'}
    sent={}
    for bi,b in enumerate('SO'):
        use=pilot_positions(n)[bi*n:(bi+1)*n];start=np.where(use,1+roles[b].astype(int)*1023,5116+roles[b].astype(int)*0+roles[b].astype(int)*2047)
        ids=start[:,None]+rng.integers(0,1000,(n,15));ids[absent]=0;sent[b]=ids.astype(np.uint16)
    recovered=expand_native({b:compact(sent[b],maps[b]) for b in 'SO'},maps)
    for b in 'SO':np.testing.assert_array_equal(recovered[b],sent[b])
