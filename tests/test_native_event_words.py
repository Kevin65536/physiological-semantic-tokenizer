import numpy as np
from src.tokenizers.native_event_words import FIRST,K,PULSE_BASE,pulse_shapes,event_wave,closest_event


def test_exhaustive_event_choice_matches_brute_force():
    rng=np.random.default_rng(53);p=pulse_shapes();norm=(p*p).sum(1);codes=np.arange(FIRST,K)
    probe=np.array([FIRST+7*128+5*16+8,PULSE_BASE+321*32+16+7,K-1])
    target=event_wave(probe,p)+rng.normal(scale=.001,size=(len(probe),400))
    selected,gain=closest_event(target,p,norm);np.testing.assert_array_equal(selected,probe)
    score=np.full(len(target),-np.inf)
    for start in range(0,len(codes),1100):
        atoms=event_wave(codes[start:start+1100],p)
        score=np.maximum(score,(2*target@atoms.T-(atoms**2).sum(1)).max(1))
    np.testing.assert_allclose(gain,score,rtol=1e-12)
