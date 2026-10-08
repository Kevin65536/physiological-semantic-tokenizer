"""Array access for the versioned public SSM prepared-feature contract.

Callers validate the exact parent root and permitted identities before calling
this boundary. This is not a raw-dataset parser and never rebuilds missing data.
"""
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=16)
def read_prepared_arrays(path):
    with np.load(path, allow_pickle=False) as arrays:
        eeg, hb = arrays['eeg_features'].copy(), arrays['hb'].copy()
    if (eeg.ndim != 3 or eeg.shape[1:] != (120, 30)
            or hb.shape != (len(eeg), 120, 2)):
        raise ValueError('expected parent EEG [N,120,30] and Hb [N,120,2]')
    eeg.flags.writeable = False
    hb.flags.writeable = False
    return eeg, hb


def prepared_target(ref, coordinate):
    eeg, hb = read_prepared_arrays(ref['array_path'])
    index = ref['array_index']
    return np.column_stack((eeg[index]@np.asarray(coordinate['pc'])*coordinate['eeg_factor'],
                            hb[index]*coordinate['hb_factor']))
