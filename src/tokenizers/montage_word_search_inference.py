"""Portable X-only encoder selection over unchanged paid montage words."""
import numpy as np
from .montage_word_search import dictionaries,synthesis_matrix,select_words
from .cross_branch_innovations import InnovationEEG, load_model as load_innovation, paid_fields


class MontageWordSearchEEG(InnovationEEG):
    def __init__(self,base,variances,variant):
        super().__init__(base.metadata,base.sources,base.pilot_books,base.residual_books,base.states)
        if self.metadata.get('encoder_extension')!='montage_word_search_v1' or variant not in self.metadata['search_variants']:
            raise ValueError('registered frozen montage selection required')
        self.selection=self.metadata['search_variants'][variant];self.variances=variances
        self.word_tables=dictionaries(self.pilot_books,self.residual_books)
        for book in self.word_tables.values():
            if book['count'].shape!=(len(book['atoms']),) or not np.any(book['count']>0):raise ValueError('real training atom occupancy required')
        for ds,names in self.metadata['positions'].items():
            if np.shape(self.variances[ds])!=(3,len(names)) or not np.all(np.isfinite(self.variances[ds])&(self.variances[ds]>0)):
                raise ValueError('positive original native training variances required')
        self.metadata=dict(self.metadata,portable_encoder_variant=variant)

    def _encode_array(self,x,dataset,seen):
        from .band_exchange_words import source_properties
        sent=super()._encode_array(x,dataset,seen)
        source=self.sources[dataset].predict_continuous(x[None],dataset,seen[None])
        target=np.concatenate([source[b][0] for b in 'SO']+[x])
        properties={b:source_properties(source[b][0]) for b in 'SO'}
        state=self.states[dataset]
        _,_,roles=paid_fields(sent,self.pilot_books,self.residual_books,state['scale'],'innovation',False)
        matrix=synthesis_matrix(roles,state)
        weight=(1/self.variances[dataset]*seen[None]).reshape(-1)
        result,_=select_words(sent,matrix,target,weight,properties,self.word_tables,roles,self.selection['limit'],self.selection['sweeps'],self.selection['spectral_strength'])
        return result


def load_model(path,variant=None):
    base=load_innovation(path)
    with np.load(path,allow_pickle=False) as z:
        variances={ds:z[ds+'_train_target_variance'].copy() for ds in base.metadata['positions']}
    return MontageWordSearchEEG(base,variances,variant)
