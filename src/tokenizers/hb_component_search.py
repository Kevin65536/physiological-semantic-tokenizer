"""Frozen Hb component decoder and finite-code search with observation-only objectives."""
from pathlib import Path
import numpy as np
import torch
from .hb_native_id_readout import load_model,apply_head


class FrozenHb111:
    """Keep the exact frozen decoder, including fold head, gains and reserved pair."""
    def __init__(self,directory):
        directory=Path(directory);self.native=load_model(directory/'native_readout.pt')
        self.silent=self.native.source;self.base=self.silent.source;self.model=self.base.model
        self.heads=torch.load(directory/'decoded_waveform__ridge512_fold_heads.pt',map_location='cpu',weights_only=False)
        self.sizes=dict(S=128,O=256);self.level=self.silent.level.copy()
        if len(self.heads)!=4 or self.base.metadata['branch_codebook_sizes']!=self.sizes:
            raise ValueError('original frozen Hb S128/O256 and four 111 heads required')
        self.model.eval();self.model.requires_grad_(False)

    @staticmethod
    def quiet(ids):
        return (np.asarray(ids['S'])==128).all(1)&(np.asarray(ids['O'])==256).all(1)

    @staticmethod
    def missing(ids):
        return (np.asarray(ids['S'])==129).all(1)&(np.asarray(ids['O'])==257).all(1)

    def decode(self,ids,fold,gamma=1.):
        if gamma not in (.5,1.,2.):raise ValueError('registered fixed invertible model scale required')
        missing=self.missing(ids);safe={b:np.where(missing[:,None],0,ids[b]) for b in 'SO'}
        raw=self.silent.decode_ids(safe);post=apply_head(safe,raw,self.heads[int(fold)])
        quiet=self.quiet(ids)
        if gamma!=1.:
            for b in post:post[b]=np.where(quiet[:,None,None],post[b],post[b]/gamma)
        for b in post:post[b][missing]=0
        return post

    def encode(self,x,original_ids=None,gamma=1.):
        x=np.asarray(x,np.float32)
        if x.ndim!=3 or x.shape[1:]!=(15,40) or not np.isfinite(x).all():raise ValueError('visible finite30s paired Hb X required')
        if original_ids is None:original_ids=self.silent.encode(x)
        if gamma==1.:return {b:np.asarray(original_ids[b]).copy() for b in 'SO'}
        ids=self.base.encode(x*gamma);quiet=self.quiet(original_ids)
        missing=self.missing(original_ids)
        for b in ids:
            ids[b][quiet]={'S':128,'O':256}[b];ids[b][missing]={'S':129,'O':257}[b]
        return ids

    def continuous(self,x,ids,fold,gamma=1.):
        quiet=self.quiet(ids);missing=self.missing(ids);safe={b:np.where(missing[:,None],0,ids[b]) for b in 'SO'}
        raw=self.base.predict_continuous(np.asarray(x,np.float32)*gamma)
        if quiet.any():
            raw={b:np.asarray(v,float).copy() for b,v in raw.items()}
            raw['S'][quiet]=np.tile(self.level,20)[None,None];raw['O'][quiet]=-np.tile(self.level,20)[None,None];raw['sum'][quiet]=0
        post=apply_head(safe,raw,self.heads[int(fold)])
        if gamma!=1.:
            for b in post:post[b]=np.where(quiet[:,None,None],post[b],post[b]/gamma)
        for b in post:post[b][missing]=0
        return post

    def proposals(self,ids,x,variance,source=None,component_variance=None,oracle=None,gamma=1.,count=4):
        """Approximate derivative ranks only propose codes; exact decoder accepts them."""
        if count!=4:raise ValueError('four predeclared ordinary candidates required')
        inactive=self.quiet(ids)|self.missing(ids);z={};tables={}
        for b in 'SO':
            q=self.model.branches[b]['quantizer'];index=torch.as_tensor(np.where(inactive[:,None],0,ids[b]),dtype=torch.long)
            z[b]=q.get_embedding(index).detach().clone().requires_grad_(True);tables[b]=q.codebook.detach()
        wave={b:self.model._decode_component(b,z[b]) for b in 'SO'}
        total=(wave['S']+wave['O']).double()/gamma
        factor=torch.tensor(self.base.metadata.get('decoder_component_transfer_gain',[1.,1.]),dtype=torch.float64).repeat(20)
        o=wave['O'].double()*factor/gamma;pred=dict(S=total-o,O=o,sum=total)
        cv=component_variance;loss=total.new_zeros(())
        if oracle is None:
            loss+=((total-torch.as_tensor(x))/torch.as_tensor(np.sqrt(variance))).square().mean()
            if source is not None:
                for b in 'SO':loss+=((pred[b]-torch.as_tensor(source[b]))/torch.as_tensor(np.sqrt(cv[b]))).square().mean()
        else:
            for b in 'SO':loss+=((pred[b]-torch.as_tensor(oracle[b]))/torch.as_tensor(np.sqrt(cv[b]))).square().mean()
        gradient=torch.autograd.grad(loss,list(z.values()));result={}
        for b,g in zip('SO',gradient):
            costs=g.reshape(-1,g.shape[-1])@tables[b].T
            rank=torch.argsort(costs,dim=-1,stable=True)[:,:count-1].numpy().reshape(*ids[b].shape,count-1)
            # Rank includes the current exact integer so search cannot force a change.
            result[b]=np.concatenate([np.asarray(ids[b])[...,None],rank],axis=-1)
        return result

    def search(self,ids,x,variance,fold,source=None,component_variance=None,oracle=None,gamma=1.,full=False,progress=None,mask=None,keep_X_ratio=None):
        """One complete interleaved coordinate sweep; returns only the paid integers."""
        selected={b:np.asarray(ids[b]).copy() for b in 'SO'};inactive=self.quiet(ids)|self.missing(ids)
        if inactive.all():return selected,dict(changed=0,attempted=0)
        proposal=None if full else self.proposals(selected,x,variance,source,component_variance,oracle,gamma)
        before=self.decode(selected,fold,gamma);initial=objective(before,x,variance,source,component_variance,oracle,mask)
        valid=np.ones_like(x,bool) if mask is None else np.asarray(mask,bool)
        reference_X=np.stack([weighted_error(before['sum']-x,variance,valid,c) for c in range(2)],axis=1)
        best=initial.copy();attempted=0
        for patch in range(15):
            for branch in 'SO':
                choices=np.broadcast_to(np.arange(self.sizes[branch]),(len(x),self.sizes[branch])) if full else proposal[branch][:,patch]
                choices=choices.copy();choices[inactive]=selected[branch][inactive,patch,None]
                width=choices.shape[1];trial={b:np.repeat(selected[b][:,None],width,axis=1) for b in 'SO'};trial[branch][:,:,patch]=choices
                flat={b:v.reshape(-1,15) for b,v in trial.items()}
                decode=self.decode(flat,fold,gamma)
                repeat=lambda v:np.repeat(v,width,axis=0)
                values=objective(decode,repeat(x),repeat(variance),None if source is None else {b:repeat(source[b]) for b in 'SO'},None if component_variance is None else {b:repeat(component_variance[b]) for b in 'SO'},None if oracle is None else {b:repeat(oracle[b]) for b in 'SO'},None if mask is None else repeat(mask)).reshape(len(x),width)
                if keep_X_ratio is not None:
                    error_X=np.stack([weighted_error(decode['sum']-repeat(x),repeat(variance),repeat(valid),c).reshape(len(x),width) for c in range(2)],axis=1)
                    values=np.where((error_X<=reference_X[:,:,None]*keep_X_ratio+1e-10).all(1),values,np.inf)
                choose=values.argmin(1);candidate=values[np.arange(len(x)),choose];improve=(candidate<best-1e-8)&~inactive
                selected[branch][improve,patch]=choices[np.arange(len(x)),choose][improve];best[improve]=candidate[improve];attempted+=int((~inactive).sum()*width)
                if progress is not None:progress(patch,branch)
        # Recompute on the canonical batch, retaining every failure rather than trusting cached trial floats.
        final=objective(self.decode(selected,fold,gamma),x,variance,source,component_variance,oracle,mask)
        good=(final<=initial+1e-8)|inactive
        if keep_X_ratio is not None:
            decoded=self.decode(selected,fold,gamma);error_X=np.stack([weighted_error(decoded['sum']-x,variance,valid,c) for c in range(2)],axis=1)
            good&=(error_X<=reference_X*keep_X_ratio+1e-10).all(1)|inactive
        for b in 'SO':selected[b][~good]=ids[b][~good]
        return selected,dict(changed=int(sum((selected[b]!=ids[b]).sum() for b in 'SO')),attempted=attempted,reverted=int((~good).sum()),initial_objective=initial,final_objective=objective(self.decode(selected,fold,gamma),x,variance,source,component_variance,oracle,mask))

    def search_pairs(self,ids,x,variance,fold,source=None,component_variance=None,oracle=None,gamma=1.,full=False,progress=None,mask=None,keep_X_ratio=1.05):
        """Coupled word moves preserve the visible sum and can cross cancellation barriers.

        The fixed decoder is branch-separable before its affine111 head. The
        Cartesian quadratic ranks pair changes efficiently. Four ranked actual
        pairs are decoded exactly before acceptance; no continuous anchor is sent.
        """
        selected={b:np.asarray(ids[b]).copy() for b in 'SO'};inactive=self.quiet(ids)|self.missing(ids);n=len(x)
        baseline=self.decode(selected,fold,gamma);valid=np.ones_like(x,bool) if mask is None else np.asarray(mask,bool)
        initial=objective(baseline,x,variance,source,component_variance,oracle,valid);best=initial.copy()
        reference_X=np.stack([weighted_error(baseline['sum']-x,variance,valid,c) for c in range(2)],axis=1)
        if inactive.all():return selected,dict(changed=0,pair_moves=0,initial_objective=initial,final_objective=initial)
        proposal=None if full else self.proposals(selected,x,variance,source,component_variance,oracle,gamma)
        moves=0;gap=0.;attempted=0
        for patch in range(15):
            current=self.decode(selected,fold,gamma);changes={};choices={}
            for branch in 'SO':
                options=np.broadcast_to(np.arange(self.sizes[branch]),(n,self.sizes[branch])).copy() if full else proposal[branch][:,patch].copy()
                options[inactive]=selected[branch][inactive,patch,None];choices[branch]=options;width=options.shape[1]
                trial={b:np.repeat(selected[b][:,None],width,axis=1) for b in 'SO'};trial[branch][:,:,patch]=options
                pred=self.decode({b:v.reshape(-1,15) for b,v in trial.items()},fold,gamma)
                changes[branch]={b:pred[b].reshape(n,width,15,40)-current[b][:,None] for b in pred}
            table=np.zeros((n,choices['S'].shape[1],choices['O'].shape[1]))
            fields=[('S',oracle['S'],component_variance['S']),('O',oracle['O'],component_variance['O'])] if oracle is not None else [('sum',x,variance)]
            if oracle is None and source is not None:fields +=[(b,source[b],component_variance[b]) for b in 'SO']
            for b,target,v in fields:table+=pair_quadratic(current[b]-target,changes['S'][b],changes['O'][b],v,valid)
            guards=np.stack([pair_quadratic(current['sum']-x,changes['S']['sum'],changes['O']['sum'],variance,valid,color=c) for c in range(2)],axis=1)
            allowed=(guards<=reference_X[:,:,None,None]*keep_X_ratio+1e-10).all(1)
            rank=np.argsort(np.where(allowed,table,np.inf).reshape(n,-1),axis=1,kind='stable')[:,:4]
            width=rank.shape[1];a,b=rank//choices['O'].shape[1],rank%choices['O'].shape[1]
            candidate={key:np.repeat(selected[key][:,None],width,axis=1) for key in 'SO'}
            candidate['S'][:,:,patch]=np.take_along_axis(choices['S'],a,1);candidate['O'][:,:,patch]=np.take_along_axis(choices['O'],b,1)
            for key in 'SO':candidate[key][inactive]=selected[key][inactive,None]
            exact=self.decode({key:v.reshape(-1,15) for key,v in candidate.items()},fold,gamma);repeat=lambda v:np.repeat(v,width,axis=0)
            cost=objective(exact,repeat(x),repeat(variance),None if source is None else {b:repeat(source[b]) for b in 'SO'},None if component_variance is None else {b:repeat(component_variance[b]) for b in 'SO'},None if oracle is None else {b:repeat(oracle[b]) for b in 'SO'},repeat(valid)).reshape(n,width)
            exact_X=np.stack([weighted_error(exact['sum']-repeat(x),repeat(variance),repeat(valid),c).reshape(n,width) for c in range(2)],axis=1)
            eligible=(exact_X<=reference_X[:,:,None]*keep_X_ratio+1e-10).all(1)
            cost=np.where(eligible,cost,np.inf);chosen=cost.argmin(1);value=cost[np.arange(n),chosen];improve=(value<best-1e-8)&~inactive
            for key in 'SO':selected[key][improve]=candidate[key][np.arange(n),chosen][improve]
            best[improve]=value[improve];moves+=int(improve.sum());attempted+=int((~inactive).sum()*width)
            for field in ['S','O','sum']:
                approx=current[field][:,None]+changes['S'][field][np.arange(n)[:,None],a]+changes['O'][field][np.arange(n)[:,None],b]
                gap=max(gap,float(abs(approx-exact[field].reshape(n,width,15,40)).max()))
            if progress is not None:progress(patch,'paired')
        final=self.decode(selected,fold,gamma);end=objective(final,x,variance,source,component_variance,oracle,valid)
        endX=np.stack([weighted_error(final['sum']-x,variance,valid,c) for c in range(2)],axis=1)
        good=((end<=initial+1e-8)&(endX<=reference_X*keep_X_ratio+1e-10).all(1))|inactive
        for key in 'SO':selected[key][~good]=ids[key][~good]
        return selected,dict(changed=int(sum((selected[b]!=ids[b]).sum() for b in 'SO')),pair_moves=moves,attempted_exact_pairs=attempted,reverted=int((~good).sum()),maximum_affine_proposal_gap=gap,initial_objective=initial,final_objective=objective(self.decode(selected,fold,gamma),x,variance,source,component_variance,oracle,valid))


def objective(prediction,x,variance,source=None,component_variance=None,oracle=None,mask=None):
    """One scalar per full 30s sequence. Oracle never enters the X-only caller."""
    y=np.asarray(x,float);valid=np.ones_like(y,bool) if mask is None else np.asarray(mask,bool)
    if y.ndim!=3 or y.shape[1:]!=(15,40) or valid.shape!=y.shape:raise ValueError('full matched30s objective support required')
    def mse(a,v):
        d=np.where(valid,a,0.)**2/np.asarray(v,float)
        return d.sum((1,2))/valid.sum((1,2)).clip(1)
    if oracle is not None:return sum(mse(np.asarray(prediction[b])-oracle[b],component_variance[b]) for b in 'SO')
    value=mse(np.asarray(prediction['sum'])-y,variance)
    if source is not None:value+=sum(mse(np.asarray(prediction[b])-source[b],component_variance[b]) for b in 'SO')
    return value


def weighted_error(difference,variance,mask,color=None):
    valid=np.asarray(mask,bool).copy()
    if color is not None:valid[...,1-color::2]=False
    return (np.where(valid,difference,0.)**2/np.asarray(variance)).sum((1,2))/valid.sum((1,2)).clip(1)


def pair_quadratic(error,left,right,variance,mask,color=None):
    valid=np.asarray(mask,bool).copy()
    if color is not None:valid[...,1-color::2]=False
    weight=(valid/np.broadcast_to(variance,error.shape)/valid.sum((1,2)).clip(1)[:,None,None]).reshape(len(error),-1)
    e=error.reshape(len(error),-1);a=left.reshape(len(error),left.shape[1],-1);b=right.reshape(len(error),right.shape[1],-1)
    constant=np.einsum('nt,nt,nt->n',e,e,weight)
    aa=np.einsum('nkt,nkt,nt->nk',a,a,weight)+2*np.einsum('nt,nkt,nt->nk',e,a,weight)
    bb=np.einsum('nkt,nkt,nt->nk',b,b,weight)+2*np.einsum('nt,nkt,nt->nk',e,b,weight)
    cross=2*np.einsum('nkt,nlt,nt->nkl',a,b,weight,optimize=True)
    return constant[:,None,None]+aa[:,:,None]+bb[:,None]+cross
