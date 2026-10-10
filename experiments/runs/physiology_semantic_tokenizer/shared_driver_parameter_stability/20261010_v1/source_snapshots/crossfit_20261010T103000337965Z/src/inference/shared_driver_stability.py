"""Array-only parameter stability diagnostics for the fixed six-state SSM.

Engineering profiles, working residual covariance and repeat statistics are
separate from calibrated physiological uncertainty. Callers own identities,
training-only selection, array access and the fixed failure denominator.
"""
from dataclasses import replace
from functools import lru_cache
import time
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import least_squares

from .shared_driver_reconstruction import (
    fit_nonlinear_shared_parameter, fit_nonlinear_shared_parameters,
    nonlinear_driver_forward, replay_nonlinear_driver,
)
from .observation_baselines import native_model_operator
from .t3a_balloon_robust_ssm import BalloonParameters, BalloonFixedParameters, BalloonFreeParameters

PARAMETERS=('tau','neurovascular_gain','kappa')
REFERENCE=np.array([2.,1.,.64])
BOUNDS=np.array([[.5,8.],[.1,10.],[.2,1.5]])


def parameter_object(base_config, values=None):
    fixed={k:v for k,v in base_config['fixed'].items() if k not in ('tau','kappa')}
    free={k:base_config['fixed'][k] for k in ('tau','kappa')}
    for k,v in (values or {}).items():
        (fixed if k=='neurovascular_gain' else free)[k]=float(v)
    return BalloonParameters(BalloonFixedParameters(**fixed), BalloonFreeParameters(**free))


def boundary_distance(values, bounds):
    z=np.log(np.asarray(values,float));b=np.log(np.asarray(bounds,float))
    f=(z-b[...,0])/(b[...,1]-b[...,0])
    return np.minimum(f,1-f)


def symmetric_relative(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return 2*np.abs(a-b)/(np.abs(a)+np.abs(b))


def icc_absolute(values):
    """ICC(A,1) for n independent subjects and exactly two independent estimates."""
    x=np.asarray(values,float)
    if x.ndim!=2 or x.shape[1]!=2 or len(x)<3 or not np.isfinite(x).all():return np.nan
    n,k=x.shape;mean=x.mean();rows=x.mean(1);cols=x.mean(0)
    msr=k*np.sum((rows-mean)**2)/(n-1)
    msc=n*np.sum((cols-mean)**2)/(k-1)
    mse=np.sum((x-rows[:,None]-cols[None,:]+mean)**2)/((n-1)*(k-1))
    den=msr+(k-1)*mse+k*(msc-mse)/n
    return float((msr-mse)/den) if den>1e-20 else np.nan


def bootstrap_icc(values,seed,repeats=2000):
    x=np.asarray(values,float);rng=np.random.default_rng(seed)
    if len(x)<3:return [np.nan,np.nan]
    sims=np.array([icc_absolute(x[rng.integers(0,len(x),len(x))]) for _ in range(repeats)])
    sims=sims[np.isfinite(sims)]
    return np.quantile(sims,[.025,.975]).tolist() if len(sims) else [np.nan,np.nan]


def estimate_working_covariance(residual_chunks, *, shrinkage=.2, floor=.05, ar_clip=.9):
    """Fit stationary working error covariance to cross-fitted standardized errors."""
    chunks=[np.asarray(x,float).reshape(-1,3) for x in residual_chunks]
    x=np.concatenate(chunks);x=x-x.mean(axis=0)
    covariance=x.T@x/len(x)
    covariance=(1-shrinkage)*covariance+shrinkage*np.diag(np.diag(covariance))
    eig,u=np.linalg.eigh(covariance)
    eig=np.maximum(eig,floor*max(float(np.mean(eig)),1e-12))
    covariance=(u*eig)@u.T
    covariance/=np.trace(covariance)/3
    inverse=np.linalg.inv(covariance)
    numerator=sum(np.einsum('ti,ij,tj->',v[1:]-v.mean(0),inverse,v[:-1]-v.mean(0)) for v in chunks)
    denominator=sum(np.einsum('ti,ij,tj->',v[:-1]-v.mean(0),inverse,v[:-1]-v.mean(0)) for v in chunks)
    rho=float(np.clip(numerator/max(denominator,1e-12),-ar_clip,ar_clip))
    return dict(covariance=covariance,ar1=rho,n_samples=len(x),n_chunks=len(chunks),
        interpretation='cross_fitted_working_residual_covariance_not_instrument_noise',
        precision_normalization='trace_per_visible_observation_equals_one')


@lru_cache(maxsize=32)
def _working_whitener(steps, covariance_tuple, rho):
    covariance=np.asarray(covariance_tuple).reshape(3,3)
    component=solve_triangular(np.linalg.cholesky(covariance),np.eye(3),lower=True)
    temporal=np.eye(steps)/np.sqrt(1-rho*rho);temporal[0,0]=1.
    temporal[np.arange(1,steps),np.arange(steps-1)]=-rho/np.sqrt(1-rho*rho)
    w=np.kron(temporal,component)
    w*=np.sqrt(3*steps/np.sum(w*w))
    return w


def working_whitener(steps, calibration, mask=None):
    w=_working_whitener(steps,tuple(np.asarray(calibration['covariance']).ravel()),float(calibration['ar1']))
    if mask is None or np.asarray(mask).all():return w
    # Marginalize hidden coordinates before whitening. Selecting rows of the
    # full precision would incorrectly make hidden observations available.
    keep=np.asarray(mask,bool).ravel()
    full_cov=np.linalg.inv(w.T@w)
    cov=full_cov[np.ix_(keep,keep)]
    return solve_triangular(np.linalg.cholesky(cov),np.eye(keep.sum()),lower=True)


def record_arrays(target, arm='C0a', block_steps=120):
    y=np.asarray(target,float).reshape(-1,3)
    if len(y)%block_steps:raise ValueError('whole original feature blocks required')
    if arm=='C1':
        return y[None],native_model_operator(len(y),block_steps=block_steps),list(range(block_steps,len(y),block_steps))
    return y.reshape(-1,block_steps,3),native_model_operator(block_steps),[]


def fit_record(target,sd,cfg,base,model,*,arm='C0a',fixed=None,prior=None,
               covariance=None,visible=None,starts=None,profile=False,n0=False,audit=True,
               objective='current_without_parameter_shrinkage',max_iterations=None):
    """Independent per-record fit. No other repeat's result is a default start."""
    began=time.monotonic();names=list(cfg['models'][model]);p=parameter_object(base,fixed)
    y,operator,breaks=record_arrays(target,arm,cfg['tensor']['block_steps'])
    b,n,_=y.shape;s=cfg['solver']
    mask=None if visible is None else np.asarray(visible,bool).reshape(y.shape)
    bounds=[cfg['parameters'][name]['bounds'] for name in names]
    initial_penalty=np.full(b,s['initial_penalty'])
    if arm=='C0b':initial_penalty[1:]=0.
    if objective=='observation_only_legal_domain':initial_penalty[:]=0.
    options=dict(parameter_names=names,parameter_bounds=bounds,sd=np.asarray(sd),mean_operator=operator,
        visible=mask,penalty=s['penalty'] if objective!='observation_only_legal_domain' else 0.,
        initial_penalty=initial_penalty,flow_prior_weight=s['flow_prior_weight'] if objective!='observation_only_legal_domain' else 0.,
        flow_prior_log_sd=s['flow_prior_log_sd'],substeps=s['substeps'],
        max_iterations=s['profile_iterations'] if profile else s['max_iterations'],
        max_evaluations_per_trial=s['profile_evaluations_per_trial'] if profile else s['max_evaluations_per_trial'],
        gradient_tolerance=s['gradient_tolerance'],numerical_backend=s['numerical_backend'],curvature_breaks=breaks)
    if max_iterations is not None:options['max_iterations']=max_iterations
    if covariance is not None:
        if arm=='C1':
            from scipy.linalg import block_diag
            if mask is not None and not mask.all():raise ValueError('masked continuous covariance is not a registered arm')
            options['observation_whiteners']=[block_diag(*([working_whitener(120,covariance)]*(n//120)))]
        else:options['observation_whiteners']=[working_whitener(n,covariance,None if mask is None else mask[i]) for i in range(b)]
    if prior is not None and names:
        options.update(parameter_prior_log_mean=[prior['mean'][name] for name in names],
            parameter_prior_precision=[prior['precision'][name] for name in names])
    if starts is None:
        starts=[dict(parameter_values=[cfg['parameters'][name]['starts'][i] for name in names])
                for i in range(3 if names else 1)]
    options['starts']=starts
    if n0 and len(names)==1 and arm=='C0a' and prior is None and covariance is None and objective!='observation_only_legal_domain':
        name=names[0]
        old_starts=[dict(parameter_value=v['parameter_values'][0],driver=v.get('driver',np.zeros((b,n))),
            initial_state=v.get('initial_state',np.tile(np.r_[0.,np.ones(4)],(b,1)))) for v in starts]
        fit=fit_nonlinear_shared_parameter(y,p,.25,parameter_name=name,parameter_bounds=bounds[0],sd=sd,
            mean_operator=operator,visible=mask,penalty=s['penalty'],initial_penalty=s['initial_penalty'],
            flow_prior_weight=s['flow_prior_weight'],flow_prior_log_sd=s['flow_prior_log_sd'],
            starts=old_starts,max_iterations=options['max_iterations'],
            max_evaluations=options['max_evaluations_per_trial']*b,substeps=s['substeps'],
            gradient_tolerance=s['gradient_tolerance'],numerical_backend=s['numerical_backend'],
            step_control='quadratic_interpolation')
        if 'prediction' in fit:
            value=fit['parameter_value'];fit['parameter_values']=np.array([value]);fit['parameter_names']=names
            fit[name]=value
            if audit:
                inspected=fit_nonlinear_shared_parameters(y,p,.25,**dict(options,max_iterations=0,
                    starts=[dict(parameter_values=[value],driver=fit['driver'],initial_state=fit['initial_state'])]))
                for key in ('information','objective_components','raw_parameter_gradient','raw_parameter_gradient_components',
                    'raw_parameter_kkt_inf_norm','projected_raw_parameter_gradient','log_parameter_boundary_distance'):
                    fit[key]=inspected[key]
                fit['audit_objective_difference']=float(inspected['objective']-fit['objective'])
    else:fit=fit_nonlinear_shared_parameters(y,p,.25,**options)
    if 'prediction' in fit:
        pred=np.asarray(fit['prediction']).reshape(-1,3)
        fit['nrmse']=np.sqrt(np.mean(((pred-np.asarray(target).reshape(-1,3))/sd)**2,axis=0))
        fit['correlation']=[float(np.corrcoef(pred[:,j],np.asarray(target).reshape(-1,3)[:,j])[0,1])
                            if np.std(pred[:,j])>1e-15 else np.nan for j in range(3)]
        fit['parameters']={name:float(fit.get(name,getattr(p.fixed if name=='neurovascular_gain' else p.free,name))) for name in PARAMETERS}
        fit['accepted']=bool(fit['converged'] and fit.get('raw_parameter_kkt_inf_norm',np.inf)<=s['raw_parameter_kkt_tolerance'])
    else:fit['accepted']=False
    fit.update(model=model,arm=arm,seconds=time.monotonic()-began,objective_kind=objective)
    return fit


def compact_fit(fit):
    omitted={'prediction','canonical_prediction','driver','initial_state','states',
             'residual_jacobians','residual_parameter_jacobians','residuals'}
    return {k:v for k,v in fit.items() if k not in omitted}


def warm_start(fit, names):
    if 'driver' not in fit:return None
    return dict(parameter_values=[fit['parameters'][name] for name in names],
                driver=fit['driver'],initial_state=fit['initial_state'])


def profile_pair(targets,sd,cfg,base,names,*,objective='current_without_parameter_shrinkage'):
    """Two-way nuisance reoptimization on shared fixed scalar/2D grids.

    Best observed values, including unconverged nodes, are retained. Feasibility
    claims additionally require both nuisance solves to pass stationarity.
    """
    from itertools import product
    bounds=np.array([cfg['parameters'][name]['bounds'] for name in names])
    pconf=cfg['profile'];nodes=pconf['scalar_initial_nodes'] if len(names)==1 else pconf['joint_axis_nodes']
    grids=[np.geomspace(*bb,nodes) for bb in bounds]
    visited={};paths=[]
    def run_points(points, direction, seeds=None):
        previous=[None,None] if seeds is None else list(seeds)
        for values in points:
            key=tuple(float(v) for v in values)
            for side in range(2):
                prior_start=warm_start(previous[side],[]) if previous[side] is not None else None
                result=fit_record(targets[side],sd,cfg,base,'P0',fixed=dict(zip(names,key)),
                    starts=[prior_start] if prior_start is not None else None,profile=True,objective=objective)
                previous[side]=result
                paths.append(dict(values=key,side=side,direction=direction,objective=result.get('objective'),
                    accepted=result['accepted'],converged=result.get('converged',False),seconds=result['seconds']))
                entry=visited.setdefault(key,[None,None]);old=entry[side]
                if old is None or result.get('objective',np.inf)<old.get('objective',np.inf):entry[side]=result
        return previous
    points=list(product(*grids))
    run_points(points,'ascending');run_points(points[::-1],'descending')
    if len(names)==1:
        # Refinement targets each repeat minimum, the shared minimum and both
        # original boundary neighborhoods, rather than only the best valley.
        while len(visited)<pconf['scalar_max_nodes']:
            order=sorted(visited);cost=np.array([[v.get('objective',np.inf) for v in visited[k]] for k in order])
            centers=set([0,len(order)-1,int(np.argmin(cost.sum(1)))]+[int(np.argmin(cost[:,i])) for i in range(2)])
            new=[]
            for i in sorted(centers):
                for j in (i-1,i+1):
                    if 0<=j<len(order):
                        value=(float(np.sqrt(order[i][0]*order[j][0])),)
                        if value not in visited and value not in new:new.append(value)
            if not new:break
            new=sorted(new)[:pconf['scalar_max_nodes']-len(visited)]
            for direction,sequence in [('refine_up',new),('refine_down',new[::-1])]:
                for key in sequence:
                    near=min(visited,key=lambda k:abs(np.log(k[0]/key[0])))
                    run_points([key],direction,visited[near])
    elif len(names)==2:
        # Add a 3x3 local mesh around the best shared original cell.
        best=min(visited,key=lambda k:sum(v.get('objective',np.inf) for v in visited[k]))
        local=[]
        for j,axis in enumerate(grids):
            i=int(np.argmin(abs(np.log(axis/best[j]))));lo=axis[max(0,i-1)];hi=axis[min(len(axis)-1,i+1)]
            local.append(np.geomspace(lo,hi,5)[1::2])
        extra=[k for k in product(*local) if k not in visited]
        run_points(extra,'joint_local_up',visited[best]);run_points(extra[::-1],'joint_local_down',visited[best])
    rows=[]
    for values,results in sorted(visited.items()):
        rows.append(dict(values=values,fits=[compact_fit(v) for v in results]))
    return dict(names=names,bounds=bounds,objective=objective,rows=rows,paths=paths,
                finite_search=True,global_optimality_proven=False)


def profile_intersections(profile,reference_objectives,*,nrmse_margin=.01,tolerances=(.005,.01,.05),inners=(.01,.05,.1)):
    values=np.array([r['values'] for r in profile['rows']]);bounds=np.asarray(profile['bounds'])
    costs=np.array([[f.get('objective',np.inf) for f in r['fits']] for r in profile['rows']])
    rmse=np.array([[f.get('nrmse',[np.inf]*3) for f in r['fits']] for r in profile['rows']])
    valid=np.array([[f.get('accepted',False) for f in r['fits']] for r in profile['rows']])
    best_indices=np.argmin(costs,axis=0);best=costs[best_indices,np.arange(2)]
    reference_rmse=rmse[best_indices,np.arange(2)]
    shared=int(np.argmin(costs.sum(1)));distance=boundary_distance(values,bounds)
    results=[]
    for inner in inners:
        interior=(distance>=inner-1e-12).all(1)
        mininner=float(np.min(costs[interior].sum(1))) if interior.any() else np.inf
        for tol in tolerances:
            acceptable=interior&valid.all(1)&((costs-best)<=tol*np.asarray(reference_objectives)[None,:]+1e-10).all(1)
            acceptable&=(rmse-reference_rmse[None,:,:]<=nrmse_margin+1e-12).all(axis=(1,2))
            width=np.ptp(np.log(values[acceptable]),axis=0)/np.log(bounds[:,1]/bounds[:,0]) if acceptable.any() else np.full(len(bounds),np.nan)
            results.append(dict(inner_fraction=inner,tolerance=tol,exists=bool(acceptable.any()),
                accepted_nodes=int(acceptable.sum()),log_range_width=width,
                witness=values[np.flatnonzero(acceptable)[np.argmin(costs[acceptable].sum(1))]] if acceptable.any() else None,
                delta_share=float(costs[shared].sum()-best.sum()),delta_inner=mininner-float(costs[shared].sum()),
                reference_objectives=reference_objectives,denominator='frozen_P0_objective_per_record',
                conclusion='finite_search_existence_witness' if acceptable.any() else 'not_found_not_proof_of_nonexistence'))
    return results


def synthetic_pair(cfg,base,subject,level,model='P2'):
    """Truth enters the forward generator, never free-fit starts or priors."""
    seed=np.random.SeedSequence([cfg['seed'],201,level,subject]);rng=np.random.default_rng(seed)
    spread=cfg['synthetic']['heterogeneity_log_sd'][level]
    logtruth=np.log(REFERENCE)+rng.normal(size=3)*spread
    logbounds=np.log(BOUNDS)
    truth=np.exp(np.clip(logtruth,logbounds[:,0]+.10*np.diff(logbounds,axis=1)[:,0],
                        logbounds[:,1]-.10*np.diff(logbounds,axis=1)[:,0]))
    if subject<cfg['synthetic']['edge_subjects_per_level'] and level>0:
        fraction=.025 if subject%2==0 else .975
        j=(subject//2)%3;truth[j]=np.exp(logbounds[j,0]+fraction*(logbounds[j,1]-logbounds[j,0]))
    for j,name in enumerate(PARAMETERS):
        if name not in cfg['models'][model]:truth[j]=REFERENCE[j]
    p=parameter_object(base,dict(zip(PARAMETERS,truth)))
    steps=cfg['tensor']['block_steps']*cfg['tensor']['blocks'];clock=np.arange(steps)*.25
    operator=native_model_operator(steps,block_steps=120)
    targets=[];drivers=[];initials=[];clean=[]
    for repeat in range(2):
        rg=np.random.default_rng(np.random.SeedSequence([cfg['seed'],202,level,subject,repeat]))
        phases=rg.uniform(-np.pi,np.pi,4)
        driver=sum(a*np.sin(clock*f+ph) for a,f,ph in zip([1.,.7,.4,.2],[.08,.25,.6,1.4],phases))
        driver*=cfg['synthetic']['driver_sd']/np.std(driver)
        initial=np.r_[rg.normal(0,.005),np.exp(rg.normal(0,.015,4))]
        forward=nonlinear_driver_forward(driver,initial,p,.25,substeps=8,numerical_backend='numba',derivative=False)
        y=(operator@forward['canonical_prediction'].ravel()).reshape(-1,3)
        # Training gauge/noise scale is fixed before any virtual subject truth.
        sd=np.full(3,.025);noise=rg.normal(size=y.shape)*sd*cfg['synthetic']['noise_fraction']
        targets.append(y+noise);drivers.append(driver);initials.append(initial);clean.append(y)
    return dict(truth=truth,target=np.array(targets),driver=np.array(drivers),initial=np.array(initials),clean=np.array(clean),sd=sd)


def oracle_fit(target,driver,initial,sd,cfg,base,names):
    indices=[PARAMETERS.index(name) for name in names];bounds=BOUNDS[indices]
    operator=native_model_operator(len(driver),block_steps=120)
    def residual(logp):
        p=parameter_object(base,dict(zip(names,np.exp(logp))))
        f=nonlinear_driver_forward(driver,initial,p,.25,substeps=4,numerical_backend='numba',derivative=False)
        return (((operator@f['canonical_prediction'].ravel()).reshape(-1,3)-target)/sd).ravel()
    result=least_squares(residual,np.log(REFERENCE[indices]),bounds=np.log(bounds).T,
                         xtol=1e-10,gtol=1e-8,ftol=1e-10,max_nfev=100)
    return dict(values=np.exp(result.x),converged=bool(result.success),objective=float(result.fun@result.fun))
