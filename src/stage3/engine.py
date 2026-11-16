"""Canonical float64 binary readout engine, independent of worker orchestration."""
import time
import warnings
import numpy as np
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.sparsefuncs import mean_variance_axis
from src.stage2.schema import check


def statistics(x, weights):
    x=x.astype('f8',copy=False); a=np.asarray(weights,dtype='f8')
    check(x.ndim==2 and a.shape==(x.shape[0],) and np.isfinite(a).all() and np.all(a>0), 'Bad statistics weights')
    if sparse.issparse(x):
        mean,var=mean_variance_axis(x,axis=0,weights=a)
    else:
        check(np.isfinite(x).all(),'Nonfinite input')
        mean=np.average(x,axis=0,weights=a)
        var=np.average((x-mean)**2,axis=0,weights=a)
    check(np.isfinite(mean).all() and np.isfinite(var).all() and np.all(var>=-1e-12), 'Invalid variance')
    var=np.maximum(var,0.)
    scale=np.where(var<=1e-12,1.,np.sqrt(var))
    return {'mean':mean,'variance':var,'scale':scale,'center':not sparse.issparse(x)}


def transform(x, stats):
    check(x.shape[1]==len(stats['scale']), 'Feature dimension mismatch')
    if sparse.issparse(x):
        check(not stats['center'], 'Cannot center sparse TF-IDF')
        return x.astype('f8').multiply(1/stats['scale']).tocsr()
    check(stats['center'], 'Dense center policy mismatch')
    return (np.asarray(x,dtype='f8')-stats['mean'])/stats['scale']


def objective_gradient(x,y,a,w,b,C):
    z=np.asarray(x@w).ravel()+b
    loss=np.dot(a,np.logaddexp(0,z)-y*z)+np.dot(w,w)/(2*C)
    residual=a*(expit(z)-y)
    gw=np.asarray(x.T@residual).ravel()+w/C
    return float(loss),gw,float(residual.sum())


def fit_heads(x,y,known,weights,C):
    check(x.dtype==np.float64 and C in (0.01,0.1,1.,10.),'Canonical dtype/C required')
    check(y.shape==known.shape==(x.shape[0],25) and known.dtype==bool,'Target/mask shape mismatch')
    a=np.asarray(weights,dtype='f8')
    check(a.shape==(x.shape[0],) and np.all(a>0),'Invalid loss weights')
    check(np.isfinite(x.data if sparse.issparse(x) else x).all(),'Nonfinite features')
    check(np.all((y[known]==0)|(y[known]==1)),'Nonbinary known targets')
    coef=np.zeros((25,x.shape[1]),dtype='<f8'); intercept=np.zeros(25,dtype='<f8')
    valid=np.zeros(25,dtype=bool); records=[]
    for k in range(25):
        mask=known[:,k]; yy=y[mask,k]; aa=a[mask]
        counts={'known':int(mask.sum()),'positive':int(yy.sum()),'weight':float(aa.sum()),
                'positive_weight':float(aa@yy)}
        if len(np.unique(yy))!=2:
            records.append(dict(counts,status='unestimable',attempts=[])); continue
        attempts=[]
        for limit in (1000,5000):
            started=time.monotonic()
            model=LogisticRegression(C=C,solver='lbfgs',penalty='l2',fit_intercept=True,class_weight=None,
                                     tol=1e-8,max_iter=limit,warm_start=False)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(x[mask],yy,sample_weight=aa)
            check(model.classes_.tolist()==[0,1],'Wrong positive orientation')
            bad=any(issubclass(v.category,ConvergenceWarning) for v in caught) or int(model.n_iter_[0])>=limit
            loss,grad,gb=objective_gradient(x[mask],yy,aa,model.coef_[0],model.intercept_[0],C)
            check(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all() and np.isfinite(loss), 'Nonfinite fit')
            attempts.append({'limit':limit,'iterations':int(model.n_iter_[0]),'converged':not bad,
                             'seconds':time.monotonic()-started,'warnings':[str(v.message) for v in caught],
                             'objective':loss,'gradient_inf':float(max(np.max(abs(grad),initial=0),abs(gb)))})
            if not bad: break
        check(not bad,'Required head failed deterministic convergence retry: '+str(k))
        coef[k],intercept[k],valid[k]=model.coef_[0],model.intercept_[0],True
        records.append(dict(counts,status='fitted',attempts=attempts))
    return coef,intercept,valid,records


def metrics(logits,y,known,valid,contexts,primary):
    """Return proper scores and fixed-cohort contextwise selection score."""
    check(logits.shape==y.shape==known.shape and np.isfinite(logits).all(),'Invalid scoring shape/values')
    contexts=np.asarray(contexts); groups=sorted(set(contexts.tolist()))
    result={}; selection=[]
    for context in groups:
        cells=[]
        for k in range(25):
            ix=(contexts==context)&known[:,k]
            yy=y[ix,k]; z=logits[ix,k]
            item={'known':int(ix.sum()),'positive':int(yy.sum()),'auroc':None,'ap':None,'brier':None,'log_loss':None,'reason':None}
            if not valid[k] or not len(yy): item['reason']='unestimable head/no known cells'
            else:
                item['brier']=float(np.mean((expit(z)-yy)**2))
                item['log_loss']=float(np.mean(np.logaddexp(0,z)-yy*z))
                if len(np.unique(yy))==2:
                    item['auroc']=float(roc_auc_score(yy,z)); item['ap']=float(average_precision_score(yy,z))
                else: item['reason']='one outcome class'
            if k in primary: selection.append(item['auroc'])
            cells.append(item)
        result[context]=cells
    score=None if any(v is None for v in selection) else float(np.mean(selection))
    return {'contexts':result,'selection_score':score}


def choose(scores):
    check(set(float(c) for c in scores)=={0.01,0.1,1.,10.} and all(v is not None and np.isfinite(v) for v in scores.values()),'Complete defined C grid required')
    best=max(scores.values())
    return min(float(c) for c,v in scores.items() if best-v<=1e-12)
