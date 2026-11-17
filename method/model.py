"""Compact NumPy models for predicting evidence changes."""

import json
from pathlib import Path

import numpy as np


KINDS={'joint':9,'marginal':6,'scalar':1,'direct':3,'independent':1}


def event_index(gain,loss,unsupported=0):
    if min(gain,loss,unsupported)<0: raise ValueError('Transition counts must be nonnegative')
    return min(int(gain),2)*3+min(int(loss+unsupported),2)


def distribution(events):
    if not events: raise ValueError('A transition group needs observations')
    p=np.zeros(9)
    for e in events: p[event_index(e['gain'],e['loss'],e['unsupported'])]+=1
    return p/p.sum()


def softmax(x):
    e=np.exp(x-x.max(axis=1,keepdims=True)); return e/e.sum(axis=1,keepdims=True)


def policy_statistics(prob):
    prob=np.asarray(prob,dtype=float)
    if prob.shape[-1]!=9 or np.any(prob<0) or not np.allclose(prob.sum(axis=-1),1):
        raise ValueError('Expected a normalized nine-cell joint distribution')
    p=prob.reshape(-1,3,3); g=np.arange(3)
    return dict(clean_gain=(p[:,:,0]*g).sum(axis=1),harm_probability=1-p[:,:,0].sum(axis=1),
        mean_gain=(p*g[None,:,None]).sum(axis=(1,2)),mean_harm=(p*g[None,None,:]).sum(axis=(1,2)))


class Student:
    def __init__(self,kind,seed=17,hidden=96):
        if kind not in KINDS: raise ValueError(kind)
        self.kind,self.seed,self.hidden=kind,seed,hidden

    def fit(self,x,y,validation,epochs=100):
        x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
        self.mean=x.mean(axis=0); self.scale=np.maximum(x.std(axis=0),1e-4)
        x=np.clip((x-self.mean)/self.scale,-10,10)
        vx,vy=validation; vx=np.clip((np.asarray(vx)-self.mean)/self.scale,-10,10); vy=np.asarray(vy)
        rng=np.random.default_rng(self.seed); n=x.shape[1]
        self.w=[rng.normal(0,np.sqrt(2/n),(n,self.hidden)),np.zeros(self.hidden),
                rng.normal(0,np.sqrt(1/self.hidden),(self.hidden,KINDS[self.kind])),np.zeros(KINDS[self.kind])]
        moments=[np.zeros_like(w) for w in self.w]; squares=[np.zeros_like(w) for w in self.w]
        best=float('inf'); saved=None; step=0; self.history=[]
        for epoch in range(epochs):
            order=rng.permutation(len(x))
            for start in range(0,len(x),32):
                ix=order[start:start+32]; a=x[ix]; target=y[ix]
                h=np.maximum(0,a@self.w[0]+self.w[1]); logits=h@self.w[2]+self.w[3]
                loss,delta=self._loss(logits,target)
                dh=(delta@self.w[2].T)*(h>0)
                grads=[a.T@dh,dh.sum(axis=0),h.T@delta,delta.sum(axis=0)]
                step+=1
                for i,grad in enumerate(grads):
                    grad+=1e-4*self.w[i]
                    norm=np.linalg.norm(grad)
                    if norm>5: grad*=5/norm
                    moments[i]=.9*moments[i]+.1*grad; squares[i]=.999*squares[i]+.001*grad*grad
                    self.w[i]-=.001*(moments[i]/(1-.9**step))/(np.sqrt(squares[i]/(1-.999**step))+1e-8)
            logits=np.maximum(0,vx@self.w[0]+self.w[1])@self.w[2]+self.w[3]
            value=self._loss(logits,vy)[0]; self.history.append(float(value))
            if value<best: best=value; saved=[w.copy() for w in self.w]
        self.w=saved; self.best_validation_loss=best
        return self

    def _loss(self,z,y):
        # y holds the empirical joint outcome distribution for every control.
        n=len(z); g=np.repeat(np.arange(3),3); h=np.tile(np.arange(3),3)
        if self.kind=='joint':
            p=softmax(z); return float(-np.sum(y*np.log(p+1e-12))/n),(p-y)/n
        if self.kind=='marginal':
            target=np.concatenate([y.reshape(-1,3,3).sum(axis=2),y.reshape(-1,3,3).sum(axis=1)],axis=1)
            a,b=softmax(z[:,:3]),softmax(z[:,3:]); p=np.concatenate([a,b],axis=1)
            return float(-np.sum(target*np.log(p+1e-12))/n),(p-target)/n
        if self.kind in ['scalar','independent']:
            target=y@(g-h); error=z[:,0]-target
            return float(np.mean(error**2)),2*error[:,None]/n
        # Direct policy-statistic control: the same clean-gain/risk decision,
        # trained without a joint distribution. This is a decisive comparator.
        risk=y@(h>0); clean=y@(g*(h==0)); net=y@(g-h)
        p=1/(1+np.exp(-np.clip(z[:,0],-30,30)))
        loss=-np.mean(risk*np.log(p+1e-12)+(1-risk)*np.log(1-p+1e-12))+np.mean((z[:,1]-clean)**2+(z[:,2]-net)**2)
        return float(loss),np.column_stack([p-risk,2*(z[:,1]-clean),2*(z[:,2]-net)])/n

    def predict(self,x):
        x=np.clip((np.asarray(x)-self.mean)/self.scale,-10,10)
        z=np.maximum(0,x@self.w[0]+self.w[1])@self.w[2]+self.w[3]
        if self.kind=='joint': return policy_statistics(softmax(z))
        if self.kind=='marginal':
            a,b=softmax(z[:,:3]),softmax(z[:,3:])
            return policy_statistics((a[:,:,None]*b[:,None,:]).reshape(-1,9))
        if self.kind in ['scalar','independent']:
            return dict(clean_gain=z[:,0],harm_probability=np.zeros(len(z)),mean_gain=z[:,0],mean_harm=np.zeros(len(z)))
        return dict(clean_gain=np.maximum(0,z[:,1]),harm_probability=1/(1+np.exp(-np.clip(z[:,0],-30,30))),
                    mean_gain=z[:,2],mean_harm=np.zeros(len(z)))

    def save(self,path):
        np.savez_compressed(path,mean=self.mean,scale=self.scale,**{'w'+str(i):w for i,w in enumerate(self.w)})
        Path(str(path)+'.json').write_text(json.dumps(dict(kind=self.kind,seed=self.seed,hidden=self.hidden,
            best_validation_loss=self.best_validation_loss,history=self.history),indent=2)+'\n',encoding='utf-8')

    @classmethod
    def load(cls,path):
        meta=json.loads(Path(str(path)+'.json').read_text(encoding='utf-8'))
        result=cls(meta['kind'],meta['seed'],meta['hidden'])
        with np.load(path,allow_pickle=False) as data:
            result.mean=data['mean']; result.scale=data['scale']; result.w=[data['w'+str(i)] for i in range(4)]
        result.best_validation_loss=meta['best_validation_loss']; result.history=meta['history']
        return result
