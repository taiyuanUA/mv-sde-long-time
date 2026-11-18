"""Analyze actually simulated whole-system discrepancies, never fitted values."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
DIMS=[2,3,4,5,8]
DATA=ROOT/'data'
B=5000

def main():
    file=DATA/'raw_errors.csv'
    if not file.exists():
        print('Main simulation not finished yet.'); return
    raw=pd.read_csv(file)
    Ns=np.sort(raw.N.unique()); reps=np.sort(raw.rep.unique()); times=np.sort(raw.t.unique())
    R=len(reps); nt=len(times)
    late=(times>=80)&(times<=100)
    arr=np.stack([raw[raw.N==N].sort_values(['rep','t'])[[f'mse_d{d}' for d in DIMS]].to_numpy().reshape(R,nt,len(DIMS)) for N in Ns])
    assert np.all(np.isfinite(arr)) and np.min(arr)>=0
    assert np.max(arr[:, :, 0, :])==0
    E=arr[:,:, -1,:]
    L=arr[:,:,late,:].mean(axis=2)
    C=arr.mean(axis=1)
    rows=[]
    for ni,N in enumerate(Ns):
        for di,d in enumerate(DIMS):
            a=E[ni,:,di]; latea=L[ni,:,di]; m=a.mean(); se=a.std(ddof=1)/np.sqrt(R)
            rows.append([d,N,m,se,max(0,m-1.96*se),m+1.96*se,latea.mean(),latea.std(ddof=1)/np.sqrt(R),C[ni,:,di].max()])
    summary=pd.DataFrame(rows,columns=['d','N','mse_T','se_T','ci_low_T','ci_high_T','mse_late','se_late','max_grid_mean'])
    summary.to_csv(DATA/'summary.csv',index=False)
    # Bootstrap whole independent systems, keeping dimension dependence.
    rng=np.random.default_rng(703092026)
    bt=np.empty((B,len(Ns),len(DIMS))); bl=np.empty_like(bt)
    for ni in range(len(Ns)):
        indices=rng.integers(0,R,size=(B,R))
        bt[:,ni,:]=E[ni,indices,:].mean(axis=1)
        bl[:,ni,:]=L[ni,indices,:].mean(axis=1)
    lx=np.log(Ns.astype(float)); weights=(lx-lx.mean())/np.sum((lx-lx.mean())**2)
    sb=(np.log(bt)*weights[None,:,None]).sum(axis=1)
    slb=(np.log(bl)*weights[None,:,None]).sum(axis=1)
    rates=[]
    for di,d in enumerate(DIMS):
        point=np.polyfit(lx,np.log(E[:,:,di].mean(axis=1)),1)[0]
        point_late=np.polyfit(lx,np.log(L[:,:,di].mean(axis=1)),1)[0]
        point_max=np.polyfit(lx,np.log(C[:,:,di].max(axis=1)),1)[0]
        rates.append([d,point,*np.quantile(sb[:,di],[.025,.975]),point_late,*np.quantile(slb[:,di],[.025,.975]),point_max])
    rates=pd.DataFrame(rates,columns=['d','slope_T','ci_low_T','ci_high_T','slope_late','ci_low_late','ci_high_late','slope_max_grid'])
    rates.to_csv(DATA/'rates.csv',index=False)
    timeseries=[]
    for ni,N in enumerate(Ns):
        for ti,t in enumerate(times):
            timeseries.append([N,t,*C[ni,ti,:]])
    pd.DataFrame(timeseries,columns=['N','t',*[f'mse_d{d}' for d in DIMS]]).to_csv(DATA/'time_mean.csv',index=False)
    checkpoints=[]
    for t in [1,5,10,20,50,80,100]:
        ti=int(np.flatnonzero(np.isclose(times,t))[0])
        for di,d in enumerate(DIMS):
            checkpoints.append([t,d,np.polyfit(lx,np.log(C[:,ti,di]),1)[0]])
    pd.DataFrame(checkpoints,columns=['T','d','slope']).to_csv(DATA/'checkpoint_rates.csv',index=False)
    print('MAIN RESULTS\n'+rates.to_string(index=False))
    print('SOLVER\n'+pd.read_csv(DATA/'solver_diagnostics.csv').max().to_string())


def refinement():
    path=DATA/'raw_refinement.csv'
    if not path.exists():return
    data=pd.read_csv(path)
    times=np.sort(data.t.unique()); reps=np.sort(data.rep.unique()); R=len(reps); nt=len(times)
    vals=data.sort_values(['rep','t']).iloc[:,3:].to_numpy().reshape(R,nt,15)
    rng=np.random.default_rng(807092026); idx=rng.integers(0,R,(B,R))
    rows=[]
    for label,inds in [('T100',np.isclose(times,100)),('late80_100',(times>=80)&(times<=100))]:
        means=vals[:,inds,:].mean(axis=1)
        for di,d in enumerate(DIMS):
            c=means[:,di]; f=means[:,di+5]; diff=means[:,di+10]
            ratio=100*(c.mean()/f.mean()-1)
            rb=100*(c[idx].mean(axis=1)/f[idx].mean(axis=1)-1)
            rms=np.sqrt(diff.mean()); relative_rms=rms/np.sqrt(f.mean())
            rows.append([d,label,c.mean(),f.mean(),ratio,*np.quantile(rb,[.025,.975]),rms,relative_rms])
    summary=pd.DataFrame(rows,columns=['d','window','coarse_mse','fine_mse','relative_percent','ci_low_percent','ci_high_percent','coupling_difference_rms','difference_relative_rms'])
    summary.to_csv(DATA/'refinement_summary.csv',index=False)
    print('\nREFINEMENT\n'+summary.to_string(index=False))

if __name__=='__main__':
    main();refinement()
