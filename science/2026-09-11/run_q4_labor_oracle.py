import json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import *
from scipy.stats import spearmanr
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c');P=Path('/media/matias/Elements1/CENSO_work/derived/eph-cpv2010-semantic-plane-2024q3-v1');E=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af');O=Path('science/2026-09-11/results/q4_labor');O.mkdir(parents=True,exist_ok=True)
F=['IX_TOT','P02','P03','P05','P07','P08','P09','P10','CONDACT','V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']; Xf=[x for x in F if x!='CONDACT']
fold=json.loads((B/'fold_manifest.json').read_text())['rows'];fm={r['row_id']:r['fold_id'] for r in fold};ind=pd.read_csv(E/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False,usecols=['CODUSU','NRO_HOGAR','COMPONENTE','ANO4','TRIMESTRE','P47T']);ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE;ind['fold_row']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR+'\x1f'+ind.COMPONENTE+'\x1f'+ind.ANO4+'\x1f'+ind.TRIMESTRE;p=pd.read_parquet(P/'eph_p1.parquet');d=p.merge(ind[['row_id','fold_row','CODUSU','NRO_HOGAR','P47T']],on='row_id',validate='one_to_one');d['y']=pd.to_numeric(d.P47T,errors='coerce');d=d[d.y.notna()&(d.y>=0)].copy();d['fold']=d.fold_row.map(fm).astype(int);d['hh']=d.CODUSU+'\x1f'+d.NRO_HOGAR
co=pd.read_json(B/'household_oof.jsonl',lines=True);complete={z.split('\x1f')[0]+'\x1f'+z.split('\x1f')[1] for z in co.loc[co.observed_household_income.notna(),'household_observation_id']}
def fitpred(cols, labor_probs=None, truth=False):
 dd=d.copy(); pred=np.zeros(len(dd));pa=np.zeros(len(dd)); pp=np.zeros(len(dd)); cats=[cols.index(x) for x in cols if x not in ['IX_TOT','P03','H15']]
 Z=dd[cols].to_numpy(float)
 if labor_probs is not None: Z=np.column_stack([Z,labor_probs]); cats=[]
 for f in range(5):
  tr=dd.fold.to_numpy()!=f;te=~tr;clf=HistGradientBoostingClassifier(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(Z[tr],dd.y.to_numpy()[tr]>0);pos=tr&(dd.y.to_numpy()>0);reg=HistGradientBoostingRegressor(loss='gamma',learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(Z[pos],dd.y.to_numpy()[pos]);pp[te]=clf.predict_proba(Z[te])[:,list(clf.classes_).index(True)];pa[te]=reg.predict(Z[te]);pred[te]=pp[te]*pa[te]
 dd['pred']=pred;return dd
# household-safe labor probabilities
L=d[Xf].to_numpy(float); lc=[Xf.index(x) for x in Xf if x not in ['IX_TOT','P03','H15']]; probs=np.zeros((len(d),4))
for f in range(5):
 tr=d.fold.to_numpy()!=f;te=~tr; m=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,learning_rate=.08,random_state=42,early_stopping=False,categorical_features=lc).fit(L[tr],d.CONDACT.astype(str).to_numpy()[tr]); q=m.predict_proba(L[te]);
 for j,c in enumerate(m.classes_): probs[te,j]=q[:,j]
# ensure fixed 4 columns enough, pad is okay
A=fitpred(Xf);Bv=fitpred(F);C=fitpred(Xf,probs)
def met(z):
 y=z.y.values;p=z.pred.values;return {'r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'disp':np.std(p)/np.std(y)}
def hm(z):
 g=z[z.hh.isin(complete)].groupby('hh').agg(y=('y','sum'),p=('pred','sum'));y=g.y.values;p=g.p.values;dy=pd.qcut(y,10,labels=False,duplicates='drop');dp=pd.qcut(p,10,labels=False,duplicates='drop');return dict(**met(pd.DataFrame({'y':y,'pred':p})),spearman=spearmanr(y,p)[0],decdisp=float(np.abs(dy-dp).mean()),bottom=float(np.mean(p[dy==0]-y[dy==0])),top=float(np.mean(p[dy==9]-y[dy==9])))
res={k:{'positive':met(v[v.y>0]),'household':hm(v)} for k,v in [('A_X',A),('B_true_labor',Bv),('C_oof_labor_probs',C)]};(O/'metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2));
# labor classification diagnostics pooled OOF impossible from probs without target labels; report accuracy/logloss from probs against CONDACT classes using known class order
