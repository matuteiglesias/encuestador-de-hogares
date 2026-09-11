import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from scipy.stats import spearmanr
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c'); P=Path('/media/matias/Elements1/CENSO_work/derived/eph-cpv2010-semantic-plane-2024q3-v1'); E=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af'); O=Path('science/2026-09-11/results/q3_ablations');O.mkdir(parents=True,exist_ok=True)
F=['IX_TOT','P02','P03','P05','P07','P08','P09','P10','CONDACT','V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']; fam={'composition':['IX_TOT'],'demographics':['P02','P03','P05'],'education':['P07','P08','P09','P10'],'labor':['CONDACT'],'housing':['V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']}
def metric(y,p): return {'r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'disp':np.std(p)/np.std(y)}
def hhmet(z):
 g=z.groupby('hh').agg(y=('y','sum'),p=('pred','sum')); y=g.y.values;p=g.p.values; dy=pd.qcut(y,10,labels=False,duplicates='drop');dp=pd.qcut(p,10,labels=False,duplicates='drop');d=np.abs(dy-dp);return {'r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'disp':np.std(p)/np.std(y),'spearman':spearmanr(y,p)[0],'decdisp':d.mean(),'same':(d==0).mean(),'one':(d<=1).mean(),'bottom':np.mean(p[dy==0]-y[dy==0]),'top':np.mean(p[dy==9]-y[dy==9])}
def run(d,cols):
 X=d[cols].to_numpy(float);y=d.y.to_numpy(float); pred=np.zeros(len(d)); pa=np.zeros(len(d)); cats=[cols.index(x) for x in cols if x not in ['IX_TOT','P03','H15']]
 for f in range(5):
  tr=d.fold.to_numpy()!=f;te=~tr; clf=HistGradientBoostingClassifier(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[tr],y[tr]>0); pos=tr&(y>0); reg=HistGradientBoostingRegressor(loss='gamma',learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[pos],y[pos]); pa[te]=reg.predict(X[te]); pred[te]=clf.predict_proba(X[te])[:,list(clf.classes_).index(True)]*pa[te]
 d=d.copy();d['pred']=pred;d['pa']=pa; pos=d.y>0; g=d[d.hh.isin(complete)]; return {'positive':metric(d.loc[pos,'y'],d.loc[pos,'pa']),'household':hhmet(g)}
fold=json.loads((B/'fold_manifest.json').read_text())['rows']; fm={r['row_id']:r['fold_id'] for r in fold}; ind=pd.read_csv(E/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False,usecols=['CODUSU','NRO_HOGAR','COMPONENTE','ANO4','TRIMESTRE','P47T']);ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE;p=pd.read_parquet(P/'eph_p1.parquet');d=p.merge(ind[['row_id','CODUSU','NRO_HOGAR','COMPONENTE','ANO4','TRIMESTRE','P47T']],on='row_id',validate='one_to_one');d['y']=pd.to_numeric(d.P47T,errors='coerce');d=d[d.y.notna()&(d.y>=0)].copy();d['fold_row']=d.CODUSU+'\x1f'+d.NRO_HOGAR+'\x1f'+d.COMPONENTE+'\x1f'+d.ANO4+'\x1f'+d.TRIMESTRE; d['fold']=d.fold_row.map(fm).astype(int);d['hh']=d.CODUSU+'\x1f'+d.NRO_HOGAR
h=pd.read_json(B/'household_oof.jsonl',lines=True);complete=set(h.loc[h.observed_household_income.notna(),'household_observation_id']); complete={x.split('\x1f')[0]+'\x1f'+x.split('\x1f')[1] for x in complete}
res={'full':run(d,F)}
for n,drop in fam.items():res['minus_'+n]=run(d,[x for x in F if x not in drop])
(O/'metrics.json').write_text(json.dumps(res,indent=2));print(json.dumps(res,indent=2))
