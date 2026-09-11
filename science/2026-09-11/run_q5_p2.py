import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import *
from scipy.stats import spearmanr
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c');E=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af');O=Path('science/2026-09-11/results/q5_p2');O.mkdir(parents=True,exist_ok=True)
ind=pd.read_csv(E/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False); hh=pd.read_csv(E/'household/usu_hogar_t324.txt',sep=';',dtype=str,keep_default_na=False)
ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE; ind['fold_row']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR+'\x1f'+ind.COMPONENTE+'\x1f'+ind.ANO4+'\x1f'+ind.TRIMESTRE; ind['hh']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR
hkey=hh.CODUSU+'\x1f'+hh.NRO_HOGAR; hh['hh']=hkey
# governed non-income blocks: raw source-faithful demographic/education/labor/housing + household age pyramid
cols=['CH04','CH07','P03','NIVEL_ED','ESTADO','CAT_INAC','CAT_OCUP','CONDACT','PP07G_59']
hcols=['V1','V2','V3','V5_02','V5_03','V6','V7','V8','V9','V10','V11_01','V11_02','V12','V13','V14','V15','V16','V17','V18','V19_A','IX_TOT']
for c in cols+hcols: ind[c]=ind[c] if c in ind else np.nan
x=ind[['row_id','fold_row','hh','P47T']+cols].merge(hh[['hh']+hcols],on='hh',validate='many_to_one'); x['y']=pd.to_numeric(x.P47T,errors='coerce');x=x[x.y.notna()&(x.y>=0)].copy(); x['fold']=x.fold_row.map({r['row_id']:r['fold_id'] for r in json.loads((B/'fold_manifest.json').read_text())['rows']}).astype(int)
# numeric conversion all; use categorical positions for discrete fields
F=cols+hcols
for c in F:x[c]=pd.to_numeric(x[c],errors='coerce')
cats=[F.index(c) for c in F if c not in ['P03','IX_TOT']]
y=x.y.to_numpy();X=x[F].to_numpy(float);pred=np.zeros(len(x));pp=np.zeros(len(x));pa=np.zeros(len(x)); rows=[]
for f in range(5):
 tr=x.fold.to_numpy()!=f;te=~tr;clf=HistGradientBoostingClassifier(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[tr],y[tr]>0);pos=tr&(y>0);reg=HistGradientBoostingRegressor(loss='gamma',learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[pos],y[pos]);pp[te]=clf.predict_proba(X[te])[:,list(clf.classes_).index(True)];pa[te]=reg.predict(X[te]);pred[te]=pp[te]*pa[te]
x['pred']=pred;x['pa']=pa;x['p_positive']=pp
co=pd.read_json(B/'household_oof.jsonl',lines=True);complete={z.split('\x1f')[0]+'\x1f'+z.split('\x1f')[1] for z in co.loc[co.observed_household_income.notna(),'household_observation_id']}
def met(y,p):return {'r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'disp':float(np.std(p)/np.std(y))}
def hm(z):
 g=z.groupby('hh').agg(y=('y','sum'),p=('pred','sum'));y=g.y.values;p=g.p.values;dy=pd.qcut(y,10,labels=False,duplicates='drop');dp=pd.qcut(p,10,labels=False,duplicates='drop');d=np.abs(dy-dp);return dict(**met(y,p),spearman=spearmanr(y,p)[0],decdisp=float(d.mean()),same=float((d==0).mean()),one=float((d<=1).mean()),bottom=float(np.mean(p[dy==0]-y[dy==0])),top=float(np.mean(p[dy==9]-y[dy==9])))
res={'positive':met(y[y>0],pa[y>0]),'household':hm(x[x.hh.isin(complete)])};x[['row_id','fold','hh','y','p_positive','pa','pred']].to_json(O/'person_oof.jsonl',orient='records',lines=True);(O/'metrics.json').write_text(json.dumps({'features':F,'n':len(x),**res},indent=2));print(json.dumps({'features':F,'n':len(x),**res},indent=2))
