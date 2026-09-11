import json
from pathlib import Path
import pandas as pd,numpy as np
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c');O=Path('science/2026-09-11/results/q5_p2');Q=Path('science/2026-09-11/results/q6_error_reservoir');Q.mkdir(parents=True,exist_ok=True)
x=pd.read_json(O/'person_oof.jsonl',lines=True);h=pd.read_json(B/'household_oof.jsonl',lines=True);complete={z.split('\x1f')[0]+'\x1f'+z.split('\x1f')[1] for z in h.loc[h.observed_household_income.notna(),'household_observation_id']}; x=x[x.hh.isin(complete)].copy()
def m(y,p):return {'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'r2':r2_score(y,p),'disp':float(np.std(p)/np.std(y))}
def one(z,pcol):
 y=z.y.to_numpy();p=z[pcol].to_numpy(); g=pd.DataFrame({'y':y,'p':p,'hh':z.hh}).groupby('hh').sum(); return {'person':m(y,p),'household':m(g.y,g.p)}
x['normal']=x.pred;x['oracle_presence']=x.pa*(x.y>0);x['amount_observed']=np.where(x.y>0,x.y*x.p_positive,0);x['full_observed']=x.y
res={k:one(x,k) for k in ['normal','oracle_presence','amount_observed','full_observed']};(Q/'metrics.json').write_text(json.dumps({'n_person':len(x),'n_households':x.hh.nunique(),**res},indent=2));print(json.dumps({'n_person':len(x),'n_households':x.hh.nunique(),**res},indent=2))
