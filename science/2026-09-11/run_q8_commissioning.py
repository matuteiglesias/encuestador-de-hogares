"""Q8 research-only Census commissioning for frozen P1-R + calibrated distribution."""
import json,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c'); E=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af'); P=Path('/media/matias/Elements1/CENSO_work/derived/eph-cpv2010-semantic-plane-2024q3-v1'); O=Path('science/2026-09-11/results/q8_census_commissioning');O.mkdir(parents=True,exist_ok=True)
F=['IX_TOT','P02','P03','P05','P07','P08','P09','P10','CONDACT','V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']; NUM=['IX_TOT','P03','H15']; PARAM=dict(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False); SHA='7e1d7fa75684d6ebdeb694176a4dcb07363f77597425405e955b62dfaa84247d'; Z=[285610,350000,400000,475000,536000,600000,660000]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def model_fit(tr,te):
 X=tr[F].to_numpy(float);Xt=te[F].to_numpy(float);y=tr.y.to_numpy(float);cats=[F.index(c) for c in F if c not in NUM]
 clf=HistGradientBoostingClassifier(**PARAM,categorical_features=cats).fit(X,y>0);pos=y>0;reg=HistGradientBoostingRegressor(loss='gamma',**PARAM,categorical_features=cats).fit(X[pos],y[pos]); pp=clf.predict_proba(Xt)[:,list(clf.classes_).index(True)];pa=reg.predict(Xt);return pp,pa,pp*pa
def load():
 ind=pd.read_csv(E/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False);ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE;ind['fold_row']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR+'\x1f'+ind.COMPONENTE+'\x1f'+ind.ANO4+'\x1f'+ind.TRIMESTRE;ind['hh']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR;ind['y']=pd.to_numeric(ind.P47T,errors='coerce');fm={r['row_id']:r['fold_id'] for r in json.loads((B/'fold_manifest.json').read_text())['rows']};ind['fold']=ind.fold_row.map(fm);ind=ind[ind.y.notna()&(ind.y>=0)].copy();e=pd.read_parquet(P/'eph_p1.parquet');c=pd.read_parquet(P/'census_p1.parquet');return ind,e,c
ind,e,c=load();assert len(c)==469172 and c.row_id.nunique()==len(c) and c.household_id.notna().all();assert set(F)==set(c.columns)-{'row_id','household_id'} and 'H11' not in c and 'H16' not in c
# preflight/support
pre={'rows':len(c),'unique_person_ids':int(c.row_id.nunique()),'duplicate_person_ids':int(c.row_id.duplicated().sum()),'unique_households':int(c.household_id.nunique()),'missing_household_ids':int(c.household_id.isna().sum()),'missing_by_feature':{x:int(c[x].isna().sum()) for x in F},'dtypes':{x:str(c[x].dtype) for x in F},'p1_fields':F,'h11_absent':'H11' not in c,'h16_absent':'H16' not in c,'p1_parquet_sha256':sha(P/'census_p1.parquet')}
# Marginal support: categorical unseen mass; numeric range and sparse tails (<1% EPH category)
support={}; cats=[x for x in F if x not in NUM]
for x in F:
 a=e[x].dropna();b=c[x].dropna();
 if x in cats:
  ec=a.value_counts(normalize=True); cc=b.value_counts(normalize=True); support[x]={'kind':'categorical','eph_levels':int(ec.size),'census_levels':int(cc.size),'unseen_mass':float(cc[~cc.index.isin(ec.index)].sum()),'low_support_mass_lt1pct':float(cc[cc.index.map(lambda z:ec.get(z,0)<.01)].sum()),'eph_frequency':{str(k):float(v) for k,v in ec.items()},'census_frequency':{str(k):float(v) for k,v in cc.items()}}
 else:
  support[x]={'kind':'numeric','eph_range':[float(a.min()),float(a.max())],'census_range':[float(b.min()),float(b.max())],'outside_eph_support':float(((b<a.min())|(b>a.max())).mean()),'eph_quantiles':{str(q):float(a.quantile(q)) for q in [.01,.05,.5,.95,.99]},'census_quantiles':{str(q):float(b.quantile(q)) for q in [.01,.05,.5,.95,.99]}}
# household support class predetermined by marginal training support
seen={x:set(e[x].dropna().unique()) for x in cats}; emin={x:e[x].min() for x in NUM};emax={x:e[x].max() for x in NUM}
c['support_weak']=False
for x in cats:c['support_weak']|=~c[x].isin(seen[x])
for x in NUM:c['support_weak']|=(c[x]<emin[x])|(c[x]>emax[x])
hsupport=c.groupby('household_id').support_weak.any(); support_summary={'person_weak_fraction':float(c.support_weak.mean()),'households':int(hsupport.size),'weak_households':int(hsupport.sum()),'high_support_households':int((~hsupport).sum()),'definition':'all categorical values seen in eligible EPH and all numeric values within EPH observed bounds'}
# Domain classifier diagnostic
mix=pd.concat([e[F].assign(domain=0),c[F].assign(domain=1)],ignore_index=True); X=mix[F].to_numpy(float);y=mix.domain.to_numpy(); catsidx=[F.index(x) for x in F if x not in NUM];dc=HistGradientBoostingClassifier(max_iter=60,max_leaf_nodes=31,min_samples_leaf=50,random_state=42,early_stopping=False,categorical_features=catsidx).fit(X,y);domain_prob=dc.predict_proba(c[F].to_numpy(float))[:,list(dc.classes_).index(1)];auc=roc_auc_score(y,dc.predict_proba(X)[:,list(dc.classes_).index(1)]);c['domain_prob']=domain_prob
support_summary.update({'domain_auc_in_sample':float(auc),'census_domain_prob_q':{str(q):float(np.quantile(domain_prob,q)) for q in [.01,.05,.5,.95,.99]},'census_extreme_domain_prob_gt99':float(np.mean(domain_prob>.99))})
# OOF EPH calibrated residuals from same frozen P1-R model; aggregate complete households
h0=pd.read_json(B/'household_oof.jsonl',lines=True);complete={z.split('\x1f')[0]+'\x1f'+z.split('\x1f')[1] for z in h0.loc[h0.observed_household_income.notna(),'household_observation_id']};e=e.merge(ind[['row_id','hh','fold','y']],on='row_id',validate='one_to_one');e['pp']=np.nan;e['pa']=np.nan;e['pred']=np.nan
for f in range(5):
 tr=e.fold!=f;te=~tr;pp,pa,pred=model_fit(e[tr],e[te]);e.loc[te,'pp']=pp;e.loc[te,'pa']=pa;e.loc[te,'pred']=pred
g=e[e.hh.isin(complete)].groupby('hh').agg(y=('y','sum'),pred=('pred','sum'),n=('y','size'));residuals=np.sort((g.y-g.pred).to_numpy());
# Final model fit on all eligible EPH and score all Census persons
pp,pa,pred=model_fit(e,c);c['p_positive']=pp;c['positive_amount_prediction']=pa;c['expected_person_income']=pred
cg=c.groupby('household_id',sort=True).agg(member_count=('row_id','size'),predicted_household_income=('expected_person_income','sum'),mean_p_positive=('p_positive','mean'),max_ix_tot=('IX_TOT','max'),support_weak=('support_weak','any'),domain_prob=('support_weak','mean')).reset_index();
# prevalence and household bootstrap conditional on frozen model
rows=[]; rng=np.random.default_rng(42)
for z in Z:
 p=np.mean(residuals[None,:] <= (z-cg.predicted_household_income.to_numpy())[:,None],axis=1);hard=cg.predicted_household_income.to_numpy()<=z
 boots=[]
 for _ in range(300):boots.append(float(np.mean(p[rng.integers(0,len(p),len(p))])))
 rows.append({'threshold':z,'hard_prevalence':float(hard.mean()),'probabilistic_prevalence':float(p.mean()),'difference_prob_minus_hard':float(p.mean()-hard.mean()),'bootstrap_95_low':float(np.quantile(boots,.025)),'bootstrap_95_high':float(np.quantile(boots,.975))})
# support sensitivity
sens=[]
for label,mask in [('all',np.ones(len(cg),bool)),('high_support',~cg.support_weak.to_numpy(bool)),('weak_support',cg.support_weak.to_numpy(bool))]:
 for z in Z:
  p=np.mean(residuals[None,:] <= (z-cg.loc[mask,'predicted_household_income'].to_numpy())[:,None],axis=1);hard=cg.loc[mask,'predicted_household_income'].to_numpy()<=z;sens.append({'universe':label,'threshold':z,'households':int(mask.sum()),'hard_prevalence':float(hard.mean()),'probabilistic_prevalence':float(p.mean())})
c[['row_id','household_id','p_positive','positive_amount_prediction','expected_person_income','support_weak','domain_prob']].to_parquet(O/'person_predictions.parquet',index=False);cg.to_parquet(O/'household_predictions.parquet',index=False)
(O/'census_preflight.json').write_text(json.dumps(pre,indent=2));(O/'feature_support_marginals.json').write_text(json.dumps(support,indent=2));(O/'multivariate_support.json').write_text(json.dumps(support_summary,indent=2));(O/'domain_shift_diagnostics.json').write_text(json.dumps({'auc_in_sample':float(auc),'note':'diagnostic only; no domain adaptation'},indent=2));(O/'threshold_prevalence.json').write_text(json.dumps({'thresholds':Z,'rows':rows,'label':'research thresholds, not official poverty lines'},indent=2));(O/'transport_sensitivity.json').write_text(json.dumps(sens,indent=2));(O/'universe_definition.json').write_text(json.dumps({'primary':'all governed Census P1-R household IDs; P1 artifact contains no private/collective dwelling indicator, so no defensible split was invented','secondary':'high-support vs weak-support diagnostic subsets','collective_indicator_available':False},indent=2));(O/'universe_counts.json').write_text(json.dumps({'all_households':len(cg),'high_support':int((~cg.support_weak).sum()),'weak_support':int(cg.support_weak.sum())},indent=2));(O/'deployment_contract.json').write_text(json.dumps({'features':F,'h11_absent':True,'h16_absent':True,'mean_model':PARAM,'terminal':'direct hurdle-Gamma','calibration':'EPH-only 5-fold OOF household residual ECDF','residual_count':len(residuals),'thresholds':Z,'weights':'none','ix_tot_clipped':False},indent=2));(O/'input_identity.json').write_text(json.dumps({'git_sha':'a8c122c','fold_manifest_sha256':SHA,'semantic_release':'eph-cpv2010-semantic-plane-2024q3-v1','census_rows':len(c),'census_parquet_sha256':pre['p1_parquet_sha256']},indent=2));print(json.dumps({'preflight':pre,'support':support_summary,'thresholds':rows},indent=2))
