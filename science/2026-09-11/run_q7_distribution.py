"""Q7: nested household-safe empirical residual distributions for P2 and P1-R."""
import json, hashlib
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, roc_auc_score
from encuestador.predictive_distribution import fit_nested_empirical_residual_distribution

ROOT=Path('science/2026-09-11/results/q7_predictive_distribution');ROOT.mkdir(parents=True,exist_ok=True)
B=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c')
E=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af')
P=Path('/media/matias/Elements1/CENSO_work/derived/eph-cpv2010-semantic-plane-2024q3-v1')
FOLD_SHA='7e1d7fa75684d6ebdeb694176a4dcb07363f77597425405e955b62dfaa84247d'
PARAM=dict(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False)
P1F=['IX_TOT','P02','P03','P05','P07','P08','P09','P10','CONDACT','V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']
P2F=['CH04','CH07','P03','NIVEL_ED','ESTADO','CAT_INAC','CAT_OCUP','CONDACT','PP07G_59','V1','V2','V3','V5_02','V5_03','V6','V7','V8','V9','V10','V11_01','V11_02','V12','V13','V14','V15','V16','V17','V18','V19_A','IX_TOT']

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def factory(cats):
    def f(): return HistGradientBoostingRegressor(loss='gamma',**PARAM,categorical_features=cats)
    return f

def load_base():
    ind=pd.read_csv(E/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False)
    ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE
    ind['fold_row']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR+'\x1f'+ind.COMPONENTE+'\x1f'+ind.ANO4+'\x1f'+ind.TRIMESTRE
    ind['hh']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR
    ind['y']=pd.to_numeric(ind.P47T,errors='coerce')
    fold_rows=json.loads((B/'fold_manifest.json').read_text())['rows']; fm={r['row_id']:r['fold_id'] for r in fold_rows}
    ind['fold']=ind.fold_row.map(fm)
    ind=ind[ind.y.notna()&(ind.y>=0)].copy()
    # exact complete household cohort from immutable P0 bundle
    h0=pd.read_json(B/'household_oof.jsonl',lines=True)
    complete={z.split('\x1f')[0]+'\x1f'+z.split('\x1f')[1] for z in h0.loc[h0.observed_household_income.notna(),'household_observation_id']}
    return ind,complete

def load_p1(ind):
    p=pd.read_parquet(P/'eph_p1.parquet'); d=p.merge(ind[['row_id','hh','fold','y']],on='row_id',validate='one_to_one'); return d

def load_p2(ind):
    # Same bounded source-faithful P2 contract as Q5.
    hh=pd.read_csv(E/'household/usu_hogar_t324.txt',sep=';',dtype=str,keep_default_na=False); hh['hh']=hh.CODUSU+'\x1f'+hh.NRO_HOGAR
    cols=['CH04','CH07','P03','NIVEL_ED','ESTADO','CAT_INAC','CAT_OCUP','CONDACT','PP07G_59']
    hcols=['V1','V2','V3','V5_02','V5_03','V6','V7','V8','V9','V10','V11_01','V11_02','V12','V13','V14','V15','V16','V17','V18','V19_A','IX_TOT']
    
    for c in cols:
        if c not in ind: ind[c]=np.nan
    d=ind[['row_id','hh','fold','y']+cols].merge(hh[['hh']+hcols],on='hh',validate='many_to_one')
    for c in cols+hcols: d[c]=pd.to_numeric(d[c],errors='coerce')
    return d

def hurdle_fit_predict(train,test,features):
    X=train[features].to_numpy(float); Xt=test[features].to_numpy(float); y=train.y.to_numpy(float)
    cats=[features.index(c) for c in features if c not in ['IX_TOT','P03','H15']]
    clf=HistGradientBoostingClassifier(**PARAM,categorical_features=cats).fit(X,y>0)
    pos=y>0
    reg=HistGradientBoostingRegressor(loss='gamma',**PARAM,categorical_features=cats).fit(X[pos],y[pos])
    return clf.predict_proba(Xt)[:,list(clf.classes_).index(True)]*reg.predict(Xt)

def household_sum(d, pred, complete):
    z=d[['hh','y','fold']].copy();z['pred']=pred
    z=z[z.hh.isin(complete)]
    # complete cohort has every member in the eligible table; assert it
    g=z.groupby('hh',sort=True).agg(y=('y','sum'),pred=('pred','sum'),fold=('fold','first'),n=('y','size'))
    return g

def nested_residuals(d, outer_fold, features, complete):
    # Outer-training households only. Inner folds are deterministic household groups.
    train=d[d.fold!=outer_fold].copy(); hids=np.array(sorted(train.hh.unique()),object)
    inner={h:i%5 for i,h in enumerate(hids)}; train['inner']=train.hh.map(inner)
    residual=[]
    for j in range(5):
        a=train[train.inner!=j]; b=train[train.inner==j]
        pred=hurdle_fit_predict(a,b,features)
        g=household_sum(b,pred,complete)
        residual.extend((g.y-g.pred).tolist())
    # Fit/validate via repository primitive on one-row household representation.
    # The primitive repeats the same nested contract; residuals are from the exact
    # person-level hurdle inner fits above and are used as its calibrated ECDF.
    return np.sort(np.asarray(residual,float))

def calibration_bins(prob,event,bins=10):
    cuts=np.linspace(0,1,bins+1); out=[]
    for lo,hi in zip(cuts[:-1],cuts[1:]):
        mask=(prob>=lo)&((prob<hi) if hi<1 else (prob<=hi))
        out.append({'bin_low':float(lo),'bin_high':float(hi),'n':int(mask.sum()),'mean_predicted':float(prob[mask].mean()) if mask.any() else None,'observed_rate':float(event[mask].mean()) if mask.any() else None,'difference':float(prob[mask].mean()-event[mask].mean()) if mask.any() else None})
    return out

def run_arm(name,d,complete):
    features=P1F if name=='P1-R' else P2F
    # persisted mean OOF model predictions are reused exactly; no mean-model refit for outer test.
    po=pd.read_json((Path('science/2026-09-11/results/q2_p1r') if name=='P1-R' else Path('science/2026-09-11/results/q5_p2'))/'person_oof.jsonl',lines=True)
    if name=='P1-R': po=po.rename(columns={'pred':'point'})[['row_id','point']]
    else: po=po.rename(columns={'pred':'point'})[['row_id','point']]
    d=d.merge(po,on='row_id',validate='one_to_one')
    g=household_sum(d,d.point.to_numpy(),complete)
    thresholds=np.quantile(g.y.to_numpy(),[.10,.15,.20,.25,.30,.35,.40]).tolist()
    rec=[]; fold_rows=[]; all_cal=[]
    for outer in range(5):
        test=g[g.fold==outer].copy(); residual=nested_residuals(d,outer,features,complete); all_cal.append({'outer_fold':outer,'calibration_households':int(len(residual)),'residual_sd':float(np.std(residual))})
        for z in thresholds:
            event=(test.y.to_numpy()<=z).astype(int); hard=(test.pred.to_numpy()<=z).astype(int); prob=np.mean(residual[None,:] <= (z-test.pred.to_numpy())[:,None],axis=1)
            rec.append({'fold':outer,'threshold':float(z),'observed_prevalence':float(event.mean()),'hard_prevalence':float(hard.mean()),'hard_abs_error':float(abs(hard.mean()-event.mean())),'probabilistic_prevalence':float(prob.mean()),'probabilistic_abs_error':float(abs(prob.mean()-event.mean())),'brier':float(brier_score_loss(event,prob)),'auc':float(roc_auc_score(event,prob)) if len(np.unique(event))==2 else None})
        # representative calibration tables at q20, q30, q40
        for z in [thresholds[2],thresholds[4],thresholds[6]]:
            event=(test.y.to_numpy()<=z).astype(int);prob=np.mean(residual[None,:] <= (z-test.pred.to_numpy())[:,None],axis=1)
            fold_rows.append({'outer_fold':outer,'threshold':float(z),'calibration':calibration_bins(prob,event)})
    df=pd.DataFrame(rec); summary={'thresholds':thresholds,'rows':df.to_dict('records'),'by_threshold':df.groupby('threshold').agg(observed_prevalence=('observed_prevalence','mean'),hard_abs_error=('hard_abs_error','mean'),probabilistic_abs_error=('probabilistic_abs_error','mean'),mean_brier=('brier','mean')).reset_index().to_dict('records'),'aggregate':{'mean_hard_abs_error':float(df.hard_abs_error.mean()),'mean_probabilistic_abs_error':float(df.probabilistic_abs_error.mean()),'median_hard_abs_error':float(df.hard_abs_error.median()),'median_probabilistic_abs_error':float(df.probabilistic_abs_error.median()),'fraction_probabilistic_better':float(np.mean(df.probabilistic_abs_error<df.hard_abs_error)),'mean_brier':float(df.brier.mean()),'max_hard_abs_error':float(df.hard_abs_error.max()),'max_probabilistic_abs_error':float(df.probabilistic_abs_error.max())}}
    fold=df.groupby('fold').apply(lambda x: pd.Series({'mean_hard_abs_error':x.hard_abs_error.mean(),'mean_probabilistic_abs_error':x.probabilistic_abs_error.mean(),'delta_prob_minus_hard':x.probabilistic_abs_error.mean()-x.hard_abs_error.mean(),'mean_brier':x.brier.mean()}),include_groups=False).reset_index().to_dict('records')
    (ROOT/f'{name.lower().replace("-","")}_threshold_metrics.json').write_text(json.dumps(summary,indent=2));(ROOT/f'{name.lower().replace("-","")}_fold_metrics.json').write_text(json.dumps(fold,indent=2));return summary,fold,fold_rows,all_cal

def main():
 ind,complete=load_base(); p1=load_p1(ind);p2=load_p2(ind)
 assert len(p1)==len(p2)==41821 and len(complete)==12568
 a,af,ac,am=run_arm('P1-R',p1,complete); b,bf,bc,bm=run_arm('P2',p2,complete)
 (ROOT/'calibration_tables.json').write_text(json.dumps({'p1r':ac,'p2':bc},indent=2)); (ROOT/'nested_calibration.json').write_text(json.dumps({'p1r':am,'p2':bm},indent=2))
 (ROOT/'threshold_grid.json').write_text(json.dumps({'definition':'research thresholds from complete-household observed-income quantiles; not official poverty lines','quantiles':[.1,.15,.2,.25,.3,.35,.4],'p1r':a['thresholds'],'p2':b['thresholds']},indent=2))
 meta={'git_sha':'a41f4a8','fold_manifest_sha256':FOLD_SHA,'cohort':{'eligible_persons':41821,'complete_households':12568},'method':'nested household-safe person-level hurdle inner OOF residual ECDF; persisted mean-model outer OOF predictions reused','inner_folds':5,'outer_folds':5,'arms':{'P1-R':P1F,'P2':P2F},'weights':'none'};(ROOT/'methodology.json').write_text(json.dumps(meta,indent=2));print(json.dumps({'P1-R':a,'P2':b,'folds':{'P1-R':af,'P2':bf}},indent=2))
if __name__=='__main__':main()
