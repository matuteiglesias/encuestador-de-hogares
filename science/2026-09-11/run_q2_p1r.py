"""Bounded, immutable-fold Q2 P0 vs P1-R execution."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from scipy.stats import spearmanr

BASE=Path('/home/matias/Downloads/real-eph-2024q3-science-evidence/encuestador-runs/real_eph_2024q3_direct_hurdle_gamma_v1-7f010f6cb22b4d9c')
PLANE=Path('/media/matias/Elements1/CENSO_work/derived/eph-cpv2010-semantic-plane-2024q3-v1')
EPH=Path('/home/matias/data/poverty-integration-20260910/eph-releases/eph-2024-q3-3b6a7a15c4af')
OUT=Path(__file__).parent/'results/q2_p1r'; OUT.mkdir(parents=True,exist_ok=True)
FIELDS=['IX_TOT','P02','P03','P05','P07','P08','P09','P10','CONDACT','V01','H05','H06','H07','H08','H09','H10','H12','H13','H14','H15','PROP']
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def metric(y,p):
 return {'r2':r2_score(y,p),'mae':mean_absolute_error(y,p),'rmse':mean_squared_error(y,p)**.5,'observed_sd':float(np.std(y)),'predicted_sd':float(np.std(p)),'dispersion_ratio':float(np.std(p)/np.std(y))}
def hh(df):
 g=df.groupby('hh',sort=False).agg(y=('y','sum'),p=('pred','sum'),n=('y','size'))
 # exact complete household cohort is recovered from P0 persisted records.
 return g
def ranks(g):
 y,p=g.y.to_numpy(),g.p.to_numpy(); decy=pd.qcut(y,10,labels=False,duplicates='drop'); decp=pd.qcut(p,10,labels=False,duplicates='drop'); d=np.abs(decy-decp)
 return {'spearman':float(spearmanr(y,p)[0]),'mean_decile_displacement':float(d.mean()),'same_decile_share':float((d==0).mean()),'within_one_decile_share':float((d<=1).mean()),'bottom_decile_bias':float(p[decy==0].mean()-y[decy==0].mean()),'top_decile_bias':float(p[decy==9].mean()-y[decy==9].mean())}
def main():
 foldp=BASE/'fold_manifest.json'; assert sha(foldp)=='7e1d7fa75684d6ebdeb694176a4dcb07363f77597425405e955b62dfaa84247d'
 folds=json.loads(foldp.read_text())['rows']; fmap={r['row_id']:r['fold_id'] for r in folds}; assert len(fmap)==47564
 p1=pd.read_parquet(PLANE/'eph_p1.parquet'); assert len(p1)==47564 and p1.row_id.nunique()==47564 and set(FIELDS)<=set(p1) and 'H11' not in p1 and 'H16' not in p1
 ind=pd.read_csv(EPH/'individual/usu_individual_t324.txt',sep=';',dtype=str,keep_default_na=False,usecols=['CODUSU','NRO_HOGAR','COMPONENTE','ANO4','TRIMESTRE','P47T'])
 ind['row_id']=ind.CODUSU+':'+ind.NRO_HOGAR+':'+ind.COMPONENTE; ind['fold_row']=ind.CODUSU+'\x1f'+ind.NRO_HOGAR+'\x1f'+ind.COMPONENTE+'\x1f'+ind.ANO4+'\x1f'+ind.TRIMESTRE
 assert len(ind)==47564 and ind.row_id.nunique()==47564
 d=p1.merge(ind[['row_id','fold_row','CODUSU','NRO_HOGAR','P47T']],on='row_id',how='outer',indicator=True,validate='one_to_one'); assert (d._merge=='both').all(),d._merge.value_counts().to_dict()
 d['fold']=d.fold_row.map(fmap); d['y']=pd.to_numeric(d.P47T,errors='coerce'); eligible=d.y.notna()&(d.y>=0); d=d[eligible].copy(); assert len(d)==41821 and d.fold.notna().all()
 p0fold=[r['fold_id'] for r in (json.loads(x) for x in (BASE/'person_oof.jsonl').read_text().splitlines()) if r['target_status'] in ('positive','zero')]
 assert [int((d.fold==i).sum()) for i in range(5)]==[p0fold.count(i) for i in range(5)]
 d['hh']=d.CODUSU+'\x1f'+d.NRO_HOGAR; assert d.groupby('hh').fold.nunique().max()==1
 X=d[FIELDS].to_numpy(float); y=d.y.to_numpy(float); pred=np.zeros(len(d)); pp=np.zeros(len(d)); pa=np.zeros(len(d)); cats=[FIELDS.index(x) for x in FIELDS if x not in ['IX_TOT','P03','H15']]
 for f in range(5):
  tr=d.fold.to_numpy()!=f; te=~tr; clf=HistGradientBoostingClassifier(learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[tr],y[tr]>0)
  pos=tr&(y>0); reg=HistGradientBoostingRegressor(loss='gamma',learning_rate=.08,max_iter=60,max_leaf_nodes=31,min_samples_leaf=20,random_state=42,early_stopping=False,categorical_features=cats).fit(X[pos],y[pos])
  pp[te]=clf.predict_proba(X[te])[:,list(clf.classes_).index(True)]; pa[te]=reg.predict(X[te]); pred[te]=pp[te]*pa[te]
 d['pred']=pred;d['p_positive']=pp;d['positive_amount_prediction']=pa
 pos=d.y>0; m={'positive':metric(y[pos],pa[pos]),'household':metric(hh(d).y,hh(d).p)}; m['household'].update(ranks(hh(d)))
 foldsout=[]
 for f in range(5):
  z=d[d.fold==f]; zp=z.y>0; q={'fold':f,'positive':metric(z.y[zp],z.positive_amount_prediction[zp]),'household':metric(hh(z).y,hh(z).p)};q['household'].update(ranks(hh(z)));foldsout.append(q)
 d[['row_id','fold','y','p_positive','positive_amount_prediction','pred','hh']].to_json(OUT/'person_oof.jsonl',orient='records',lines=True)
 hh(d).reset_index().to_json(OUT/'household_oof.jsonl',orient='records',lines=True)
 (OUT/'metrics.json').write_text(json.dumps(m,indent=2));(OUT/'foldwise_metrics.json').write_text(json.dumps(foldsout,indent=2));(OUT/'preflight.json').write_text(json.dumps({'p1_rows':47564,'eligible':len(d),'fold_counts':[int((d.fold==i).sum()) for i in range(5)],'fold_sha256':sha(foldp),'p1_sha256':sha(PLANE/'eph_p1.parquet'),'fields':FIELDS},indent=2)); print(json.dumps(m,indent=2))
if __name__=='__main__': main()
