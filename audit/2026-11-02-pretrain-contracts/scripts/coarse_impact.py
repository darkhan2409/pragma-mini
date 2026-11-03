import json, numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from scipy.stats import spearmanr
b=json.load(open('data/03_vocab/buckets.json'))
def scale(key, when=None, split=False):
    items=[v for v in b[key].values() if (not split or v.get('when')==when) and not (v['min']==0 and v['max']==0)]
    return sorted(items, key=lambda v: -1e30 if v['min'] is None else v['min'])
def rep(vals, items):
    # representative value of bucket: geometric-ish mid; open ends -> the finite edge
    out=np.empty(len(vals)); idx=np.empty(len(vals),int)
    edges=[v['max'] for v in items[:-1]]
    k=np.searchsorted(edges, vals, side='right')
    for i,v in enumerate(items):
        lo,hi=v['min'],v['max']
        r = hi if lo is None else (lo if hi is None else (lo+hi)/2)
        out[k==i]=r
    out[vals==0]=0
    return out, np.where(vals==0,-1,k)
T=pd.Timestamp('2026-02-28T19:00:00Z')
rows=pd.read_parquet('churn_baseline/reports/eval_rows.parquet').client_id
f=pq.ParquetFile('data/02_preprocessed/val/events.parquet')
parts=[]
for i in range(f.num_row_groups):
    t=f.read_row_group(i,columns=['client_id','event_time','type','direction','amount','balance_after','reason'])
    t=t.filter(pc.and_(pc.less(t.column('event_time'),pa.scalar(T.to_pydatetime(),pa.timestamp('us',tz='UTC'))),
                       pc.greater_equal(t.column('event_time'),pa.scalar((T-pd.Timedelta(days=90)).to_pydatetime(),pa.timestamp('us',tz='UTC')))))
    parts.append(t.to_pandas())
df=pd.concat(parts,ignore_index=True); df=df[df.client_id.isin(set(rows))].reset_index(drop=True)
age=(T-df.event_time).dt.total_seconds()/86400
deb=scale('transaction_amount','debit',True); cre=scale('transaction_amount','credit',True)
def flowsum(mask, items, days):
    d=df[mask&(age<=days)&df.amount.notna()].copy()
    d['rep'],_=rep(d.amount.values.astype(float), items)
    g=d.groupby('client_id').agg(exact=('amount','sum'),approx=('rep','sum'))
    return g.reindex(rows).fillna(0)
for name,mask,items,days in [('transfer_sum_30',df.type.isin(['transfer_out','p2p_out']),deb,30),
                             ('cash_out_sum_30',df.type.eq('cash_withdrawal'),deb,30),
                             ('purchase_sum_90',df.type.eq('purchase')&df.reason.eq('purchase'),deb,90),
                             ('transfer_in_sum_90',df.type.isin(['transfer_in','p2p_in']),cre,90),
                             ('income_sum_90',df.type.isin(['salary_credit','pension_credit','other_income_credit']),cre,90),
                             ('cash_in_sum_30',df.type.eq('cash_deposit'),cre,30)]:
    g=flowsum(mask,items,days); nz=g[g.exact>0]
    print(f'{name}: clients {len(g)} nonzero {len(nz)} spearman(exact, bucket-midpoint sum) {spearmanr(nz.exact,nz.approx).statistic:.4f} pearson log {np.corrcoef(np.log1p(nz.exact),np.log1p(nz.approx))[0,1]:.4f}')
# balance: last value and 30d ago
bal=scale('balance_after')
d=df[df.balance_after.notna()].sort_values(['client_id','event_time'])
last=d.groupby('client_id').balance_after.last()
ago=d[(age.loc[d.index]>30).values].groupby('client_id').balance_after.last()
j=pd.concat([last.rename('last'),ago.rename('ago')],axis=1).dropna()
_,kl=rep(j['last'].values.astype(float),bal); _,ka=rep(j['ago'].values.astype(float),bal)
diff=j['last']!=j['ago']
print('balance: clients',len(j),'changed over 30d',diff.sum(),'changed but same bucket',(diff&(kl==ka)).sum(), 'share %.3f'%((diff&(kl==ka)).sum()/max(1,diff.sum())))
rel=(j['last']-j['ago']).abs()/j['ago'].abs().clip(lower=1)
print('   of those same-bucket changes, median |relative change| %.3f'%rel[diff&(kl==ka)].median())
