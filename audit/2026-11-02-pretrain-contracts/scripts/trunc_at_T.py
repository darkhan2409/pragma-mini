import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
rows=pd.concat([pd.read_parquet('churn_baseline/reports/train_rows.parquet'),pd.read_parquet('churn_baseline/reports/eval_rows.parquet')])
for g in ['train','val']:
    T={'train':pd.Timestamp('2025-12-31T19:00:00Z'),'val':pd.Timestamp('2026-02-28T19:00:00Z')}[g]
    emb=pq.read_table(f'data/13_downstream/ref-2026-10-09/{g}.parquet',columns=['client_id','n_events','n_tokens']).to_pandas().set_index('client_id')
    f=pq.ParquetFile(f'data/02_preprocessed/{g}/events.parquet')
    counts={}; first_kept={}
    times_by={}
    for i in range(f.num_row_groups):
        t=f.read_row_group(i,columns=['client_id','event_time'])
        t=t.filter(pc.less(t.column('event_time'), pa.scalar(T.to_pydatetime(), pa.timestamp('us',tz='UTC'))))
        df=t.to_pandas()
        for cid, s in df.groupby('client_id', sort=False)['event_time']:
            times_by.setdefault(cid, []).append(s.values)
    res=[]
    for cid, parts in times_by.items():
        tt=np.sort(np.concatenate(parts))
        n=len(tt)
        k=int(emb.n_events.get(cid, 0))
        first = tt[n-k] if k>0 else None
        res.append((cid,n,k,first, tt[0]))
    r=pd.DataFrame(res,columns=['client_id','n_before_T','n_model','first_kept','first_all']).set_index('client_id')
    r['excluded']=r.n_before_T-r.n_model
    r['kept_days']=(np.datetime64(T.tz_convert(None).to_datetime64())-r.first_kept.values.astype('datetime64[ns]'))/np.timedelta64(1,'D')
    rr=rows[rows.group==g].set_index('client_id').join(r, how='left')
    print('==',g,'T',T,'clients in 02 before T',len(r),'emb clients',len(emb))
    print('all clients: truncated',(r.excluded>0).sum(),'excluded events',int(r.excluded.sum()),'of',int(r.n_before_T.sum()), 'mismatch neg', (r.excluded<0).sum())
    tr=rr[rr.excluded>0]
    print('churn rows',len(rr),'truncated',len(tr),'share',round(len(tr)/len(rr),4),'pos rows',int(rr.churn.sum()),'truncated pos',int(tr.churn.sum()))
    print('kept_days among truncated churn rows: min %.1f p10 %.1f median %.1f'%(tr.kept_days.min(), tr.kept_days.quantile(.1), tr.kept_days.median()))
    print('excluded share of history among truncated: median %.3f max %.3f'%((tr.excluded/tr.n_before_T).median(), (tr.excluded/tr.n_before_T).max()))
    print('churn rate truncated %.4f vs not %.4f'%(tr.churn.mean(), rr[~(rr.excluded>0)].churn.mean()))
    r.to_parquet(f'/tmp/claude-1000/-home-darkhan-projects-mini-pragma-v2/ad0b697f-bc2f-4f6e-b37d-1776eb7ba124/scratchpad/pretrain-audit/trunc_{g}.parquet')
