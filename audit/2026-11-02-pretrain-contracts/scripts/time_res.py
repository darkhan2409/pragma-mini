import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc
f=pq.ParquetFile('data/05_dataset/val/samples.parquet')
T=np.datetime64('2026-04-30T19:00:00','us')
tot=0; same_ts=0; within60=0; dup_f32=0; dup_f32_distinct_ts=0; within1=0
by_age=[]
for i in range(0,f.num_row_groups,4):
    t=f.read_row_group(i,columns=['event_time'])
    for et in t.column('event_time').to_pylist():
        if not et: continue
        a=np.array([np.datetime64(x.replace(tzinfo=None),'us') for x in et])
        d=(T-a).astype('timedelta64[us]').astype(np.int64)/1e6
        tl=(8*np.log1p(d/8)).astype(np.float32)
        gap=np.diff(a).astype('timedelta64[us]').astype(np.int64)/1e6
        tot+=len(a)-1; same_ts+=(gap==0).sum(); within60+=((gap>0)&(gap<=60)).sum(); within1+=((gap>0)&(gap<=1)).sum()
        eq=np.diff(tl)==0
        dup_f32+=eq.sum(); dup_f32_distinct_ts+=(eq&(gap>0)).sum()
        by_age.append(pd.DataFrame({'age_days':d[1:]/86400,'collapsed':eq&(gap>0),'gap':gap}))
b=pd.concat(by_age)
print('adjacent pairs',tot,'same timestamp',same_ts,round(same_ts/tot,4),'0<gap<=1s',within1,'0<gap<=60s',within60,round(within60/tot,4))
print('float32 time_log equal for distinct timestamps',dup_f32_distinct_ts, round(dup_f32_distinct_ts/tot,5))
b['bin']=pd.cut(b.age_days,[0,1,7,30,90,365,900])
print(b.groupby('bin',observed=True).agg(pairs=('collapsed','size'),collapsed=('collapsed','sum'),max_collapsed_gap=('gap',lambda s: s[b.loc[s.index,'collapsed']].max() if False else np.nan)))
c=b[b.collapsed]; print('collapsed gaps seconds: max',c.gap.max(),'median',c.gap.median(), 'min age days', c.age_days.min())
