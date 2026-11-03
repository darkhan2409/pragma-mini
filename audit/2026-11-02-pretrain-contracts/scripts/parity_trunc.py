import json, collections, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
S='/tmp/claude-1000/-home-darkhan-projects-mini-pragma-v2/ad0b697f-bc2f-4f6e-b37d-1776eb7ba124/scratchpad/pretrain-audit/'
r=pd.read_parquet(S+'trunc_train.parquet')
tr=r[r.excluded>0].sort_values('excluded')
pick=[tr.index[0], tr.index[len(tr)//2], tr.index[-1]]
vs=pa.array(pick)
vocab=json.load(open('data/03_vocab/final_vocab.json')); name={v:k for k,v in vocab.items()}
ET=vocab['key:event_type']
def scan(path, cols):
    f=pq.ParquetFile(path); out=[]
    for i in range(f.num_row_groups):
        c=f.read_row_group(i,columns=['client_id']).column('client_id')
        m=pc.is_in(c,value_set=vs)
        if pc.any(m).as_py():
            out.append(f.read_row_group(i,columns=cols).filter(m))
    return pa.concat_tables(out).to_pandas()
pre=scan('data/02_preprocessed/train/events.parquet',['client_id','type','event_time'])
smp=scan('data/05_dataset/train/samples.parquet',['client_id','key_ids','value_ids','event_starts','event_time'])
for cid in pick:
    p=pre[pre.client_id==cid].reset_index(drop=True); sm=smp[smp.client_id==cid].iloc[0]
    k=np.asarray(sm.key_ids); v=np.asarray(sm.value_ids)
    st=[name[x].split('=',1)[1] for x in v[k==ET]]
    n=len(p); m=len(st)
    tail_ok = list(p.type.iloc[n-m:])==st and (pd.to_datetime(pd.Series(sm.event_time),utc=True).values==p.event_time.iloc[n-m:].values).all()
    lost=collections.Counter(p.type.iloc[:n-m])
    print(cid,'pre',n,'ds',m,'excluded',n-m,'tail exact',tail_ok,'first kept',p.event_time.iloc[n-m],'tokens',len(k))
    print('   lost types top:',lost.most_common(6))
    print('   types only in lost part:',sorted(set(lost)-set(st)))
