import json, collections, numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
rng=np.random.default_rng(20261102)
s=pq.ParquetFile('data/05_dataset/train/samples.parquet')
ids=pq.read_table('data/05_dataset/train/samples.parquet',columns=['client_id']).column('client_id').to_pylist()
pick=sorted(rng.choice(ids,20,replace=False).tolist())
vs=pa.array(pick)
vocab=json.load(open('data/03_vocab/final_vocab.json')); name={v:k for k,v in vocab.items()}
ET=vocab['key:event_type']
def scan(path, cols):
    f=pq.ParquetFile(path); out=[]
    for i in range(f.num_row_groups):
        c=f.read_row_group(i,columns=['client_id']).column('client_id')
        m=pc.is_in(c,value_set=vs)
        if pc.any(m).as_py():
            t=f.read_row_group(i,columns=cols).filter(m); out.append(t)
    return pa.concat_tables(out).to_pandas()
raw=scan('data/01_raw/train/events.parquet',['client_id','event_time','payload'])
raw['type']=[json.loads(p)['type'] for p in raw.payload]
pre=scan('data/02_preprocessed/train/events.parquet',['client_id','type','event_time'])
tok=scan('data/04_tokenized/train/events.parquet',['client_id','key_ids','value_ids'])
smp=scan('data/05_dataset/train/samples.parquet',['client_id','key_ids','value_ids','event_starts','event_lengths'])
def types_of(k,v):
    k=np.asarray(k); v=np.asarray(v); return [name[x].split('=',1)[1] for x in v[k==ET]]
tok['type']=[types_of(k,v)[0] if len(types_of(k,v))==1 else ('BAD',len(types_of(k,v))) for k,v in zip(tok.key_ids,tok.value_ids)]
rows=[]
for cid in pick:
    r=raw[raw.client_id==cid]; p=pre[pre.client_id==cid]; t=tok[tok.client_id==cid]; sm=smp[smp.client_id==cid].iloc[0]
    st=types_of(sm.key_ids, sm.value_ids)
    rc=collections.Counter(r.type); pc_=collections.Counter(p.type); tc=collections.Counter(t.type); sc=collections.Counter(st)
    tail=collections.Counter(p.type.iloc[len(p)-len(st):]) if len(st) else collections.Counter()
    rows.append(dict(client=cid, raw=len(r), pre=len(p), tok=len(t), ds=len(sm.event_starts), ds_types=len(st),
        types_raw=len(rc), types_ds=len(sc), raw_eq_pre=rc==pc_, pre_eq_tok=pc_==tc, ds_eq_tail=sc==tail,
        lost_types=sorted(set(rc)-set(sc))))
df=pd.DataFrame(rows); print(df.to_string())
print('sum raw',df.raw.sum(),'pre',df.pre.sum(),'tok',df.tok.sum(),'ds',df.ds.sum())
