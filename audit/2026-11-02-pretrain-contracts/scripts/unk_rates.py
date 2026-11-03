import json, collections, numpy as np, pyarrow.parquet as pq, pyarrow.compute as pc
v=json.load(open('data/03_vocab/final_vocab.json')); name={i:k for k,i in v.items()}
UNK=v['[UNK]']; EVT=v['[EVT]']
byte=np.zeros(len(v),bool)
for k,i in v.items():
    if k.startswith('bpe:<0x'): byte[i]=True
res={}
for g in ['train','val','test']:
    f=pq.ParquetFile(f'data/04_tokenized/{g}/events.parquet')
    n=f.num_row_groups
    rgs=np.linspace(0,n-1,min(n,400)).astype(int)
    vals=collections.Counter(); unk=collections.Counter(); bytev=collections.Counter(); tok=0; unk_tok=0
    for i in np.unique(rgs):
        t=f.read_row_group(int(i),columns=['key_ids','value_ids','positions'])
        k=pc.list_flatten(t.column('key_ids')).to_numpy(); vv=pc.list_flatten(t.column('value_ids')).to_numpy(); p=pc.list_flatten(t.column('positions')).to_numpy()
        # value index: start at position 0 and not marker
        start=(p==0)&(k!=vv)
        vid=np.cumsum(start)-1
        content=k!=vv
        tok+=content.sum(); unk_tok+=(vv[content]==UNK).sum()
        ks=k[start]; isunk=np.zeros(start.sum(),bool)
        np.logical_or.at(isunk, vid[content], vv[content]==UNK)
        hasbyte=np.zeros(start.sum(),bool); np.logical_or.at(hasbyte, vid[content], byte[vv[content]])
        for key,u,b in zip(ks,isunk,hasbyte):
            vals[key]+=1; unk[key]+=u; bytev[key]+=b
    res[g]=(vals,unk,bytev)
    print(f'== {g}: clients sampled {len(np.unique(rgs))}, content tokens {tok}, UNK tokens {unk_tok} ({unk_tok/tok:.5%}); values {sum(vals.values())}, UNK values {sum(unk.values())} ({sum(unk.values())/sum(vals.values()):.5%})')
    rows=[(name[k],vals[k],unk[k],unk[k]/vals[k],bytev[k]/vals[k]) for k in vals if unk[k] or bytev[k]]
    for r in sorted(rows,key=lambda r:-r[2])[:20]: print('   %-34s values %8d unk %6d (%.4f) byte-fallback %.4f'%r)
