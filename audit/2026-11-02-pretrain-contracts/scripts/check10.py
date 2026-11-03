import sys, json, math
from collections import defaultdict, Counter
import pyarrow.parquet as pq
D='/home/darkhan/projects/mini_pragma_v2/data'
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
f=pq.ParquetFile(f'{D}/05_dataset/train/samples.parquet')
c={'merchant_name':Counter(),'counterparty':Counter()}
for rg in range(8):
    for r in f.read_row_group(rg, columns=['key_ids','value_ids','positions']).to_pylist():
        K,V,P=r['key_ids'],r['value_ids'],r['positions']
        n=len(K); i=0
        while i<n:
            k=kname.get(K[i])
            if k in c and P[i]==0:
                j=i+1
                while j<n and P[j]!=0: j+=1
                c[k][(j-i, tuple(V[i:j]))]+=1; i=j
            else: i+=1
def H(cn):
    t=sum(cn.values()); return -sum(x/t*math.log(x/t) for x in cn.values())
for k,cn in c.items():
    marg=Counter(); byl=defaultdict(Counter)
    for (l,v),x in cn.items(): marg[v]+=x; byl[l][v]+=x
    t=sum(cn.values()); hc=sum(sum(b.values())/t*H(b) for b in byl.values())
    print(k, 'n', t, 'H(value)=%.3f H(value|n_pieces)=%.3f  lengths'%(H(marg),hc), sorted(Counter({l:sum(b.values()) for l,b in byl.items()}).items())[:8])
