import sys, json, math
from collections import defaultdict, Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import pyarrow.parquet as pq
D='/home/darkhan/projects/mini_pragma_v2/data'
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
f=pq.ParquetFile(f'{D}/05_dataset/train/samples.parquet')
want=[('direction','transaction_amount'),('direction','event_type'),('app_domain','firebase_screen'),('app_domain','operation'),
      ('mcc','merchant_category'),('merchant_category','mcc'),('mcc','merchant_name'),('merchant_category','merchant_name'),
      ('campaign_code','offer'),('offer','campaign_code'),('purpose','template'),('campaign_code','template'),('is_online','operation_channel'),
      ('event_type','__keyset__'),('merchant_city','merchant_name'),('merchant_country','merchant_name')]
joint={p:Counter() for p in want}
for rg in range(0,8):
    for r in f.read_row_group(rg, columns=['key_ids','value_ids','positions','event_starts','event_lengths']).to_pylist():
        K,V,P=r['key_ids'],r['value_ids'],r['positions']
        for s,l in zip(r['event_starts'],r['event_lengths']):
            vals={}; i=s+1
            while i<s+l:
                j=i+1
                while j<s+l and P[j]!=0: j+=1
                vals[kname[K[i]]]=tuple(V[i:j]); i=j
            vals['__keyset__']=tuple(sorted(k for k in vals if k!='event_type'))
            for b,a in want:
                if a in vals and b in vals: joint[(b,a)][(vals[a],vals[b])]+=1
def H(c):
    t=sum(c.values()); return -sum(n/t*math.log(n/t) for n in c.values())
for (b,a),c in joint.items():
    mb=Counter(); byA=defaultdict(Counter)
    for (va,vb),n in c.items(): mb[vb]+=n; byA[va][vb]+=n
    t=sum(c.values()); hc=0
    for va,cb in byA.items():
        na=sum(cb.values()); hc-=sum(n/t*math.log(n/na) for n in cb.values())
    hb=H(mb)
    print(f'{b:18s} | {a:18s} n={t:7d} H(B)={hb:.3f} H(B|A)={hc:.4f} ratio={hc/hb if hb else 0:.3f}')
