import sys, json, math, itertools
from collections import defaultdict, Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import pyarrow.parquet as pq
import numpy as np
D='/home/darkhan/projects/mini_pragma_v2/data'
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
fv=json.load(open(f'{D}/03_vocab/final_vocab.json')); tname={v:k for k,v in fv.items()}
f=pq.ParquetFile(f'{D}/05_dataset/val/samples.parquet')
pairs=defaultdict(Counter)   # (A,B) -> Counter((a,b))
cntA=defaultdict(Counter)
keysets=defaultdict(Counter)  # frozenset(keys) -> Counter(event_type)
nev=0
for rg in range(0,6):
    t=f.read_row_group(rg, columns=['key_ids','value_ids','positions','event_starts','event_lengths','target_event_mask']).to_pylist()
    for row in t:
        K=row['key_ids']; V=row['value_ids']; P=row['positions']
        for s,l,m in zip(row['event_starts'],row['event_lengths'],row['target_event_mask']):
            if not m: continue
            nev+=1
            vals={}
            i=s+1
            while i<s+l:
                j=i+1
                while j<s+l and P[j]!=0: j+=1
                vals[kname[K[i]]]=tuple(V[i:j])
                i=j
            et=vals.get('event_type')
            keysets[frozenset(k for k in vals if k!='event_type')][et]+=1
            ks=sorted(vals)
            for a in ks:
                for b in ks:
                    if a!=b:
                        pairs[(a,b)][(vals[a],vals[b])]+=1
def condH(c):
    byA=defaultdict(Counter)
    for (a,b),n in c.items(): byA[a][b]+=n
    tot=sum(c.values()); h=0
    for a,cb in byA.items():
        na=sum(cb.values())
        for n in cb.values(): h-= n/tot*math.log(n/na)
    return h, tot
def H(c):
    tot=sum(c.values()); return -sum(n/tot*math.log(n/tot) for n in c.values())
print('target events', nev)
res=[]
for (a,b),c in pairs.items():
    hb=H(Counter({bb:n for (aa,bb),n in c.items()} ) if False else None) if False else None
    margB=Counter()
    for (aa,bb),n in c.items(): margB[bb]+=n
    h,tot=condH(c)
    HB=H(margB)
    if tot>2000 and HB>0.3 and h<0.05*HB:
        res.append((b,a,round(HB,3),round(h,4),tot))
res.sort()
print('B fully determined by A in same event (H(B)>0.3 nats, H(B|A)<5% of H(B)):')
for r in res: print('  predict', r[0], 'from', r[1], 'H(B)=',r[2],'H(B|A)=',r[3],'co-occ',r[4])
# event_type given key set
tot=sum(sum(c.values()) for c in keysets.values())
h=0
for ks,c in keysets.items():
    n=sum(c.values())
    for m in c.values(): h-=m/tot*math.log(m/n)
allet=Counter()
for c in keysets.values(): allet.update(c)
print('event_type: H=',round(H(allet),3),'H(event_type | visible key set)=',round(h,4), 'distinct key sets', len(keysets))
