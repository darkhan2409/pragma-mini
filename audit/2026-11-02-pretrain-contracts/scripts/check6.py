import sys, json
from collections import Counter, defaultdict
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import pyarrow.parquet as pq
D='/home/darkhan/projects/mini_pragma_v2/data'
fv=json.load(open(f'{D}/03_vocab/final_vocab.json')); tname={v:k for k,v in fv.items()}
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
f=pq.ParquetFile(f'{D}/05_dataset/val/samples.parquet')
same_time=Counter(); shared=Counter(); n_target=0; n_with_twin=0; twin_keys=Counter()
for rg in range(6):
    rows=f.read_row_group(rg, columns=['key_ids','value_ids','positions','event_starts','event_lengths','event_time','target_event_mask']).to_pylist()
    for r in rows:
        K,V,P=r['key_ids'],r['value_ids'],r['positions']
        evs=[]
        for s,l,t,m in zip(r['event_starts'],r['event_lengths'],r['event_time'],r['target_event_mask']):
            vals={}; i=s+1
            while i<s+l:
                j=i+1
                while j<s+l and P[j]!=0: j+=1
                vals[kname[K[i]]]=tuple(V[i:j]); i=j
            evs.append((t,m,vals))
        bytime=defaultdict(list)
        for idx,(t,m,vals) in enumerate(evs): bytime[t].append(idx)
        for t,idxs in bytime.items():
            if len(idxs)<2: continue
            for a in idxs:
                ta,ma,va=evs[a]
                if not ma: continue
                for b in idxs:
                    if a==b: continue
                    vb=evs[b][2]
                    common=[k for k in va if k!='event_type' and k in vb and va[k]==vb[k]]
                    same_time[(tname[va['event_type'][0]].split('=')[1], tname[vb['event_type'][0]].split('=')[1])]+=1
                    if common:
                        shared[(tname[va['event_type'][0]].split('=')[1], tname[vb['event_type'][0]].split('=')[1], tuple(sorted(common)))]+=1
        for t,m,vals in evs:
            if m: n_target+=1
print('target events', n_target)
print('top same-timestamp pairs (target, other):')
for k,n in same_time.most_common(15): print('  ',k,n)
print('top pairs sharing equal values:')
for k,n in shared.most_common(15): print('  ',k,n)
