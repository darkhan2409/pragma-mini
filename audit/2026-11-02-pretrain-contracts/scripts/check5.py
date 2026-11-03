import sys, json
from collections import Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import pyarrow.parquet as pq, numpy as np
D='/home/darkhan/projects/mini_pragma_v2/data'
fv=json.load(open(f'{D}/03_vocab/final_vocab.json')); tname={v:k for k,v in fv.items()}
kv=json.load(open(f'{D}/03_vocab/key_vocab.json'))
LIFE=kv['profile_lifelong']; ET=kv['event_type']
for g in ('train','val'):
    f=pq.ParquetFile(f'{D}/05_dataset/{g}/samples.parquet')
    coinc=Counter(); coinc_t=Counter(); ms=Counter(); ms_in=Counter()
    for rg in range(0, min(f.num_row_groups, 47)):
        rows=f.read_row_group(rg, columns=['key_ids','value_ids','positions','event_starts','event_lengths','event_time','target_event_mask','profile_key_ids','profile_value_ids','profile_time']).to_pylist()
        for r in rows:
            times={}
            for k,v,t in zip(r['profile_key_ids'], r['profile_value_ids'], r['profile_time']):
                if k==LIFE:
                    kind=tname[v].split('=')[1]; ms[kind]+=1
                    if t is not None and t>=r['event_time'][0] if r['event_time'] else False: ms_in[kind]+=1
                    times.setdefault(t, []).append(kind)
            for s,l,t,m in zip(r['event_starts'], r['event_lengths'], r['event_time'], r['target_event_mask']):
                if t in times:
                    et=None
                    for j in range(s+1,s+l):
                        if r['key_ids'][j]==ET and r['positions'][j]==0: et=tname[r['value_ids'][j]]
                    for kind in times[t]:
                        coinc[(kind, et, bool(m))]+=1
    print('==', g, 'milestones', dict(ms), 'inside history', dict(ms_in))
    for (kind,et,m),n in sorted(coinc.items(), key=lambda x:-x[1])[:25]:
        print('  ', kind, et, 'target' if m else 'context', n)
