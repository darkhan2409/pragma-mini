import json, collections, sys
import pyarrow.parquet as pq
p='data/01_raw/train/events.parquet'
f=pq.ParquetFile(p)
keys=collections.defaultdict(collections.Counter)
src=collections.defaultdict(collections.Counter)
n=0
for rg in [0, 10, 20, 33]:
    t=f.read_row_group(rg, columns=['source','payload'])
    for s,pl in zip(t.column('source').to_pylist(), t.column('payload').to_pylist()):
        d=json.loads(pl); ty=d.get('type')
        src[ty][s]+=1
        for k,v in d.items():
            if v is not None: keys[ty][k]+=1
        n+=1
print('rows',n)
allk=collections.Counter()
for ty,c in keys.items():
    for k in c: allk[k]+=c[k]
pre=set(pq.ParquetFile('data/02_preprocessed/train/events.parquet').schema_arrow.names)
print('payload keys not in 02:', sorted(k for k in allk if k not in pre))
for ty in sorted(keys):
    print(ty, dict(src[ty]), sum(src[ty].values()), sorted(k for k in keys[ty] if k not in pre))
