import json, collections, pyarrow.parquet as pq, pyarrow.compute as pc
v=json.load(open('data/03_vocab/final_vocab.json'))
known={k.split('=',1)[1] for k in v if k.startswith('value:event_type=')}
for g in ['train','val','test']:
    f=pq.ParquetFile(f'data/02_preprocessed/{g}/events.parquet')
    c=collections.Counter()
    for i in range(f.num_row_groups):
        vc=pc.value_counts(f.read_row_group(i,columns=['type']).column('type'))
        for x in vc.to_pylist(): c[x['values']]+=x['counts']
    print(g, sum(c.values()), 'types',len(c),'unknown types:',{k:n for k,n in c.items() if k not in known})
