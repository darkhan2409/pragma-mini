import json, numpy as np, pandas as pd, pyarrow.parquet as pq
b=json.load(open('data/03_vocab/buckets.json'))
def edges(when):
    return sorted([v['max'] for k,v in b['transaction_amount'].items() if v.get('when')==when and v['max'] not in (None,0.0)])
f=pq.ParquetFile('data/02_preprocessed/train/events.parquet')
rgs=list(range(0,f.num_row_groups,8))
df=pd.concat([f.read_row_group(i,columns=['type','direction','amount','balance_after']).to_pandas() for i in rgs])
print('rows',len(df), 'rgs',rgs)
for when in ('debit','credit'):
    e=edges(when); top=e[-1]
    d=df[(df.direction==when)&df.amount.notna()]
    print(f'== {when}: bins {len(e)+2} (incl zero) top edge {top}')
    g=d.groupby('type').amount
    out=pd.DataFrame({'n':g.size(),'median':g.median(),'share_top':g.apply(lambda s:(s>=top).mean()),
                      'n_bins_used':g.apply(lambda s: len(np.unique(np.searchsorted(e, s.values, side='right'))))})
    print(out.sort_values('n',ascending=False).head(25).to_string())
nd=df[df.direction.isna()&df.amount.notna()]
print('no_direction', nd.type.value_counts().head())
