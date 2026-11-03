import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa, collections, json
from tokenizers import Tokenizer
def counts(g, col):
    f=pq.ParquetFile(f'data/02_preprocessed/{g}/events.parquet'); c=collections.Counter()
    for i in range(f.num_row_groups):
        vc=pc.value_counts(pc.drop_null(f.read_row_group(i,columns=[col]).column(col)))
        for x in vc.to_pylist(): c[x['values']]+=x['counts']
    return c
tok=Tokenizer.from_file('data/03_vocab/bpe.json')
for col in ['merchant_name','counterparty']:
    tr=counts('train',col)
    for g in ['train','val','test']:
        c=tr if g=='train' else counts(g,col)
        tot=sum(c.values()); new=sum(n for k,n in c.items() if k not in tr)
        sample=list(c.items())
        pieces=sum(len(tok.encode(k).tokens)*n for k,n in sample)/tot
        print(f'{col} {g}: values {tot}, distinct {len(c)}, values with string unseen in train {new} ({new/tot:.4%}), distinct unseen {sum(1 for k in c if k not in tr)}, mean BPE pieces {pieces:.2f}')
