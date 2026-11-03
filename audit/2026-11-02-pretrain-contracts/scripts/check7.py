import sys, json
from collections import defaultdict
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, pyarrow.parquet as pq
from src.masking.settings import MaskingConfig
from src.masking.weights import load_value_weights
from src.masking.choose import _parse
D='/home/darkhan/projects/mini_pragma_v2/data'
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
w=load_value_weights(); m=MaskingConfig()
known={}
for k,item in json.load(open(f'{D}/03_vocab/value_weights.json'))['keys'].items():
    known[item['key_id']]=set(tuple(r['tokens']) for r in item['values'])
for g in ('train','val'):
    f=pq.ParquetFile(f'{D}/05_dataset/{g}/samples.parquet')
    sums=defaultdict(float); cnt=defaultdict(int); unseen=defaultdict(int)
    for rg in range(0,8):
        for row in f.read_row_group(rg, columns=['key_ids','value_ids','positions','event_starts','event_lengths','target_event_mask']).to_pylist():
            vals=_parse(row)
            mask=np.asarray(row['target_event_mask'])[vals.event]
            vals=vals.take(mask)
            V=np.asarray(row['value_ids'],dtype=np.int64)
            p=w.probabilities(vals.key_id, vals.start, vals.length, V, m)
            for k,s,l,pp in zip(vals.key_id.tolist(), vals.start.tolist(), vals.length.tolist(), p.tolist()):
                sums[k]+=pp; cnt[k]+=1
                if tuple(V[s:s+l].tolist()) not in known.get(k,()): unseen[k]+=1
    tot=sum(cnt.values())
    print('==',g,'eligible values',tot,'mean P', round(sum(sums.values())/tot,4))
    for k in sorted(cnt, key=lambda k:-cnt[k])[:40]:
        if kname[k] in ('counterparty','merchant_name','merchant_city','balance_after','transaction_amount','firebase_screen','app_domain','event_type','mcc','merchant_category','direction','offer','campaign_code','template'):
            print(f'   {kname[k]:22s} share {cnt[k]/tot:.3f} meanP {sums[k]/cnt[k]:.3f} unseen {unseen[k]/cnt[k]:.3f}')
