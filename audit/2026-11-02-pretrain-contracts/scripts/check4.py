import sys, json
from collections import Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, torch
from dataclasses import replace
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, IGNORE
from src.mlm.train import for_epoch
D='/home/darkhan/projects/mini_pragma_v2/data'
s=torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)
print('ckpt masking', s['masking'])
m=MaskingConfig.from_dict(s['masking'])
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
a=Source('val', masking=m); b=Source('val', masking=m)
RG=[0,1,2,3]
same=True; tot_targets=0; tot_tokens=0
for i in RG:
    for ca,cb in zip(a.batch(i), b.batch(i)):
        same &= np.array_equal(ca.value_ids, cb.value_ids) and np.array_equal(ca.labels, cb.labels)
        # labels -100 exactly where value_ids != MASK
        assert np.array_equal(ca.labels != IGNORE, ca.value_ids == a.mask_id), 'labels/mask mismatch'
        tot_targets += ca.n_targets; tot_tokens += ca.n_tokens
print('val deterministic across instances:', same, 'targets', tot_targets, 'tokens', tot_tokens)
# epoch masks differ on train
t0=Source('train', masking=for_epoch(m,0)); t1=Source('train', masking=for_epoch(m,1)); t0b=Source('train', masking=for_epoch(m,0))
c0=t0.batch(0); c1=t1.batch(0); c0b=t0b.batch(0)
print('train epoch0 == epoch0 again:', all(np.array_equal(x.labels,y.labels) for x,y in zip(c0,c0b)),
      ' epoch0 vs epoch1 differ:', any(not np.array_equal(x.labels,y.labels) for x,y in zip(c0,c1)))
print('train target fraction of tokens (rg0):', sum(c.n_targets for c in c0)/sum(c.n_tokens for c in c0))
# zero configs
z1=MaskingConfig(value_probability=0.0, event_probability=0.0, key_probability=0.0, unknown_probability=0.0)
z2=MaskingConfig(value_probability=0.0, event_probability=0.0, key_probability=0.0, unknown_probability=0.0, informativeness_weighted_masking=False)
import pyarrow.parquet as pq
raw=pq.ParquetFile(f'{D}/05_dataset/val/samples.parquet').read_row_group(0, columns=['value_ids']).column('value_ids').to_pylist()
for name,z in (('diag',z1),('finetune',z2)):
    cs=Source('val', masking=z).batch(0)
    print(name, 'targets', sum(c.n_targets for c in cs), 'value_ids unchanged', all(np.array_equal(c.value_ids, np.asarray(r)) for c,r in zip(cs,raw)))
# reasons distribution and trivially-recoverable targets on val
pairs={'direction':['transaction_amount','event_type'], 'app_domain':['firebase_screen','operation'],
       'campaign_code':['offer','template','product'], 'offer':['campaign_code'], 'purpose':['campaign_code','template'],
       'merchant_category':['merchant_name','mcc'], 'mcc':['merchant_category','merchant_name']}
inv={k:v for k,v in kv.items()}
reasons=Counter(); triv=Counter(); bykey=Counter(); et_t=Counter()
for i in RG:
    for c in a.batch(i):
        lab=c.labels; P=c.positions; K=c.key_ids
        starts=np.flatnonzero((lab!=IGNORE) & (P==0))
        for st in starts:
            r=c.reason[st]; k=kname[int(K[st])]
            reasons[r]+=1; bykey[k]+=1
            if k=='event_type': et_t[r]+=1
            if k in pairs:
                ev=np.searchsorted(c.event_starts, st, side='right')-1
                s0=c.event_starts[ev]; e0=s0+c.event_lengths[ev]
                vis={kname[int(K[j])] for j in range(s0+1,e0) if P[j]==0 and c.value_ids[j] not in (a.mask_id, a.unknown_id)}
                if any(p in vis for p in pairs[k]):
                    triv[(k,r)]+=1
n=sum(reasons.values())
print('val target values', n, 'by reason', dict(reasons))
print('event_type targets', sum(et_t.values()), dict(et_t), 'share of all', round(sum(et_t.values())/n,4))
tv=sum(triv.values())
print('targets with deterministic partner visible in same event:', tv, round(tv/n,4))
for (k,r),c_ in sorted(triv.items(), key=lambda x:-x[1]): print('   ',k,r,c_)
print('top target keys', bykey.most_common(12))
