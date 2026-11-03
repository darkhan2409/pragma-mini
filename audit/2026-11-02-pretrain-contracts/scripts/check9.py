import sys, json
from collections import Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, torch
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, IGNORE
from src.mlm.train import for_epoch
D='/home/darkhan/projects/mini_pragma_v2/data'
m=MaskingConfig.from_dict(torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)['masking'])
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
strict={'direction':['transaction_amount','event_type'],'app_domain':['firebase_screen','operation'],
        'campaign_code':['offer','template'],'offer':['campaign_code'],'purpose':['template','campaign_code'],'is_online':['operation_channel']}
near={'mcc':['merchant_category'],'merchant_category':['mcc','merchant_name']}
for name,src in (('val',Source('val',masking=m)),('train-e0',Source('train',masking=for_epoch(m,0)))):
    tiers=Counter(); n=0; tok_t=0; tok_triv=0
    for i in range(4):
        for c in src.batch(i):
            lab=c.labels; P=c.positions; K=c.key_ids; Vv=c.value_ids
            for st in np.flatnonzero((lab!=IGNORE)&(P==0)):
                n+=1; k=kname[int(K[st])]
                ev=np.searchsorted(c.event_starts, st, side='right')-1
                s0=c.event_starts[ev]; e0=s0+c.event_lengths[ev]
                vis={kname[int(K[j])] for j in range(s0+1,e0) if P[j]==0 and Vv[j] not in (src.mask_id, src.unknown_id)}
                if k=='event_type': tiers['event_type<-keyset']+=1
                elif k in strict and any(p in vis for p in strict[k]): tiers['strict(<3%)']+=1
                elif k in near and any(p in vis for p in near[k]): tiers['near(~10%)']+=1
    print(name, 'target values', n, {t:(v, round(v/n,3)) for t,v in tiers.items()}, 'sum', round(sum(tiers.values())/n,3))
