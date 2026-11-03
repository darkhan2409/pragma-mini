import sys, json
from collections import Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, torch
from src.masking.settings import MaskingConfig
from src.masking.choose import choose
from src.masking.weights import load_value_weights
from src.mlm.train import for_epoch
from src.temporal.samples import TemporalGroup
from src.mlm.inputs import INPUT_COLUMNS, _rows
D='/home/darkhan/projects/mini_pragma_v2/data'
kv=json.load(open(f'{D}/03_vocab/key_vocab.json')); kname={v:k for k,v in kv.items()}
m=MaskingConfig.from_dict(torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)['masking'])
w=load_value_weights(); g=TemporalGroup('train')
spoiled=Counter(); clients=0; noneligible=0
for e in range(3):
    me=for_epoch(m,e)
    for rg in range(0, g.count, 6):
        for row in _rows(g.row_group(rg, columns=INPUT_COLUMNS)):
            if e==0:
                clients+=1; noneligible+=int((~np.asarray(row['target_event_mask'])).sum())
            s=choose('train', row, me, w)
            for v in s.corrupted: spoiled[kname[v.key_id]]+=1
print('train clients sampled', clients, 'non-eligible events', noneligible)
print('context-corrupted values on train over 3 epochs (README says none):', sum(spoiled.values()), dict(spoiled.most_common(8)))
