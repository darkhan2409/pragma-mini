import sys
from collections import Counter
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, torch
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, IGNORE
from src.mlm.train import for_epoch
D='/home/darkhan/projects/mini_pragma_v2/data'
m=MaskingConfig.from_dict(torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)['masking'])
for name,src in (('val',Source('val',masking=m)),('train-e0',Source('train',masking=for_epoch(m,0)))):
    r=Counter(); toks=0; unk=0
    for i in range(2):
        for c in src.batch(i):
            st=(c.labels!=IGNORE)&(c.positions==0)
            for x in np.asarray(c.reason, dtype=object)[st]: r[x]+=1
            toks+=c.n_tokens; unk+=int((c.value_ids==src.unknown_id).sum())
    n=sum(r.values())
    print(name, 'target values', n, {k:round(v/n,3) for k,v in r.items()}, 'UNK share of tokens', round(unk/toks,4))
