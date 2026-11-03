import sys, json
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import numpy as np, torch
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.specials import load_special_tokens
from src.masking.settings import MaskingConfig
from src.mlm.inputs import Source, IGNORE
from src.mlm.train import for_epoch
D='/home/darkhan/projects/mini_pragma_v2/data'
fv=json.load(open(f'{D}/03_vocab/final_vocab.json'))
sp=load_special_tokens(); a=FrozenArtifacts.load()
print('specials file', sp, 'final_vocab', {k:fv[k] for k in sp}, 'artifacts', {k:a.special(k) for k in sp})
kv=json.load(open(f'{D}/03_vocab/key_vocab.json'))
print('key id range', min(kv.values()), max(kv.values()), 'all key names in final vocab with same id:', all(fv.get('key:'+k)==v for k,v in kv.items()))
m=MaskingConfig.from_dict(torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)['masking'])
for name,src in (('val',Source('val',masking=m)),('train-e0',Source('train',masking=for_epoch(m,0)))):
    labs=[]; unk_src=0; ev_k_in_val=0
    for i in range(4):
        for c in src.batch(i):
            l=c.labels[c.labels!=IGNORE]; labs.append(l)
            # key slot never touched: value_ids at marker positions equal key_ids
            mk=np.isin(c.key_ids,[3,4]); assert np.array_equal(c.key_ids[mk], c.value_ids[mk])
            # values never contain key ids
            ev_k_in_val += int(np.isin(c.value_ids, list(kv.values())).sum())
            # profile untouched (no MASK/UNK injected) 
            assert not np.isin(c.profile_value_ids,[2]).any()
    L=np.concatenate(labs)
    print(name, 'labels', L.size, 'in specials', int((L<5).sum()), '== [UNK]', int((L==1).sum()), 'in key range', int(((L>=5)&(L<=106)).sum()), 'key ids in value slot', ev_k_in_val)
