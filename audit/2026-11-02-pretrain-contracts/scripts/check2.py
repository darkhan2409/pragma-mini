import sys, json, hashlib
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
import torch
D='/home/darkhan/projects/mini_pragma_v2/data'
s=torch.load(f'{D}/runs/ref-2026-10-09/best_checkpoint.pt', map_location='cpu', weights_only=True)
print(sorted(s.keys()))
print('data', json.dumps(s.get('data'), indent=1))
cfg=s.get('config')
print('config keys', sorted(cfg.keys()) if isinstance(cfg, dict) else type(cfg))
for k in ('masking','seed','epochs','label_smoothing','token_budget'):
    if isinstance(cfg, dict) and k in cfg: print(k, cfg[k])
print('epoch', s.get('epoch'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        while c:=f.read(1<<24): h.update(c)
    return h.hexdigest()
for g in ('train','val'):
    print(g, sha(f'{D}/05_dataset/{g}/samples.parquet'))
print('vw', sha(f'{D}/03_vocab/value_weights.json'))
b=s.get('backbone')
print('backbone vocabulary', b.get('vocabulary') if b else None)
