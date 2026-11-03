import sys
import numpy as np
sys.path.insert(0, "/home/darkhan/projects/mini_pragma_v2")
from src.mlm.inputs import Source, micro_batches
src = Source("train")
sizes = list(src.sizes())
w, tok, B = [], [], []
for batch in micro_batches(sizes, 48000):
    for s in batch:
        w.append(1.0 / len(batch)); tok.append(s.n_tokens)
    B.append(len(batch))
w, tok, B = map(np.asarray, (w, tok, B))
print("micro-batches", B.size, "clients", tok.size, "clients per micro-batch: mean %.2f median %d max %d" % (B.mean(), np.median(B), B.max()))
q = np.quantile(tok, [0, .25, .5, .75, 1.0])
for lo, hi in zip(q[:-1], q[1:]):
    m = (tok >= lo) & (tok <= hi)
    print("tokens [%d, %d]: mean aux weight per client %.3f, MLM share of targets ~ tokens share %.3f" % (lo, hi, w[m].mean(), tok[m].sum() / tok.sum()))
print("share of the epoch aux weight carried by the top quartile by tokens: %.3f" % (w[tok >= q[3]].sum() / w.sum()))
