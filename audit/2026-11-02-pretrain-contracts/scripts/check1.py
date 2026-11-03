import json, hashlib, sys
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
sys.path.insert(0, '/home/darkhan/projects/mini_pragma_v2')
from src.tokenization.finalvocab import vocabulary_digest
D='/home/darkhan/projects/mini_pragma_v2/data'
print('vocab digest now', vocabulary_digest())
w=json.load(open(f'{D}/03_vocab/value_weights.json'))
print('value_weights vocab', w['vocabulary'])
ids={}
for g in ('train','val','test'):
    f=pq.ParquetFile(f'{D}/05_dataset/{g}/samples.parquet')
    t=f.read(columns=['client_id'])
    ids[g]=set(t.column('client_id').to_pylist())
    print(g,'rows',f.metadata.num_rows,'row groups',f.num_row_groups,'unique',len(ids[g]), 'rg sizes', sorted(set(f.metadata.row_group(i).num_rows for i in range(f.num_row_groups))))
    raw=pq.read_table(f'{D}/01_raw/{g}/profile.parquet',columns=['client_id']).column('client_id').to_pylist()
    print('  raw profile clients', len(raw), 'same set as 05', set(raw)==ids[g])
    et=f.read(columns=['event_time']).column('event_time').combine_chunks().flatten()
    print('  event_time min', pc.min(et).as_py(), 'max', pc.max(et).as_py(), 'n', len(et))
    pt=f.read(columns=['profile_time']).column('profile_time').combine_chunks().flatten()
    print('  profile_time min', pc.min(pt).as_py(), 'max', pc.max(pt).as_py())
print('train&val', len(ids['train']&ids['val']), 'train&test', len(ids['train']&ids['test']), 'val&test', len(ids['val']&ids['test']))
