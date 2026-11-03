import sys, json, collections, numpy as np, pandas as pd
from datetime import datetime, timezone
sys.path.insert(0,'.')
from src.preprocessing.read import Group
from src.tokenization.finalvocab import FrozenArtifacts
from src.tokenization.encode import encode_profile
from src.tokenization.settings import TokenizerConfig
g=sys.argv[1]; T={'train':datetime(2025,12,31,19,tzinfo=timezone.utc),'val':datetime(2026,2,28,19,tzinfo=timezone.utc)}[g]
feat=pd.read_parquet(f'churn_baseline/data/{g}/features.parquet').set_index('client_id')
rows=pd.read_parquet('churn_baseline/reports/'+('train' if g=='train' else 'eval')+'_rows.parquet').client_id
ids=sorted(rows)[::max(1,len(rows)//300)]
grp=Group(g); art=FrozenArtifacts.load(); limit=TokenizerConfig.load(None).max_pieces_per_value
name=art.name_of
mism=collections.Counter(); unk=collections.Counter(); n=0; notes=collections.Counter(); neg_tenure=0; lif_dupe=0; lif_diff=collections.Counter()
CAT=['gender','family_status','education','region','city','housing_type','income_type','industry']
b=json.load(open('data/03_vocab/buckets.json'))['profile_declared_income']
for cid in ids:
    h=grp.history(cid,T); n+=1
    for x in h.limitations: notes[x.split(':')[0]]+=1
    rec,times=encode_profile(art,h,limit)
    toks={}
    for k,v,p in zip(rec.key_ids[1:],rec.value_ids[1:],rec.positions[1:]):
        kn=name[k][4:]; vn=name[v]
        if vn=='[UNK]': unk[kn]+=1
        toks.setdefault(kn,[]).append(vn.split('=',1)[1] if '=' in vn else vn)
    f=feat.loc[cid]
    for c in CAT:
        pv=toks.get('profile_'+c,[None])[0]; bv=f[c]
        if (pv if pv is not None else 'NA')!=bv: mism[c]+=1
    for c in ['age','children','income_day']:
        pv=toks.get('profile_'+c,[None])[0]; bv=f[c]
        if not ((pv is None and pd.isna(bv)) or (pv is not None and not pd.isna(bv) and int(pv)==int(bv))): mism[c]+=1
    # declared income bucket
    pv=toks.get('profile_declared_income',[None])[0]; bv=f['declared_income']
    if pv is None or pd.isna(bv):
        if not (pv is None and pd.isna(bv)): mism['declared_income']+=1
    else:
        bk=b[pv.replace('bucket:','')]; lo=bk['min'] if bk['min'] is not None else -1e18; hi=bk['max'] if bk['max'] is not None else 1e18
        ok = (bv==0) if (bk['min']==0 and bk['max']==0) else (lo<=bv<hi)
        if not ok: mism['declared_income']+=1
    pv=toks.get('profile_job_tenure_months',[None])[0]; bv=f['job_tenure_months']
    if pv is None or pd.isna(bv):
        if not (pv is None and pd.isna(bv)): mism['job_tenure_months']+=1
    else:
        lo,hi=pv.rsplit('-',1) if not pv.startswith('-') else (None,None)
        if lo is None or not (int(lo)<=bv<=int(hi)): mism['job_tenure_months']+=1
        if pv.startswith('-'): neg_tenure+=1
    types=[t for t,_ in h.lifelong]
    if len(types)!=len(set(types)): lif_dupe+=1
    for kind in ['bank_registered','app_registered','first_card_activated','first_loan_opened','first_deposit_opened']:
        ms=[m for t,m in h.lifelong if t==kind]; bv=f['days_since_'+kind]
        if (not ms) != pd.isna(bv): lif_diff[kind]+=1
        elif ms and abs((T-min(ms)).total_seconds()/86400-bv)>1e-6: lif_diff[kind]+=1
print(g,'clients',n,'mismatches',dict(mism),'UNK in profile',dict(unk),'notes',dict(notes),'neg tenure',neg_tenure,'dup lifelong types',lif_dupe,'lifelong diffs',dict(lif_diff))
print('profile keys example',sorted(toks))
