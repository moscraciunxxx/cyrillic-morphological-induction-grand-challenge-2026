import numpy as np
import pandas as pd
from paradigm_model import build_indexes, build_rules, clean_train, predict_one, score

tr = pd.read_csv('data/train.csv').fillna('')
rng = np.random.default_rng(20260925)
canon = tr[((tr.pos == 'N') & (tr.feats == 'N;ACC;SG') | (tr.pos == 'V') & (tr.feats == 'V;PRS;3;SG')) & (tr.dialect == 'SEV')]
seed_rows = canon.groupby('lemma_ru', sort=False).sample(n=1, random_state=7)
held = set(rng.choice(sorted(seed_rows.lemma_ru), size=max(1, seed_rows.lemma_ru.nunique() // 2), replace=False))
seeds = seed_rows[seed_rows.lemma_ru.isin(held)].copy()
val = tr[tr.lemma_ru.isin(held)].copy()
keys = set(zip(seeds.lemma_ru, seeds.pos, seeds.feats, seeds.dialect))
val = val[[tuple(x) not in keys for x in val[['lemma_ru','pos','feats','dialect']].itertuples(index=False, name=None)]]
clean, _ = clean_train(tr[~tr.lemma_ru.isin(held)].copy())
print('held',len(held),'val',len(val),flush=True)
rules, stress = build_rules(clean)
aug = pd.concat([clean, seeds[['lemma_ru','pos','feats','dialect','yat_flag','form_vvz']]], ignore_index=True)
modal, lemma_rows = build_indexes(aug)
rows=[]
for row in val.itertuples(index=False):
    d,a,c,f=predict_one(row,modal,lemma_rows,rules,stress)
    rows.append((d[0] if d else None,a,c,f,str(row.lemma_ru),str(row.pos)))
y=val.form_vvz.astype(str).tolist()
policies = {
    'direct_analogy_fallback': lambda d,a,f,lemma: d if d is not None else a if a is not None else f if f is not None else lemma,
    'direct_fallback_analogy': lambda d,a,f,lemma: d if d is not None else f if f is not None else a if a is not None else lemma,
    'analogy_fallback_direct': lambda d,a,f,lemma: a if a is not None else f if f is not None else d if d is not None else lemma,
    'direct_else_lemma': lambda d,a,f,lemma: d if d is not None else lemma,
    'fallback_only': lambda d,a,f,lemma: f if f is not None else lemma,
}
for name, fn in policies.items():
    p=[fn(d,a,f,lemma) for d,a,c,f,lemma,pos in rows]
    print(name,score(y,p),flush=True)
for th in [0,.25,.5,.75,.9,.99,1.01]:
    p=[a if a is not None and c>=th else d if d is not None else a if a is not None else f if f is not None else lemma for d,a,c,f,lemma,pos in rows]
    print('analogy_threshold',th,score(y,p),flush=True)
for pos in sorted(set(x[5] for x in rows)):
    ix=[i for i,x in enumerate(rows) if x[5]==pos]
    pred=[rows[i][0] if rows[i][0] is not None else rows[i][1] if rows[i][1] is not None else rows[i][3] if rows[i][3] is not None else rows[i][4] for i in ix]
    print('POS',pos,len(ix),score([y[i] for i in ix],pred),flush=True)
