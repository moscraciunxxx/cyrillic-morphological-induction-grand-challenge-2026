import numpy as np
import pandas as pd

from paradigm_model import build_indexes, build_rules, clean_train, predict_one, score


def main():
    tr = pd.read_csv("data/train.csv").fillna("")
    rng = np.random.default_rng(20260925)
    canon = tr[((tr.pos == "N") & (tr.feats == "N;ACC;SG") | (tr.pos == "V") & (tr.feats == "V;PRS;3;SG")) & (tr.dialect == "SEV")]
    seed_rows = canon.groupby("lemma_ru", sort=False).sample(n=1, random_state=7)
    hold_lemmas = set(seed_rows.lemma_ru)
    # Randomly evaluate about half of eligible seed-bearing lemmas to keep the run quick.
    held = set(rng.choice(sorted(hold_lemmas), size=max(1, len(hold_lemmas) // 2), replace=False))
    seeds = seed_rows[seed_rows.lemma_ru.isin(held)].copy()
    val = tr[tr.lemma_ru.isin(held)].copy()
    seed_keys = set(zip(seeds.lemma_ru, seeds.pos, seeds.feats, seeds.dialect))
    val = val[[tuple(x) not in seed_keys for x in val[["lemma_ru", "pos", "feats", "dialect"]].itertuples(index=False, name=None)]]
    model = tr[~tr.lemma_ru.isin(held)].copy()
    clean, sim = clean_train(model)
    print("held-out", len(held), "seed rows", len(seeds), "validation rows", len(val), flush=True)
    rules, stress_rules = build_rules(clean)
    augmented = pd.concat([clean, seeds[["lemma_ru", "pos", "feats", "dialect", "yat_flag", "form_vvz"]]], ignore_index=True)
    modal, lemma_rows = build_indexes(augmented)
    preds = []
    for row in val.itertuples(index=False):
        d, a, c, f = predict_one(row, modal, lemma_rows, rules, stress_rules)
        preds.append(d[0] if d is not None else a if a is not None else f if f is not None else str(row.lemma_ru))
    print("wug-style unseen validation", score(val.form_vvz.values, preds), flush=True)
    for pos in sorted(val.pos.unique()):
        idx = val.pos.values == pos
        print(pos, int(idx.sum()), score(val.form_vvz.values[idx], np.asarray(preds)[idx]), flush=True)


if __name__ == "__main__":
    main()
