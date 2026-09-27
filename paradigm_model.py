import collections
import math
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd


DATA = "data"
FEAT = "feats"
KEY = ["lemma_ru", "pos", FEAT, "dialect"]


def plain(s):
    return str(s).replace("\u0301", "")


def stress_pos(s):
    s = str(s)
    # Stress marks are combining acute accents after the stressed vowel.
    n = 0
    for c in s:
        if c == "\u0301":
            return n - 1
        n += 1
    return -1


def put_stress(s, pos):
    s = plain(s)
    if pos < 0 or pos >= len(s):
        return s
    return s[: pos + 1] + "\u0301" + s[pos + 1 :]


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def clean_train(tr, threshold=0.25):
    # Drop only obvious injected distractors. Most real forms retain the lemma.
    from difflib import SequenceMatcher

    sim = np.fromiter(
        (SequenceMatcher(None, plain(a), plain(b), autojunk=False).ratio()
         for a, b in zip(tr.lemma_ru, tr.form_vvz)),
        dtype=np.float32,
        count=len(tr),
    )
    return tr.loc[sim > threshold].copy(), sim


def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(y, p, weights=None):
    em = np.fromiter((a == b for a, b in zip(y, p)), bool, len(y))
    if weights is None:
        weights = np.ones(len(y), dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weighted_em = float((weights * em).sum() / weights.sum())
    cers = np.fromiter(
        (lev(plain(a), plain(b)) / max(1, len(plain(a))) for a, b in zip(y, p)),
        dtype=np.float32,
        count=len(y),
    )
    weighted_cer = float((weights * cers).sum() / weights.sum())
    return {"wem": weighted_em, "wcer": weighted_cer, "score": .9 * weighted_em + .1 * (1 - weighted_cer),
            "em": float(em.mean()), "cer": float(cers.mean())}


def build_rules(tr):
    # Pairwise paradigm analogies: replace an observed form ending with the
    # target-feature ending learned from other lemmas.
    rule_counts = collections.Counter()
    stress_counts = collections.Counter()
    by_lemma = tr.groupby("lemma_ru", sort=False)
    pair_n = 0
    for _, frame in by_lemma:
        rows = list(frame[["pos", "yat_flag", FEAT, "dialect", "form_vvz"]].itertuples(index=False, name=None))
        n = len(rows)
        for i in range(n):
            p1, y1, f1, d1, form1 = rows[i]
            s1 = plain(form1)
            a1 = stress_pos(form1)
            for j in range(n):
                if i == j:
                    continue
                p2, y2, f2, d2, form2 = rows[j]
                if p1 != p2:
                    continue
                s2 = plain(form2)
                c = lcp(s1, s2)
                # Do not learn a broad rewrite from an unrelated/distractor form.
                t1, t2 = s1[c:], s2[c:]
                if c < 2 or len(t1) > 7 or len(t2) > 10:
                    continue
                # A relation is specific to source and target feature/dialect.
                key = (p1, f1, d1, f2, d2, t1)
                rule_counts[(key, t2)] += 1
                key_yat = (p1, str(y1), f1, d1, f2, d2, t1)
                rule_counts[(key_yat, t2)] += 1
                a2 = stress_pos(form2)
                if a1 >= 0 and a2 >= 0:
                    stress_counts[(key, a2 - a1)] += 1
                    stress_counts[(key_yat, a2 - a1)] += 1
                pair_n += 1
    rules = {}
    for (key, replacement), count in rule_counts.items():
        old = rules.get(key)
        if old is None or count > old[1]:
            rules[key] = (replacement, count)
    stress = {}
    for key, delta in stress_counts:
        count = stress_counts[(key, delta)]
        old = stress.get(key)
        if old is None or count > old[1]:
            stress[key] = (delta, count)
    print("pair_examples", pair_n, "ending_rules", len(rules), "stress_rules", len(stress), flush=True)
    return rules, stress


def build_indexes(tr):
    # Exact same lemma+feature transfer across dialects; majority is robust to noise.
    modal = {}
    for (lemma, pos, feat), group in tr.groupby(["lemma_ru", "pos", FEAT], sort=False):
        counts = group.form_vvz.value_counts()
        modal[(lemma, pos, feat)] = (counts.index[0], int(counts.iloc[0]), len(group), len(counts))
    lemma_rows = {}
    for lemma, group in tr.groupby("lemma_ru", sort=False):
        lemma_rows[lemma] = list(group[["pos", "yat_flag", FEAT, "dialect", "form_vvz"]].itertuples(index=False, name=None))
    return modal, lemma_rows


def predict_one(row, modal, lemma_rows, rules, stress_rules):
    key = (row.lemma_ru, row.pos, getattr(row, FEAT))
    direct = modal.get(key)
    candidates = collections.Counter()
    donors = lemma_rows.get(row.lemma_ru, [])
    for pos, yf, sf, sd, form in donors:
        if pos != row.pos:
            continue
        src = plain(form)
        for tail_len in range(0, min(7, len(src)) + 1):
            tail = src[len(src) - tail_len :] if tail_len else ""
            rkey = (pos, sf, sd, getattr(row, FEAT), row.dialect, tail)
            rkey_yat = (pos, str(row.yat_flag), sf, sd, getattr(row, FEAT), row.dialect, tail)
            got = rules.get(rkey_yat) or rules.get(rkey)
            if got is None:
                continue
            replacement, support = got
            base = src[: len(src) - tail_len] if tail_len else src
            dst = base + replacement
            stress = stress_rules.get(rkey_yat) or stress_rules.get(rkey)
            sp = stress_pos(form)
            if stress and sp >= 0:
                sp = sp + stress[0]
            elif sp >= 0 and replacement:
                # Default to a stress near the same relative position.
                sp = min(len(dst) - 1, sp)
            dst = put_stress(dst, sp)
            candidates[dst] += max(1, min(support, 50))
    analogy = None
    confidence = 0.0
    if candidates:
        analogy, support = candidates.most_common(1)[0]
        confidence = support / max(1, sum(candidates.values()))
    fallback = None
    if donors:
        qtags = set(getattr(row, FEAT).split(";"))
        best = None
        for pos, yf, sf, sd, form in donors:
            if pos != row.pos:
                continue
            tags = set(sf.split(";"))
            sim = len(qtags & tags)
            rank = (sim, sd == row.dialect, -abs(len(sf) - len(getattr(row, FEAT))))
            if best is None or rank > best[0]:
                best = (rank, form)
        if best:
            fallback = best[1]
    return direct, analogy, confidence, fallback


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dev"
    tr = pd.read_csv(f"{DATA}/train.csv").fillna("")
    dv = pd.read_csv(f"{DATA}/{mode}.csv").fillna("")
    threshold = float(os.environ.get("MORPH_SIM_THRESHOLD", ".25"))
    clean, similarities = clean_train(tr, threshold)
    print("outlier_rows", int((similarities <= threshold).sum()), "of", len(tr), "threshold", threshold, flush=True)
    # Wug seed forms are available only at test time. Keep development scoring clean.
    modal, lemma_rows = build_indexes(clean)
    rules, stress_rules = build_rules(clean)
    if "form_vvz" in dv.columns:
        direct, analogy, confidence, fallback = [], [], [], []
        for row in dv.itertuples(index=False):
            d, a, c, f = predict_one(row, modal, lemma_rows, rules, stress_rules)
            direct.append(d[0] if d else None)
            analogy.append(a)
            confidence.append(c)
            fallback.append(f)
        y = dv.form_vvz.astype(str).values
        base = [d if d is not None else a if a is not None else f if f is not None else str(lemma)
                for d, a, f, lemma in zip(direct, analogy, fallback, dv.lemma_ru)]
        print("default direct/analogy/fallback", score(y, base), flush=True)
        for name, preds in [("direct_only", [d if d is not None else f if f is not None else str(lemma) for d, f, lemma in zip(direct, fallback, dv.lemma_ru)]),
                            ("analogy_first", [a if a is not None else d if d is not None else f if f is not None else str(lemma) for d, a, f, lemma in zip(direct, analogy, fallback, dv.lemma_ru)])]:
            print(name, score(y, preds), flush=True)
        for thresh in [0.0, 0.25, 0.5, 0.75, 0.9, 0.99]:
            preds = [a if a is not None and c >= thresh else d if d is not None else a if a is not None else f if f is not None else str(lemma)
                     for d, a, c, f, lemma in zip(direct, analogy, confidence, fallback, dv.lemma_ru)]
            print("analogy_over_direct_conf", thresh, score(y, preds), flush=True)
        print("availability", {"direct": sum(x is not None for x in direct), "analogy": sum(x is not None for x in analogy), "fallback": sum(x is not None for x in fallback)}, flush=True)
    if mode == "test_features":
        wg = pd.read_csv(f"{DATA}/wug_seeds.csv").fillna("")
        # Add one-shot labeled examples as extra donors for unseen test lemmas.
        waug = pd.concat([clean, wg[["lemma_ru", "pos", FEAT, "dialect", "form_vvz"]]], ignore_index=True)
        modal, lemma_rows = build_indexes(waug)
        preds = []
        for row in dv.itertuples(index=False):
            d, a, c, f = predict_one(row, modal, lemma_rows, rules, stress_rules)
            preds.append(d[0] if d is not None else a if a is not None else f if f is not None else str(row.lemma_ru))
        out = pd.DataFrame({"id": dv.id, "form_vvz": preds})
        out.to_csv("submission.csv", index=False)
        print("wrote submission.csv", out.shape, flush=True)


if __name__ == "__main__":
    main()
