# Cyrillic Morphological Induction Grand Challenge 2026

Experiments for Kaggle's synthetic Cyrillic inflection task. The model predicts `form_vvz` from a Russian lemma, part of speech, morphological features, dialect, and auxiliary features.

## Outcome

- The paradigm model combines same-lemma feature evidence, dialect transfer, and learned ending and stress changes.
- The best local development estimate was **0.7247** using an unweighted proxy for Kaggle's metric. Every development lemma appeared in training, making that estimate optimistic for unseen lemmas.
- A leave-lemma-out experiment with one seed per held-out lemma scored **0.2788**. Copying a seed form as fallback scored **0.1590**; confidence threshold changes did not help.
- The 60,000-row `submission.csv` passed structural checks and was accepted by Kaggle. Its public score was **0.31989**. The submission was selected as a final candidate in the competition UI.
- A character GRU experiment had a data assignment bug that invalidated its reported results. The source bug is corrected in `train_char.py`; the corrected model has not been retrained or evaluated.

This repository records the work and its limitations. It does not claim a winning result.

## Source files

- `paradigm_model.py`: cleaning, paradigm rules, predictions, and submission generation.
- `simulate_wug.py`: leave-lemma-out evaluation with one seed per held-out lemma.
- `tune_wug.py`: seed, analogy, and fallback strategy comparison.
- `train_char.py`: experimental character GRU source, corrected after the original run.
- `WORK_LOG.md`: detailed account of the experiments and results.

## Data and use

Obtain the competition data from Kaggle and place these files under `data/`:

```text
data/train.csv
data/dev.csv
data/wug_seeds.csv
data/test_features.csv
```

The data, installed runtimes, checkpoints, and generated predictions are ignored by Git. Follow Kaggle's current rules before redistributing any competition materials.

```bash
python -m pip install -r requirements.txt
python paradigm_model.py dev
python paradigm_model.py test_features
```

The test command writes `submission.csv` in the project root. The character model remains experimental and needs a genuine held-out-lemma evaluation before use.

## Metric

Kaggle describes the competition score as `0.9 × Weighted Exact Match + 0.1 × (1 − Weighted CER)`. Our local development helper uses unweighted row averages, so its score is an approximation rather than an exact reconstruction of Kaggle's hidden weighting.
