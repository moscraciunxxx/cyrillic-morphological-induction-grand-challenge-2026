# Work log

## Competition and data

The task is to infer inflected forms in a synthetic Cyrillic language. Inputs include Russian lemmas, POS, grammatical feature bundles, dialect, lemma frequency, and a yat flag. The target is `form_vvz`; submission columns are `id,form_vvz`. Kaggle's score combines weighted exact match (90%) and weighted character error rate (10%).

The local run used 300,000 training rows, 20,000 development rows, 4,800 wug seed examples, and 60,000 test feature rows. Source data remains local and is intentionally omitted from Git.

## Modeling work

1. Read the competition description and data schema and obtained the training, development, seed, and test files.
2. Measured lemma overlap. Every development lemma occurred in training, while test contained unseen lemmas. This made the development estimate optimistic for test generalization.
3. Built a paradigm model that removes obvious distractor rows, transfers exact same-lemma feature forms across dialects, and learns ending and stress changes from pairs of forms within a lemma.
4. The stricter cleaning threshold removed 3,344 of 300,000 training rows. The best local development proxy was **0.72475** (exact match **0.6987**, CER **0.04081**). The proxy used unweighted row averages rather than Kaggle's hidden weights.
5. Simulated unseen words using 4,750 held-out lemmas and 61,727 non-seed rows. With one canonical seed per lemma, the model scored **0.27885**. Several analogy confidence thresholds gave the same result. Preferring the copied seed form as fallback scored **0.15903**.
6. Trained a character GRU. A data-preparation assignment error made its development and test frames copies of training data, invalidating its reported score and generated character predictions. The source bug is fixed in `train_char.py`; no corrected model result is claimed.
7. Generated the rule-based `submission.csv` with 60,000 rows. The file had the required columns, unique IDs in test order, and no blank predictions. Kaggle accepted it and reported a public score of **0.31989**. The entry was selected as one of the final submissions in the competition UI.

## Artifacts

- `paradigm_model.py` is the baseline that generated the submitted CSV.
- `simulate_wug.py` and `tune_wug.py` capture seed-based evaluation.
- `train_char.py` is corrected experimental source, not a validated alternative.
- Competition files, generated predictions, model checkpoints, and local dependencies are ignored by Git.

## Limits

The public score exposed a large gap between the seen-lemma development split and hidden evaluation. The leave-lemma-out simulation gave a more relevant warning about test generalization. The character model needs retraining and a clean held-out-lemma evaluation before further use.
