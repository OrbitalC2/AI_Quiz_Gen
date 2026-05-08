# RACE Reading Comprehension Quiz Generator

This project implements the Intelligent Reading Comprehension and Quiz Generation System described in the FAST AI lab project brief.

## Structure

```text
race_rc_project/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── model_a/
│   │   ├── neural/
│   │   └── traditional/
│   └── model_b/
│       ├── neural/
│       └── traditional/
├── src/
│   ├── preprocessing.py
│   ├── model_a_train.py
│   ├── model_b_train.py
│   ├── inference.py
│   ├── evaluate.py
│   └── evaluate_distractors.py
├── ui/
│   ├── app.py
│   └── components/
├── notebooks/
├── tests/
├── requirements.txt
└── report/
```

## Current Pipeline

Model A:
- Template-based answer and question generation.
- Traditional verifier ensemble with Logistic Regression, calibrated SVM, Random Forest, and shared feature engineering.
- Optional T5 baseline checkpoint lives under `models/model_a/neural/model_a_final/`.

Model B:
- Corpus-backed RACE distractor retrieval with Random Forest reranking.
- Extractive hint masking from the sentence containing the answer.

UI:
- Streamlit app with Article Input, Quiz View, Hint Panel, and Developer Dashboard.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place RACE CSV files in:

```text
data/raw/train.csv
data/raw/test.csv
data/raw/val.csv
```

The extracted model files live in:

```text
models/model_a/neural/model_a_final/
models/model_a/traditional/
models/model_b/traditional/
```

Original model zip archives should stay outside the repo. The current backup location is:

```text
~/Desktop/race_rc_project_model_archives/
```

## Run The UI

```bash
streamlit run ui/app.py
```

## Run Tests

```bash
pytest
```

## Train Models

Traditional Model A verifier:

```bash
python3 src/model_a_train.py --train-csv data/raw/train.csv --output-dir models/model_a/traditional
```

Model B distractor ranker:

```bash
python3 src/model_b_train.py --train-csv data/raw/train.csv --output-dir models/model_b/traditional
```

Distractor quality evaluation:

```bash
python3 src/evaluate_distractors.py --data data/raw/train.csv --limit 25
```

Question ranker for rule-generated questions:

```bash
python3 src/question_ranker_train.py --train-csv data/raw/train.csv --output-dir models/model_a/traditional --sample-size 30000
```

For a faster experiment without Model A margin features:

```bash
python3 src/question_ranker_train.py --train-csv data/raw/train.csv --output-dir models/model_a/traditional --sample-size 30000 --no-model-a-feature
```
