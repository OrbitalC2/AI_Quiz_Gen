# RACE Quiz Studio

An end-to-end reading comprehension quiz generator trained on the RACE dataset. Give it any English passage and it produces a four-option MCQ with graduated hints and a developer analytics dashboard.

## Project Structure

```
AI_PROJECT/
├── features.py                  # Shared feature engineering (training + inference)
├── question_ranker.py           # RF question ranker features and candidate generation
├── template_quiz_pipeline.py    # Full inference pipeline (EnsembleAnswerPredictor, QuestionRankerPredictor, ...)
├── src/
│   ├── model_a_train.py         # Train Model A: LR+SVM+NB+RF ensemble + T5 baseline
│   ├── model_b_train.py         # Train Model B: distractor RF ranker
│   ├── question_ranker_train.py # Train RF question ranker
│   ├── preprocessing.py         # Shared data loading and cleaning
│   ├── inference.py             # ReadingComprehensionPipeline class
│   ├── evaluate.py              # CLI: evaluate MCQ accuracy on train.csv
│   └── evaluate_distractors.py  # CLI: evaluate distractor quality
├── ui/
│   ├── app.py                   # Streamlit single-page app
│   └── components/
│       ├── state.py             # Session state and pipeline calls
│       ├── styles.py            # CSS injection and metrics data
│       └── sample_data.py       # Built-in sample passages
├── models/
│   ├── model_a/traditional/     # LR, SVM, NB, RF verifiers + TF-IDF + KMeans + SVD
│   ├── model_a/neural/          # T5-small fine-tuned weights (not in git, >100MB)
│   └── model_b/traditional/     # RF distractor ranker + distractor index
├── notebooks/
│   ├── EDA.ipynb                # Dataset exploration
│   └── experiments.ipynb        # Training experiments log
├── tests/
│   └── test_inference.py        # Pytest unit tests
├── report/
│   └── report.pdf               # Final project report
├── data/
│   ├── raw/train.csv            # RACE dataset (not in git, 149MB)
│   └── processed/               # Intermediate processed files
└── requirements.txt
```

## Quick Setup

```bash
# 1. Clone and enter the repo
git clone git@github.com:OrbitalC2/AI_Quiz_Gen.git
cd AI_Quiz_Gen

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the RACE dataset
#    Place train.csv inside data/raw/
#    Dataset: https://huggingface.co/datasets/ehovy/race
```

## Running the App

The small model artifacts (LR, SVM, NB, KMeans, SVD, TF-IDF, idfDict) are committed to the repo. The app works out of the box without retraining.

```bash
streamlit run ui/app.py
```

The first page load pre-warms the pipeline in a background thread. By the time you click "Generate Quiz" the models are loaded.

## Training

### Model A — Answer Verifier Ensemble + T5 Baseline

```bash
# Train traditional ensemble only (LR + Calibrated SVM + NB + RF + KMeans)
python3 src/model_a_train.py --task traditional --train-csv data/raw/train.csv

# Train T5-small neural baseline only
python3 src/model_a_train.py --task neural --train-csv data/raw/train.csv

# Train both
python3 src/model_a_train.py --task all --train-csv data/raw/train.csv
```

Artifacts saved to `models/model_a/traditional/` and `models/model_a/neural/model_a_final/`.

Training time on full dataset (~87k articles):
- Traditional ensemble: ~20-40 min depending on CPU
- T5 fine-tuning (5 epochs, batch 16): ~3-6 hours on GPU, much longer on CPU

### Model B — Distractor Ranker

```bash
python3 src/model_b_train.py --train-csv data/raw/train.csv --sample-size 3000
```

Artifact saved to `models/model_b/traditional/rfDistractorRanker.joblib`.

Note: model_b_train.py uses `sentence-transformers` to encode options. First run will download `all-MiniLM-L6-v2` (~90MB).

### RF Question Ranker

```bash
python3 src/question_ranker_train.py \
  --train-csv data/raw/train.csv \
  --model-dir models/model_a/traditional
```

Artifact saved to `models/model_a/traditional/rfQuestionRanker.joblib`.

## Evaluation

```bash
# MCQ accuracy on first 25 articles
python3 src/evaluate.py --data data/raw/train.csv --limit 25

# Distractor quality breakdown
python3 src/evaluate_distractors.py --data data/raw/train.csv --limit 25
```

## Running Tests

```bash
pytest tests/ -v
```

Tests cover answer candidate extraction, option shuffling, question template quality, feature vector shape, and corpus-backed distractor filtering.

## Model Results Summary

| Component | Metric | Value |
|---|---|---|
| Verifier Ensemble | MCQ Accuracy | 36.5% (random: 25%) |
| Verifier Ensemble | Binary F1 | 0.376 |
| Verifier Ensemble | 5-fold Cross-Val F1 | 0.526 ± 0.002 |
| RF Question Ranker | ROC-AUC | 0.972 |
| RF Question Ranker | Top-1 Match Rate | 66.9% |
| Distractor Ranker | Accuracy | 88.0% |
| Distractor Ranker | F1 | 0.83 |

## Large Files (not in git)

These exceed GitHub's 100MB limit and must be trained locally or obtained separately:

| File | Size | How to get |
|---|---|---|
| `models/model_a/traditional/rfVerifierModel.joblib` | 116 MB | Run `model_a_train.py --task traditional` |
| `models/model_a/traditional/rfQuestionRanker.joblib` | 44 MB | Run `question_ranker_train.py` |
| `models/model_b/traditional/raceDistractorIndex.joblib` | 225 MB | Run `model_b_train.py` |
| `models/model_a/neural/model_a_final/model.safetensors` | 231 MB | Run `model_a_train.py --task neural` |
| `data/raw/train.csv` | 149 MB | Download from HuggingFace RACE |

## Dataset

Lai, G., Xie, Q., Liu, H., Yang, Y., & Hovy, E. (2017). RACE: Large-scale ReAding Comprehension Dataset From Examinations. EMNLP 2017.  
https://aclanthology.org/D17-1082

## References

See `report/report.pdf` for full references and project report.
