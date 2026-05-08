# Final Report Draft

## Abstract

This project builds a reading comprehension quiz generator for RACE-style passages using a Model A question/verifier pipeline and a Model B distractor/hint pipeline.

## Model A

Model A generates a template question from an extracted answer candidate and verifies selected options using Logistic Regression.

## Model B

Model B ranks distractor candidates using a Random Forest classifier trained on edit distance, length difference, article membership, and cosine similarity.

## UI

The Streamlit interface contains Article Input, Quiz View, Hint Panel, and Developer Dashboard screens.

## Limitations

The current hint generator is extractive and rule-based because the Model B asset zip contains only the distractor ranker.

