# NLP POS Tagging: Rule-Based vs HMM

This project implements and compares two classic Part-of-Speech (POS) tagging approaches in Natural Language Processing: a Rule-Based Tagger and a Hidden Markov Model (HMM) Tagger using the Viterbi algorithm.

## Overview

POS tagging assigns grammatical labels (noun, verb, adjective, etc.) to each word in a sentence. It is a fundamental step in many NLP applications.

This project builds both approaches from scratch and evaluates their performance on the Penn Treebank dataset.

## Features

* Rule-Based POS Tagger using lexical and suffix rules
* HMM Bigram Tagger with Viterbi decoding
* Hybrid approach for handling unknown (OOV) words
* Accuracy comparison between both models
* Visualization (accuracy graphs & confusion matrix)

## Key Idea

The project introduces a hybrid technique where the HMM model uses rule-based suffix heuristics for unknown words instead of assigning uniform probabilities.

## Results

* Rule-Based Accuracy: ~69%
* HMM Accuracy: ~93%

## Dataset

* Penn Treebank (Wall Street Journal corpus)
* Loaded using NLTK

## Installation

pip install nltk numpy matplotlib seaborn tabulate
python -c "import nltk; nltk.download('treebank')"

## How to Run

python nlp_updated.py

## Concepts Covered

* POS Tagging
* HMM
* Viterbi Algorithm
* NLP Basics

## Conclusion

HMM performs significantly better than the rule-based approach due to its ability to use context and learned probabilities.
