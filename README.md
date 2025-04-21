# Analysis-and-Optimisation-of-Text-Generating-Models

A research-focused project exploring enhancements in Recurrent Neural Network (RNN)-based language models by integrating syntactic (POS) information.

> Developed as part of the SURA project at IIT Delhi.

---

## 📚 Project Overview

This repository contains implementations of text-generating models trained on the Brown Corpus using RNN architectures. It explores both baseline language modeling and enhanced versions using part-of-speech (POS) tagging.

---

## 📁 Repository Structure

### `train_brown.py`
Train a baseline RNN-based language model **without POS tags**.

**Key components:**
- `RNNForLM`: Defines a one-hot encoded RNN language model.
- `ParallelSequentialIterator`: Creates mini-batches with BPTT-compatible sequences.
- `BPTTUpdater`: Implements truncated Backpropagation Through Time.
- `compute_perplexity`: Evaluates model performance using perplexity.

---

### `posword.py`
Train a language model with **POS tag integration**.

**Highlights:**
- `Rec2Network`: Takes both word embeddings and POS tag inputs.
- `RNNForLM`: Used as a component within the larger `Rec2Network`.
- Custom iterator and updater adapted for POS-tagged data.

---

### `custom_classifier.py`
Helper module that provides:
- Standard classifier setup
- Loss function and accuracy calculation
- Evaluation hooks

---

## ⚙️ Installation & Setup

Make sure you have the following installed:

```bash
pip install chainer numpy keras nltk
