# Analysis-and-Optimisation-of-Text-Generating-Models

A research-focused project exploring enhancements in Recurrent Neural Network (RNN)-based language models by integrating syntactic (POS) information.


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

## Setup

- python 3.4+
- chainer
- numpy
- keras
- argparse
- nltk (with brown corpus and wordnet data)

## File Synopsis

1. [`train_brown.py`](train_brown.py): Code for running base model (no parts of speech information)
    1. *class* **RNNForLM**: Main recurrent network taking one hot word vectors as input
    2. *class* **ParallelSequentialIterator**: Dataset iterator to create a batch of sequences at different positions
    3. *class* **BPTTUpdater**: Custom updater for truncated BackProp Through Time (BPTT)
    4. *function* **compute_perplexity**: Routine to rewrite the result dictionary of LogReport to add perplexity values

2. [`custom_classifier.py`](custom_classifier.py): Classifier wrapper to setup standard classifier components (Loss function, Metrics etc)


3. [`posword.py`](posword.py): Chainer based RNN model (words + parts of speech information incorporated)
    1. *class* **BPTTUpdater** : Custom updater for truncated BackProp Through Time (BPTT)
    2. *class* **ParallelSequentialIterator**: Dataset iterator to create a batch of sequences at different positions
    3. *class* **Rec2Network**: Main RNN model which takes word and POS tags as inputs
    4. *class* **RNNForLM**: Subnetwork for Rec2Network Class
    5. *class* **RecNetwork**: Trial RNN model, not used
    6. *class* **RNNForBrown**: Subnetwork for RecNetwork
    7. *class* **StackedLSTMLayers**: Class defining the LSTM subnetwork used in Rec2Network
    8. *class* **StackedLinearLayers**: Class defining the Linear layers used in Rec2Network
    9. *function* **compute_perplexity**: Routine to rewrite the result dictionary of LogReport to add perplexity values

4. [`test_recnetwork.py`](test_recnetwork.py): Driver program for [`posword.py`](posword.py)
   # Rec2 PyTorch

This is a PyTorch reimplementation of the Rec2 model for part-of-speech prediction on the Brown corpus.

## Train

```bash
python train.py --r1units 64 --r2units 64 --epochs 10

