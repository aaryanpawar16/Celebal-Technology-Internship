# Deep Learning Text Generation — Vanilla RNN vs LSTM vs GRU

## Overview

A learning-focused notebook comparing three sequence model architectures — Vanilla RNN, LSTM, and GRU — on a small text corpus for next-word prediction and text generation. The goal is to understand *why* gated architectures (LSTM, GRU) outperform a vanilla RNN on tasks that need to capture grammar, sentence flow, and longer-range contextual dependencies.

**Learning goal:** compare training loss, generated text quality, memory handling, and long-term dependency learning across the three architectures.

## Corpus & Data Prep

- A small built-in multi-line text corpus (easily swappable for Shakespeare, lyrics, chatbot data, or custom extracted text).
- **Tokenization:** Keras `Tokenizer` converts words to integer IDs.
- **Sequence generation:** each line is broken into n-gram sequences (growing prefixes) for next-word prediction.
- **Padding:** sequences are pre-padded to a uniform length; the last token in each sequence is the prediction target (`y`), the rest is the input (`X`).

Initial run: vocabulary size 37, `X` shape (35, 7), `y` shape (35,).

## Models

All three models share the same shape: `Embedding → [Recurrent Layer] → Dense(softmax)`.

1. **Vanilla RNN** — `SimpleRNN(64)`. Baseline; prone to vanishing gradients, struggles with longer-term dependencies.
2. **LSTM** — `LSTM(64)`. Uses input/forget/output gates to preserve long-term memory.
3. **GRU** — `GRU(64)`. Uses reset/update gates; computationally lighter than LSTM with often-similar results.

Each model is trained for 100 epochs with `sparse_categorical_crossentropy` loss and the Adam optimizer, then compared by training loss curve and by generating text from the seed "deep learning."

## Initial Results (100 epochs, 64 units)

| Model | Generated text (seed: "deep learning", 5 words) |
|---|---|
| RNN | deep learning is transforming artificial intelligence sentences |
| LSTM | deep learning models can generate meaningful sentences |
| GRU | deep learning is transforming artificial artificial intelligence |

On this tiny corpus, LSTM produced the most coherent continuation; the vanilla RNN and GRU showed minor repetition/incoherence artifacts.

## Extended Experiment (Student Tasks)

A second, larger corpus and upgraded configuration were used to test how scale affects each architecture:

- **Custom corpus:** longer, more varied text (vocabulary size 56, `X` shape (62, 11))
- **Embedding dimension:** 32 → 64
- **Hidden units:** 64 → 128
- **Epochs:** 100 → 200
- **Generation length:** 5 → 10 words

### Extended Results

With the larger corpus, more capacity, and more training epochs, all three architectures converged to generating the **same fluent, grammatically correct continuation**:

> "deep learning uses neural networks with many layers to model complex patterns"

This shows that with enough capacity and training time on a small, simple corpus, the practical gap between RNN, LSTM, and GRU narrows — the architectural differences matter most when data or training is constrained, or when dependencies span longer distances.

## Key Takeaways

- Vanilla RNNs can learn short-range patterns but are theoretically more prone to vanishing gradients, making them less reliable for longer dependencies.
- LSTM's gating mechanism is designed to preserve long-range context better than a vanilla RNN.
- GRU offers a lighter-weight alternative to LSTM (fewer gates, faster to train) with comparable output quality.
- On very small corpora, differences between architectures can show up as coherence differences in generated text; with more data, capacity, and training, all three can converge to similar quality.
- This notebook is a practical starting point for understanding sequence modeling before moving to attention-based/Transformer architectures.

## Requirements

```
tensorflow
numpy
matplotlib
```

## How to Run

1. Install dependencies: `pip install tensorflow numpy matplotlib`
2. Run the notebook cells sequentially from top to bottom.
3. To experiment further: swap in your own corpus, adjust `EMBED_DIM`, `HIDDEN_UNITS`, and `EPOCHS`, or increase `next_words` in the generation function.
