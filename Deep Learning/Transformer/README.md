# Transformer

This repository contains a transformer model built from scratch in PyTorch, with an in-depth explanation of each layer. Using a simple example sentence, we detail each computational step, matrix transformation, and operation within the model.

## Table of Contents

1. [Introduction](#introduction)
2. [Running Example: "I like cats"](#running-example)
3. [Components & Detailed Calculations](#components-and-detailed-calculations)
    - [Tokenizer](#tokenizer)
    - [Token Embedding](#token-embedding)
    - [Positional Encoding](#positional-encoding)
    - [Self-Attention](#self-attention)
    - [Feed-Forward Network](#feed-forward-network)
    - [Transformer Block](#transformer-block)
    - [Output Layer](#output-layer)

## Introduction

This project explores each part of a transformer model, inspired by "Attention Is All You Need." By using the sentence **"I like cats,"** we calculate and display each transformation matrix, illustrating how the transformer processes input tokens.

## Running Example: "I like cats"

We’ll follow **"I like cats"** step-by-step, with detailed matrix manipulations and calculations.

### Example Details

1. **Input Sentence**: "I like cats"
2. **Tokenization**:
    - The tokenizer splits the text into subword units or tokens using a Byte Pair Encoding (BPE) mechanism.
    - Each token is encoded as an integer based on a trained vocabulary.
    - For example:
        - Sentence: `"I like cats"`
        - Tokens: `[1, 2, 3]` (IDs corresponding to `"I"`, `"like"`, and `"cats"`).
3. **Embedding Dimension**: For simplicity, we set $d_{\text{model}} = 3$, with a vocabulary size $V = 10$.

Each transformation matrix will be explicitly calculated and explained.

## Components and Detailed Calculations

### Tokenizer

The tokenizer is based on Byte Pair Encoding (BPE) and consists of the following main steps:

1. **Training**: 
    - The tokenizer is trained on a corpus of text to learn a vocabulary of tokens and merge rules.
    - Starting from UTF-8 encoded bytes, frequent byte pairs are merged iteratively to form new tokens, expanding the vocabulary.
2. **Encoding**:
    - Input text is split into chunks using a regex pattern and converted into initial byte-level tokens.
    - The learned merge rules are applied iteratively to compress tokens into higher-level subword units.
    - Example:
        - Input: `"I like cats"`
        - Output: `[1, 2, 3]` (token IDs based on the trained vocabulary).
3. **Decoding**:
    - A sequence of token IDs is converted back to text using the reverse of the vocabulary and merge rules.

For example:
- Input: `"I like cats"`
- Encoding: `[1, 2, 3]`
- Decoding: `"I like cats"`

The tokenizer code is implemented in `tokenizer.py` and supports training, saving/loading states, and efficient encoding/decoding.

### Token Embedding

Tokens are converted to embeddings using an embedding matrix $E$ with dimensions $V \times d_{\text{model}} = 10 \times 3$. Each word maps to a 3-dimensional vector.

- **Embedding Matrix $E$** (simplified for explanation):
  $E = \begin{bmatrix}
    0.1 & 0.2 & 0.3 \\
    0.4 & 0.5 & 0.6 \\
    0.7 & 0.8 & 0.9 \\
    \vdots & \vdots & \vdots 
  \end{bmatrix}$

- **Token Embeddings for "I," "like," "cats"** (example calculation):
  - "I" (ID 1): $E[1] = [0.4, 0.5, 0.6]$
  - "like" (ID 2): $E[2] = [0.7, 0.8, 0.9]$
  - "cats" (ID 3): $E[3] = [0.1, 0.2, 0.3]$

The token embedding layer outputs a matrix:
$X_{\text{emb}} = \begin{bmatrix} 
  0.4 & 0.5 & 0.6 \\
  0.7 & 0.8 & 0.9 \\
  0.1 & 0.2 & 0.3 
\end{bmatrix}$

### Positional Encoding

In this implementation, positional encodings are learned embeddings, allowing the model to discover optimal position representations through training. Each position in the input sequence has a corresponding embedding in a positional embedding table, with dimensions $\text{block size} \times d_{\text{model}}$.

- **Positional Embedding Matrix**: 
  $P = \begin{bmatrix} 
      p_{1,1} & p_{1,2} & p_{1,3} \\ 
      p_{2,1} & p_{2,2} & p_{2,3} \\ 
      \vdots & \vdots & \vdots \\
      p_{T,1} & p_{T,2} & p_{T,3}
  \end{bmatrix}$
  where $T$ is the sequence length (block size).

The positional embeddings are added element-wise to the token embeddings, giving each position a unique encoding:

$X_{\text{emb}} + P = \begin{bmatrix} 
  0.4 + p_{1,1} & 0.5 + p_{1,2} & 0.6 + p_{1,3} \\ 
  0.7 + p_{2,1} & 0.8 + p_{2,2} & 0.9 + p_{2,3} \\ 
  0.1 + p_{3,1} & 0.2 + p_{3,2} & 0.3 + p_{3,3} 
\end{bmatrix}$

We add $\text{PE}$ to $X_{\text{emb}}$ for a combined embedding.

### Self-Attention

#### Step 1: Calculate $Q, K, V$ Matrices

We calculate query $Q$, key $K$, and value $V$ matrices by multiplying $X$ with learned weights $W^Q$, $W^K$, and $W^V$, each sized $d_{\text{model}} \times d_k$ (assume $d_k = d_{\text{model}}$ for simplicity).

Example:
$Q = X W^Q, \quad K = X W^K, \quad V = X W^V$

#### Step 2: Scaled Dot-Product Attention

Attention scores are derived by calculating $QK^T$ and scaling by $\frac{1}{\sqrt{d_k}}$:
$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

This step will show matrices for $Q$, $K$, and $V$, intermediate products $QK^T$, scaling, softmax, and the weighted sum result.

### Feed-Forward Network

The feed-forward network applies two linear transformations with a ReLU activation in between. For each token representation, we calculate:
1. **Linear 1**: $X_{\text{ffn}} = \text{ReLU}(X W_1 + b_1)$
2. **Linear 2**: $\text{output} = X_{\text{ffn}} W_2 + b_2$

Example weight matrices and their multiplication with attention outputs will be detailed here.

### Transformer Block

A transformer block is a sequence of:
1. **Self-Attention** (computed above)
2. **Feed-Forward Network** (computed above)
3. **Residual Connections**: Adding the inputs back after attention and feed-forward steps
4. **Layer Normalization**: Ensuring normalized outputs

Each residual and normalization step will include calculations with intermediate matrix outputs.

### Output Layer

The output layer projects the processed embeddings back to the vocabulary size $V$ with a linear transformation:
$\text{logits} = X W_{\text{out}} + b_{\text{out}}$
where $W_{\text{out}}$ has dimensions $d_{\text{model}} \times V$.