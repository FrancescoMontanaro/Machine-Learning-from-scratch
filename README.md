# Machine Learning from Scratch

This repository contains implementations of various machine learning algorithms developed from scratch using only Python and Numpy for educational and entertainment purposes. Each algorithm is implemented in Python and organized according to its category (Supervised, Unsupervised, Reinforcement Learning, and Deep Learning). The code follows a clear structure, with each folder containing separate modules for the core algorithm and training routines.

Unlike standard implementations using existing libraries, **every module (except for the Transformer and GPT-2 section) has been built entirely from scratch using only Python and NumPy**. This includes the implementation of a custom **Deep Learning framework** that allows users to create and train neural networks without relying on external deep learning libraries like TensorFlow or PyTorch. The challenge of manually implementing features such as training pipelines, automatic differentiation, optimization algorithms, activation functions, and backpropagation adds significant depth to this project.

---

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Algorithms](#algorithms)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
  - [Deep Learning](#deep-learning)
- [Usage](#usage)

---

## Overview

This repository serves as a comprehensive collection of foundational machine learning algorithms, with each algorithm implemented from scratch. Each model is accompanied by its training pipeline, allowing users to test the algorithms on data.

## Requirements

Ensure you have Python 3.11 or later. Install the required dependencies using:

```bash
pip install -r requirements.txt
```

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/FrancescoMontanaro/Machine-learning-kit.git
   cd Machine-learning-kit
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

The project structure is organized into categories and subcategories for each algorithm, as shown below:

```plaintext
├── Deep Learning
│   ├── GPT-2
│   ├── Neural Networks
│   ├── Transformer
│   └── common
├── Machine Learning
│   ├── Reinforcement Learning
│   ├── Supervised Learning
│   │   ├── Classification
│   │   ├── General Algorithms
│   │   ├── Regression
│   ├── Unsupervised Learning
│   │   ├── Clustering
│   │   └── Dimensionality Reduction
├── README.md
└── requirements.txt
```

## Algorithms

### Deep Learning

1. **GPT-2**: A Transformer-based language model implementation using only the Pytorch libraries.
2. **Neural Networks**: **Custom deep learning framework built from scratch** without external libraries but Numpy, implementing:
   - **Fully connected layers**
   - **Convolutional layers**
   - **Pooling layers**
   - **Normalization layers**
   - **Custom optimizers and loss functions**
   - **Backpropagation and automatic differentiation**
3. **Transformer**: A custom transformer implementation for language modeling, including:
   - **Attention Mechanism** (`attention_mechanism.py`)
   - **Feed-Forward Network** (`feed_forward.py`)
   - **Data Handling and Utilities** (`data_loader.py`, `utils.py`)

### Reinforcement Learning

1. **Multi-Armed Bandit** - Implementations of algorithms for the Multi-Armed Bandit problem, such as:
   - **Upper Confidence Bound (UCB1)**: `ucb1.py`
   - **Thompson Sampling**: `thompson_sampling.py`

### Supervised Learning

1. **K-Nearest Neighbours (KNN)** (`knn.py`)
2. **Linear Regression** (`linear_regression.py`)
3. **Logistic Regression** (`logistic_regression.py`)
4. **Naive Bayes** (`naive_bayes.py`)
5. **Perceptron** (`perceptron.py`)
6. **Random Forest** (`random_forest.py`)
7. **Support Vector Machine (SVM)** (`svm.py`)

### Unsupervised Learning

1. **K-Means Clustering** (`k_means.py`)
2. **Principal Component Analysis (PCA)** (`pca.py`)

## Usage

Each algorithm comes with a `train.ipynb` notebook that can be used to train the model with sample data.
