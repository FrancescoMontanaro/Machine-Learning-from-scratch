# Machine Learning from Scratch

This repository contains implementations of various machine learning algorithms developed from scratch **using only Python and NumPy**. Each algorithm is implemented and organized according to its category (Supervised, Unsupervised, Reinforcement Learning, and Deep Learning). The code follows a clear structure, with each folder containing separate modules for the core algorithm and training routines.

Unlike standard implementations using existing libraries, **every module has been built entirely from scratch using only Python and NumPy**. This includes the implementation of a custom **Deep Learning framework** that allows users to create and custom DL models without relying on external libraries like TensorFlow or PyTorch. The challenge of manually implementing features such as training pipelines, automatic differentiation, optimization algorithms, activation functions, and backpropagation adds has been a great educational and fun experience.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Folder Structure](#folder-structure)
- [Algorithms](#algorithms)
  - [Deep Learning](#deep-learning)
  - [Reinforcement Learning](#reinforcement-learning)
  - [Supervised Learning](#supervised-learning)
  - [Unsupervised Learning](#unsupervised-learning)
- [Usage](#usage)

---

## Overview

This repository serves as a comprehensive collection of foundational machine learning algorithms, with each algorithm implemented from scratch. Each one is accompanied by its training pipeline, allowing users to test the algorithms on data.

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/FrancescoMontanaro/Machine-Learning-from-scratch
   cd Machine-Learning-from-scratch
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Folder Structure

The project structure is organized into categories and subcategories for each algorithm, as shown below:

```plaintext
├── Deep Learning
│   ├── Auto-Encoder
│   ├── Convolutional Neural Network (CNN)
│   └── Multi-Layer Perceptron (MLP)
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

1. **Custom Deep Learning framework built from scratch** without external libraries but Numpy, implementing:
   - **Fully connected layers**
   - **Convolutional layers**
   - **Pooling layers**
   - **Up-sampling layers**
   - **Normalization layers**
   - **Optimizers and loss functions**
   - **Activation functions**
   - **Backpropagation and automatic differentiation**
   - **Custom model definition and training pipeline**
   
   The main modules implemented so far are:
   1. [**Auto-Encoder**](Deep%20Learning/Auto-Encoder)
   2. [**Multi-Layer Perceptron (MLP)**](Deep%20Learning/Multi-Layer%20Perceptron%20(MLP))
   3. [**Convolutional Neural Network (CNN)**](Deep%20Learning/Convolutional%20Neural%20Network%20(CNN))


### Reinforcement Learning

1. [**Multi-Armed Bandit**](Machine%20Learning/Reinforcement%20Learning/Multi%20Armed%20Bandit)
   - Implementations of algorithms for the Multi-Armed Bandit problem, such as:
      - **Upper Confidence Bound (UCB1)**
      - **Thompson Sampling**

### Supervised Learning

1. [**K-Nearest Neighbours (KNN)**](Machine%20Learning/Supervised%20Learning/General%20Algorithms/K%20Nearest%20Neghbours)
2. [**Linear Regression**](Machine%20Learning/Supervised%20Learning/Regression/Linear%20Regression)
3. [**Logistic Regression**](Machine%20Learning/Supervised%20Learning/Classification/Logistic%20Regression)
4. [**Naive Bayes**](Machine%20Learning/Supervised%20Learning/Classification/Naive%20Bayes)
5. [**Perceptron**](Machine%20Learning/Supervised%20Learning/General%20Algorithms/Perceptron)
6. [**Random Forest**](Machine%20Learning/Supervised%20Learning/General%20Algorithms/Random%20Forest)
7. [**Support Vector Machine (SVM)**](Machine%20Learning/Supervised%20Learning/Classification/Support%20Vector%20Machine)

### Unsupervised Learning

1. [**K-Means Clustering**](Machine%20Learning/Unsupervised%20Learning/Clustering/K%20Means%20Clusetring)
2. [**Principal Component Analysis (PCA)**](Machine%20Learning/Unsupervised%20Learning/Dimensionality%20Reduction/Principal%20Component%20Analysis)

## Usage

Each algorithm comes with one or more training notebooks with the extension `.ipynb`, that can be used to train the model with sample data.
