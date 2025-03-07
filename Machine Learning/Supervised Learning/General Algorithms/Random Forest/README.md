
# Random Forest

This repository includes a random forest model built from scratch, using decision trees as base estimators. Each tree in the forest is constructed independently, with the final model aggregating their predictions for robust classification or regression. The model is implemented with minimal dependencies in Python.

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
    - [Impurity Measures](#impurity-measures)
    - [Information Gain](#information-gain)
    - [Recursive Splitting](#recursive-splitting)
3. [Random Forest Algorithm Example](#random-forest-algorithm-example)
4. [Detailed Components & Calculations](#detailed-components-and-calculations)
    - [Decision Tree Structure](#decision-tree-structure)
    - [Node Representation](#node-representation)
    - [Model Training](#model-training)
    - [Prediction](#prediction)

## Introduction

The random forest algorithm is an ensemble learning technique that combines the predictions of multiple decision trees to enhance accuracy and prevent overfitting. Each tree in the forest is trained on a random subset of data and a random subset of features, contributing to the robustness of the overall model.

## Mathematical Foundations

### Impurity Measures

To select optimal splits, decision trees minimize impurity at each node. The impurity measure here includes:

2. **Entropy** (Classification): Measures disorder in the dataset, used in information gain calculations.
   $$
   Entropy(S) = - \sum_{i=1}^C p_i \log_2(p_i)
   $$

### Information Gain

Information gain quantifies the reduction in impurity from splitting data at a node. For a given split with subsets $S_l$ and $S_r$, information gain is calculated as:

$$
IG(S, A) = Impurity(S) - \frac{|S_l|}{|S|} Impurity(S_l) - \frac{|S_r|}{|S|} Impurity(S_r)
$$

This value determines the optimal split by maximizing the decrease in impurity.

### Recursive Splitting

Trees are grown by recursively partitioning the dataset based on feature splits that maximize information gain. Splitting continues until reaching a stopping criterion:

- **Maximum Depth**: The tree is limited to a set depth.
- **Minimum Samples Split**: Nodes must contain a minimum number of samples to be split further.

## Random Forest Algorithm Example

1. **Dataset**: Matrix $X$ of shape $(n, d)$ with $n$ samples and $d$ features.
2. **Labels**: Vector $y \in \mathbb{R}^n$ of target values.
3. **Number of Trees**: $n_{\text{trees}}$ trees, set by the user (default: 20).
4. **Tree Depth**: Depth $d_{\text{max}}$ for each tree to prevent overfitting (default: 100).

## Detailed Components and Calculations

### Decision Tree Structure

Each decision tree is a series of nodes connected by left and right children pointers. A `Node` splits data recursively using impurity reduction methods until the tree reaches maximum depth or other criteria.

### Node Representation

The `Node` class represents each split in a tree:

1. **Feature**: Index of the split feature.
2. **Threshold**: Split threshold for the feature.
3. **Left & Right Children**: Child pointers.
4. **Value**: Leaf node prediction.

A node splits data by finding the feature and threshold that yield the highest information gain or impurity reduction.

### Model Training

Training consists of generating multiple trees by bootstrap sampling:

1. **Bootstrap Sampling**: A random subset of samples is selected for each tree.
2. **Feature Sampling**: A random subset of features is chosen at each node.

Each decision tree is trained independently to reach a maximum depth or stop criterion.

### Prediction

Predictions are aggregated across trees for robustness:

1. **Classification**: Returns the class with the majority vote from all trees.
2. **Regression**: Outputs the mean of predictions across trees.
