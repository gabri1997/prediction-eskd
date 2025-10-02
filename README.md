# ANN Training for ESKD Prediction

This repository contains a PyTorch implementation of a **binary classification neural network** to predict the occurrence of End-Stage Kidney Disease (ESKD) using clinical variables. The network is trained using **stratified 10-fold cross-validation** on 80% of the data and validated/tested on the remaining 20%. The training process is integrated with **Weights & Biases (wandb)** for experiment tracking.

---

## Features

- Handles missing or infinite values in the dataset.
- Uses a simple feed-forward neural network with 4 hidden layers of 100 units each.
- Dropout and batch normalization for regularization.
- Exponential learning rate scheduler.
- Binary cross-entropy loss with class imbalance handling.
- Automatic saving of the best model based on **F1 score**.
- Stratified splitting ensures class balance between training and test sets.
- Logging of training metrics (accuracy, precision, recall, F1 score, loss, learning rate) with **wandb**.

---

## Dataset

The model expects a **Pandas DataFrame** with the following columns:

- `Eskd` → binary target (1 if ESKD occurred, 0 otherwise)
- `Code` → patient identifier (not used in training)
- `Gender` → 'M' or 'F' (automatically encoded as 0/1)
- Remaining columns → clinical features (numerical)

**Notes:**
- Only numeric features are used for training.
- Missing and infinite values are automatically replaced with 0.

---

## Neural Network Architecture

- Input layer → size equals number of features
- 4 hidden layers:
  - 100 neurons each
  - ELU activation
  - BatchNorm
  - Dropout (configurable)
- Output layer → 1 neuron (logit for binary classification)
- Loss → `BCEWithLogitsLoss` with `pos_weight` to handle class imbalance

---