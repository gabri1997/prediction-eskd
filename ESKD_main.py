import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score,f1_score,roc_curve,precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation, Dropout,ELU
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.optimizers.schedules import ExponentialDecay
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

"""
Questa è l'implementazione in Keras della rete di Schena et al. (2020)

"""

def ProxyAUCLoss():
    """
    Pairwise ranking loss approximating AUC maximization (Rendle et al., 2009).
    - Handles y_true of shape (batch_size,) or (batch_size, 1)
    - Handles y_pred of shape (batch_size,) or (batch_size, 2)
    """

    def loss(y_true, y_pred):
        # --- Ensure correct dtype and shape ---
        y_true = K.cast(y_true, "float32")
        y_true = tf.reshape(y_true, [-1])  # flatten to 1D (batch_size,)

        # --- If output is 2D (batch_size, 2), take positive-class logit/probability ---
        if len(y_pred.shape) == 2 and y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]

        # --- Separate positive and negative predictions ---
        pos = tf.boolean_mask(y_pred, tf.equal(y_true, 1))
        neg = tf.boolean_mask(y_pred, tf.equal(y_true, 0))

        # --- Safe pairwise logistic loss ---
        def safe_loss():
            diffs = tf.expand_dims(neg, 0) - tf.expand_dims(pos, 1)
            return tf.reduce_mean(tf.nn.softplus(diffs))  # log(1 + exp(diff))

        # --- Skip batch if only one class present ---
        return tf.cond(
            tf.logical_and(tf.size(pos) > 0, tf.size(neg) > 0),
            safe_loss,
            lambda: 0.0,
        )

    return loss

def build_classifier_proxy(input_dim):
    """
    IgAN Classifier — Architecture faithful to Schena et al. (2020)
    ----------------------------------------------------------------
    - 4 hidden layers, 100 neurons each
    - ELU activation (α=1.0)
    - Batch Normalization after each Dense layer
    - Dropout(0.5) between layers
    - Output: 2 neurons, softmax activation
    """

    model = Sequential(name="IgAN_ProxyAUC_Classifier")

    # ---- Hidden Layer 1 ----
    model.add(Dense(100, input_dim=input_dim, name="Dense_1"))
    model.add(BatchNormalization(name="BN_1"))
    model.add(ELU(alpha=1.0, name="ELU_1"))
    model.add(Dropout(0.5, name="Dropout_1"))

    # ---- Hidden Layer 2 ----
    model.add(Dense(100, name="Dense_2"))
    model.add(BatchNormalization(name="BN_2"))
    model.add(ELU(alpha=1.0, name="ELU_2"))
    model.add(Dropout(0.5, name="Dropout_2"))

    # ---- Hidden Layer 3 ----
    model.add(Dense(100, name="Dense_3"))
    model.add(BatchNormalization(name="BN_3"))
    model.add(ELU(alpha=1.0, name="ELU_3"))
    model.add(Dropout(0.5, name="Dropout_3"))

    # ---- Hidden Layer 4 ----
    model.add(Dense(100, name="Dense_4"))
    model.add(BatchNormalization(name="BN_4"))
    model.add(ELU(alpha=1.0, name="ELU_4"))
    model.add(Dropout(0.5, name="Dropout_4"))

    # ---- Output Layer ----
    model.add(Dense(2, activation="softmax", name="Output"))

    return model

def preprocess_data(df):

    df = pd.read_csv("/work/grana_far2023_fomo/ESKD/Data/final_merge_with_Year_last.csv")
    print(df.head())
    #1. Hypertension (from BP) 
    df["Hypertension"] = ((df["systolic"] >= 140) | (df["Diastolic"] >= 90)).astype(int)

    #2. Therapy (combine 4 therapy attributes)
    therapy_cols = ["RAS blockers", "Immunotherapies", "fish oil", "Tonsillectomy"]

    # Check for existence of the columns
    existing_therapy_cols = [c for c in therapy_cols if c in df.columns]
    if not existing_therapy_cols:
        raise ValueError(f"None of these therapy columns found: {therapy_cols}")

    # Make sure these columns are numeric (0/1)
    df[existing_therapy_cols] = df[existing_therapy_cols].fillna(0).astype(int)

    # Create Therapy = 1 if any therapy type == 1, else 0
    df["Therapy"] = (df[existing_therapy_cols].sum(axis=1) > 0).astype(int)

    # Define baseline features (from paper, excluding BP)
    baseline_features = [
        "age", "SEX", "systolic", "Diastolic","creat_mg_dl", "Uprot",
        "M", "E", "S", "T", "C",
        "Therapy", "Hypertension"
    ]

    # Keep followup_years + outcome separately
    needed_columns = baseline_features + ["years_since_biopsy", "outcome"]

    # 1. Drop rows with missing values in required features
    df_clean = df.dropna(subset=needed_columns).copy()
    df_clean = df[needed_columns].dropna().copy()
    # 2. Encode sex (M=1, F=0)
    df_clean["SEX"] = df_clean["SEX"].map({"M": 1, "F": 0})
    df_clean.head(10)

    return df_clean

df_clean = preprocess_data(pd.DataFrame())

baseline_features = [
    "age", "SEX", "systolic", "Diastolic", "creat_mg_dl", "Uprot",
    "M", "E", "S", "T", "C", "Therapy", "Hypertension"
]

X = df_clean[baseline_features]   # 14 baseline clinical/pathology features
y = df_clean["outcome"]           # Binary outcome: 1 = ESKD, 0 = non-ESKD

# Stratified split to preserve class balance
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
# Attach follow-up info for test subsets
test_set = df_clean.loc[X_test.index, baseline_features + ["years_since_biopsy", "outcome"]].copy()

# Subsets for ≥5 years and ≥10 years (as in the paper’s Figure 3)
test_5y = test_set[test_set["years_since_biopsy"] >= 5]
test_10y = test_set[test_set["years_since_biopsy"] >= 10]

# Extract X and y for ≥5y and ≥10y subsets
X_test_5y = test_5y[baseline_features]
y_test_5y = test_5y["outcome"]

X_test_10y = test_10y[baseline_features]
y_test_10y = test_10y["outcome"]

# Convert to NumPy arrays for Keras compatibility
X_train = np.asarray(X_train, dtype=np.float32)
X_test = np.asarray(X_test, dtype=np.float32)
y_train = np.asarray(y_train, dtype=int).ravel()
y_test = np.asarray(y_test, dtype=int).ravel()

X_test_5y = np.asarray(X_test_5y, dtype=np.float32)
y_test_5y = np.asarray(y_test_5y, dtype=int).ravel()

X_test_10y = np.asarray(X_test_10y, dtype=np.float32)
y_test_10y = np.asarray(y_test_10y, dtype=int).ravel()
# Print diagnostic information
print("Stratified Train-Test Split Completed")
print(f"Train set: {X_train.shape},  Positives = {y_train.sum()}, Negatives = {len(y_train)-y_train.sum()}")
print(f"Test set:  {X_test.shape},  Positives = {y_test.sum()}, Negatives = {len(y_test)-y_test.sum()}")
print(f"≥5y subset:  {X_test_5y.shape},  Positives = {y_test_5y.sum()}, Negatives = {len(y_test_5y)-y_test_5y.sum()}")
print(f"≥10y subset: {X_test_10y.shape},  Positives = {y_test_10y.sum()}, Negatives = {len(y_test_10y)-y_test_10y.sum()}")

# Initialize scaler
scaler = StandardScaler()

# Fit on training data only, then transform both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)
X_test_5y_scaled  = scaler.transform(X_test_5y)
X_test_10y_scaled = scaler.transform(X_test_10y)


def train_with_paper_early_stopping(model, X_tr, y_tr, X_val, y_val,
                                    batch_size=32, max_epochs=2000,
                                    check_interval=5, patience_checks=3):
    """
    Train model with paper-style early stopping (Prechelt rule, 1998).
    Validation loss is checked every 5 epochs instead of every epoch.
    """

    best_val_loss = np.inf
    best_weights = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, max_epochs + 1):
        # Train one epoch
        model.fit(X_tr, y_tr, batch_size=batch_size, epochs=1, verbose=0)

        # Evaluate validation loss only every 5 epochs
        if epoch % check_interval == 0:
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            history["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = model.get_weights()
                no_improve = 0
            else:
                no_improve += 1

            # Stop if validation loss did not improve in N consecutive checks
            if no_improve >= patience_checks:
                
                break

    # Restore best weights
    if best_weights is not None:
        model.set_weights(best_weights)

   
    return model

#  Metric utilities

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return auc, prec, rec, f1, acc


def evaluate_classifier(model, X_eval, y_eval, name="Test", threshold=0.5):
    y_eval = np.ravel(y_eval)
    y_pred = model.predict(X_eval, verbose=0)
    y_prob = tf.nn.softmax(y_pred, axis=1).numpy()[:, 1]
    auc, prec, rec, f1, acc = compute_metrics(y_eval, y_prob, threshold=threshold)
    print(f"{name}:  AUC={auc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Acc={acc:.3f}")
    return auc, prec, rec, f1, acc

#  Training setup
LR0 = 2e-3
DECAY = 3200
BATCH_SIZE = 32
N_SPLITS = 10

X_arr = X_train_scaled.values if hasattr(X_train_scaled, "values") else X_train_scaled
y_arr = y_train.values if hasattr(y_train, "values") else y_train
y_arr = y_arr.astype(int).ravel()

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

results = []
best_model_info = None

# Cross-validation loop

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_arr, y_arr), 1):
    print(f"\n===== Fold {fold} / {N_SPLITS} =====")

    # Split fold data
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y_arr[tr_idx], y_arr[val_idx]

    # --- Z-score normalization (fit only on training fold) ---
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_val = scaler.transform(X_val)

    # --- Learning rate schedule (reset each fold) ---
    lr_schedule = ExponentialDecay(
        initial_learning_rate=LR0,
        decay_steps=DECAY,
        decay_rate=0.96,
        staircase=True
    )

    # --- Build and compile ---
    model = build_classifier_proxy(input_dim=X_tr.shape[1])
    model.compile(optimizer=Adam(learning_rate=lr_schedule),
                  loss=ProxyAUCLoss())

    # --- Train with early stopping (every 5 epochs per Prechelt) ---
    model = train_with_paper_early_stopping(
        model, X_tr, y_tr, X_val, y_val, batch_size=BATCH_SIZE
    )

    # --- Validation metrics ---
    y_pred = model.predict(X_val, verbose=0)
    y_prob = tf.nn.softmax(y_pred, axis=1).numpy()[:, 1]

    auc, prec, rec, f1, acc = compute_metrics(y_val, y_prob, threshold=0.5)
    results.append((fold, auc, prec, rec, f1, acc))
    print(f"Fold {fold} → AUC={auc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Acc={acc:.3f}")

    # --- Select best model by balanced F1 ---
    balance_penalty = abs(prec - rec)
    score = f1 - 0.1 * balance_penalty

    if best_model_info is None or score > best_model_info["score"]:
        best_model_info = {
            "fold": fold,
            "auc": auc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "acc": acc,
            "score": score,
            "weights": model.get_weights(),
            "scaler": scaler
        }

# Summary and reconstruct best model

aucs, precs, recs, f1s, accs = zip(*[(a, p, r, f, acc) for _, a, p, r, f, acc in results])

print("\n===== 10-Fold Balanced Summary =====")
print(f"Mean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
print(f"Mean Precision = {np.mean(precs):.3f} ± {np.std(precs):.3f}")
print(f"Mean Recall = {np.mean(recs):.3f} ± {np.std(recs):.3f}")
print(f"Mean F1 = {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
print(f"Mean Acc = {np.mean(accs):.3f} ± {np.std(accs):.3f}")

print("\n===== Best Balanced Model =====")
print(f"Fold {best_model_info['fold']} → "
      f"AUC={best_model_info['auc']:.3f}, "
      f"Prec={best_model_info['prec']:.3f}, "
      f"Rec={best_model_info['rec']:.3f}, "
      f"F1={best_model_info['f1']:.3f}, "
      f"Acc={best_model_info['acc']:.3f}")

# --- Rebuild model and apply best weights ---
best_model = build_classifier_proxy(input_dim=X_arr.shape[1])
best_model.compile(optimizer=Adam(learning_rate=lr_schedule),
                   loss=ProxyAUCLoss())
best_model.set_weights(best_model_info["weights"])
best_scaler = best_model_info["scaler"]

print("Best model ready in memory (weights + scaler restored).")


def evaluate_classifier(model, X_eval, y_eval, name="Test", threshold=0.5):

    y_eval = np.ravel(y_eval)
    y_pred = model.predict(X_eval, verbose=0)
    
    # Convert logits to probabilities
    y_prob = tf.nn.softmax(y_pred, axis=1).numpy()[:, 1]
    y_class = (y_prob >= threshold).astype(int)

    # Compute metrics
    auc = roc_auc_score(y_eval, y_prob)
    prec = precision_score(y_eval, y_class, zero_division=0)
    rec  = recall_score(y_eval, y_class, zero_division=0)
    f1   = f1_score(y_eval, y_class, zero_division=0)
    acc  = accuracy_score(y_eval, y_class)

    print(f"{name}: AUC={auc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Acc={acc:.3f}")
    return auc, prec, rec, f1, acc

print("\n===== Final Test Evaluation (Best CV Model) =====")

results_full = evaluate_classifier(best_model, X_test_scaled, y_test, name="Full Test")
results_5y   = evaluate_classifier(best_model, X_test_5y_scaled, y_test_5y, name="≥5y Test")
results_10y  = evaluate_classifier(best_model, X_test_10y_scaled, y_test_10y, name="≥10y Test")

