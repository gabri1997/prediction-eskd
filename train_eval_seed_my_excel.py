import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from model import build_classifier_proxy, ProxyAUCLoss

# ===== IMPOSTAZIONI DETERMINISTICHE =====
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass
# ==========================================


# =========================================================
# 🔧 PREPROCESSING IDENTICO A QUELLO USATO IN PYTORCH
# =========================================================
def preprocess_data(df):
    df = df.copy()

    # Converti dateAssess in anni (era in giorni)
    if "dateAssess" in df.columns:
        df["dateAssess"] = df["dateAssess"] / 365.25
    # Gender mapping
    df['Gender'] = df['Gender'].replace({'M': 0, 'F': 1})

    # Crea colonna Therapy aggregata
    df['Therapy'] = df[['Antihypertensive', 'Immunosuppressants', 'FishOil']].max(axis=1)

    # Rimuovi colonne inutili
    drop_cols = ['Eskd', 'Code', 'dateAssess', 'Antihypertensive', 'Immunosuppressants', 'FishOil']
    features = [c for c in df.columns if c not in drop_cols]

    # Prepara X, y
    X = df[features].to_numpy(dtype=float)
    y = df['Eskd'].to_numpy(dtype=int)

    # Gestione NaN e infiniti
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    print(f"Features usate per il training ({len(features)}): {features}")
    print(f"Shape X={X.shape}, y={y.shape}")

    return df, features


def preparing_data(df_clean, baseline_features):
    X = df_clean[baseline_features]
    y = df_clean["Eskd"]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    test_set = df_clean.loc[X_test.index, baseline_features + ["dateAssess", "Eskd"]]
    test_5y = test_set[test_set["dateAssess"] >= 5]
    test_10y = test_set[test_set["dateAssess"] >= 10]
    print(f"Full test size: {len(test_set)}")
    print(f"≥5y subset size: {len(test_5y)}")
    print(f"≥10y subset size: {len(test_10y)}")
    print("Min/Max dateAssess in test:", test_set['dateAssess'].min(), test_set['dateAssess'].max())


    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    X_test_5y, y_test_5y = test_5y[baseline_features].to_numpy(), test_5y["Eskd"].to_numpy()
    X_test_10y, y_test_10y = test_10y[baseline_features].to_numpy(), test_10y["Eskd"].to_numpy()

    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, X_test_5y, y_test_5y, X_test_10y, y_test_10y


def train_with_paper_early_stopping(model, X_tr, y_tr, X_val, y_val, batch_size=32, max_epochs=2000, check_interval=5, patience_checks=3):
    best_val_loss = np.inf
    best_weights = None
    no_improve = 0

    for epoch in range(1, max_epochs + 1):
        model.fit(X_tr, y_tr, batch_size=batch_size, epochs=1, verbose=0)
        if epoch % check_interval == 0:
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            if val_loss < best_val_loss:
                best_val_loss, best_weights, no_improve = val_loss, model.get_weights(), 0
            else:
                no_improve += 1
            if no_improve >= patience_checks:
                break

    if best_weights is not None:
        model.set_weights(best_weights)
    return model


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return auc, prec, rec, f1, acc


def evaluate_classifier(model, X_eval, y_eval, name="Test", threshold=0.5):
    y_pred = model.predict(X_eval, verbose=0)
    y_prob = tf.nn.softmax(y_pred, axis=1).numpy()[:, 1]
    auc, prec, rec, f1, acc = compute_metrics(y_eval, y_prob, threshold)
    print(f"{name}: AUC={auc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Acc={acc:.3f}")
    return auc, prec, rec, f1, acc


def loop_over_folds(skf, X_train, y_train):
    results = []
    best_model_info = None

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        print(f"\n===== Fold {fold}/{N_SPLITS} =====")

        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        lr_schedule = ExponentialDecay(LR0, DECAY, decay_rate=0.96, staircase=True)
        model = build_classifier_proxy(input_dim=X_tr.shape[1])
        model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=ProxyAUCLoss())

        model = train_with_paper_early_stopping(model, X_tr, y_tr, X_val, y_val, batch_size=BATCH_SIZE)

        y_pred = model.predict(X_val, verbose=0)
        y_prob = tf.nn.softmax(y_pred, axis=1).numpy()[:, 1]
        auc, prec, rec, f1, acc = compute_metrics(y_val, y_prob, threshold=0.731)
        results.append((auc, prec, rec, f1, acc))
        print(f"Fold {fold} → AUC={auc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}, Acc={acc:.3f}")

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

    return results, best_model_info


if __name__ == "__main__":
    data_path = "/work/grana_far2023_fomo/ESKD/Data/final_cleaned_minDateAssess.xlsx"
    df = pd.read_excel(data_path)

    df_clean, baseline_features = preprocess_data(df)
    
    # Fix: sostituisci eventuali NaN residui
    df_clean = df_clean.fillna(0)
    
    X_train, y_train, X_test, y_test, X_test_5y, y_test_5y, X_test_10y, y_test_10y = preparing_data(df_clean, baseline_features)

    X_train, y_train, X_test, y_test, X_test_5y, y_test_5y, X_test_10y, y_test_10y = preparing_data(df_clean, baseline_features)

    LR0 = 2e-3
    DECAY = 3200
    BATCH_SIZE = 32
    N_SPLITS = 10

    lr_schedule = ExponentialDecay(initial_learning_rate=LR0, decay_steps=DECAY, decay_rate=0.96, staircase=True)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    results, best_model_info = loop_over_folds(skf, X_train, y_train)
    aucs, precs, recs, f1s, accs = zip(*results)
    print("\n===== CV Summary =====")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")

    # Normalizzazione test
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test_5y = scaler.transform(X_test_5y)
    X_test_10y = scaler.transform(X_test_10y)

    # Ricostruzione del modello migliore
    best_model = build_classifier_proxy(input_dim=X_test.shape[1])
    best_model.compile(optimizer=Adam(learning_rate=lr_schedule), loss=ProxyAUCLoss())
    best_model.set_weights(best_model_info["weights"])

    evaluate_classifier(best_model, X_test, y_test, "Full Test")
    evaluate_classifier(best_model, X_test_5y, y_test_5y, "≥5y Test")
    evaluate_classifier(best_model, X_test_10y, y_test_10y, "≥10y Test")
