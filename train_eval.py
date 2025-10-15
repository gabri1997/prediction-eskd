# train_eval.py
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from processing import preprocess_data
from model import build_classifier_proxy, ProxyAUCLoss
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def preparing_data(df_clean, baseline_features):
    X = df_clean[baseline_features]
    y = df_clean["outcome"]

    # Split train/test stratificato 
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Sottogruppi follow-up 
    test_set = df_clean.loc[X_test.index, baseline_features + ["years_since_biopsy", "outcome"]]
    test_5y = test_set[test_set["years_since_biopsy"] >= 5]
    test_10y = test_set[test_set["years_since_biopsy"] >= 10]

    # Conversione in array 
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()
    X_test_5y, y_test_5y = test_5y[baseline_features].to_numpy(), test_5y["outcome"].to_numpy()
    X_test_10y, y_test_10y = test_10y[baseline_features].to_numpy(), test_10y["outcome"].to_numpy()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test, y_test, X_test_5y, y_test_5y, X_test_10y, y_test_10y


def train_with_paper_early_stopping(model, X_tr, y_tr, X_val, y_val,batch_size=32, max_epochs=2000, check_interval=5, patience_checks=3):

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
        print("..........")

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
        metrics = compute_metrics(y_val, y_prob)
        results.append(metrics)
    
    return results, model



if __name__ == "__main__":

    # Caricamento e preprocessing 
    csv_path = "/work/grana_far2023_fomo/ESKD/Data/final_merge_with_Year_last.csv"
    df_clean, baseline_features = preprocess_data(csv_path)
    X_train, y_train, X_test, y_test, X_test_5y, y_test_5y, X_test_10y, y_test_10y = preparing_data(df_clean, baseline_features)
    LR0 = 2e-3
    DECAY = 3200
    BATCH_SIZE = 32
    N_SPLITS = 10
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    results, model = loop_over_folds(skf, X_train, y_train)
    aucs, precs, recs, f1s, accs = zip(*results)
    print("\n===== CV Summary =====")
    print(f"AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"F1: {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test_5y = scaler.transform(X_test_5y)
    X_test_10y = scaler.transform(X_test_10y)
    evaluate_classifier(model, X_test, y_test, "Full Test")
    evaluate_classifier(model, X_test_5y, y_test_5y, "≥5y Test")
    evaluate_classifier(model, X_test_10y, y_test_10y, "≥10y Test")
