import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class MySimpleRegressorNet(nn.Module):
    def __init__(self, input_size, dropout=0.3):
        super(MySimpleRegressorNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 125),
            nn.BatchNorm1d(125),
            nn.SELU(),
            nn.Dropout(0.5),

            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(125, 125),
            nn.BatchNorm1d(125),
            nn.SELU(),
            nn.Dropout(0.3),

            nn.Linear(125, 1)
        )

    def forward(self, x):
        output = self.layers(x)
        return torch.clamp(output, min=0.0, max=10.0)


def count_eskd_patients(df):
    if 'Eskd' not in df.columns:
        raise ValueError("Colonna 'Eskd' non trovata nel dataset.")

    total = len(df)
    eskd_count = (df['Eskd'] == 1).sum()
    eskd_percentage = (eskd_count / total) * 100 if total > 0 else 0

    print(f"Pazienti totali: {total}")
    print(f"Pazienti con ESKD = 1: {eskd_count} ({eskd_percentage:.2f}%)")

    return eskd_count, total, eskd_percentage


def preprocess_data(df, target_column='TimeToESKD_years'):
    if 'dateAssess' not in df.columns:
        raise ValueError("Colonna 'dateAssess' non trovata.")
    if 'Eskd' not in df.columns:
        raise ValueError("Colonna 'Eskd' non trovata.")

    df['dateAssess'] = pd.to_numeric(df['dateAssess'], errors='coerce')

    if df['dateAssess'].max() > 50:
        print("'dateAssess' sembra essere in giorni → converto in anni")
        df['dateAssess'] = df['dateAssess'] / 365.25
    else:
        print("'dateAssess' interpretato come anni (nessuna conversione)")

    if 'CODE' in df.columns:
        df = df.sort_values(['CODE', 'dateAssess'])
        df = df.groupby('CODE').tail(1)

    df['TimeToESKD_years'] = np.nan
    df.loc[df['Eskd'] == 1, 'TimeToESKD_years'] = df.loc[df['Eskd'] == 1, 'dateAssess']
    df_with_event = df[df['Eskd'] == 1].copy()

    for sex_col in ['SEX', 'Gender']:
        if sex_col in df_with_event.columns:
            df_with_event[sex_col] = df_with_event[sex_col].replace({'M': 0, 'F': 1, 'm': 0, 'f': 1})

    cols_to_drop = ['Eskd', 'dateAssess', target_column, 'CODE']
    cols_to_drop = [c for c in cols_to_drop if c in df_with_event.columns]

    X = df_with_event.drop(columns=cols_to_drop).select_dtypes(include=[np.number]).fillna(0).values
    y = df_with_event[target_column].fillna(0).values

    print(f"Pazienti totali: {len(df)}, con ESKD: {len(df_with_event)}")
    print(f"Feature shape: {X.shape}, Target shape: {y.shape}")

    return X, y


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
            return False
        else:
            self.counter += 1
            if self.verbose:
                print(f"No improvement. Patience {self.counter}/{self.patience}")
            return self.counter >= self.patience


def train(df, num_epochs, save_pth, loss_fn='mse'):
    print("Starting 10-FOLD training for TIME-TO-ESKD regression...")
    torch.manual_seed(123)
    np.random.seed(123)

    X, y = preprocess_data(df)

    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
    for train_idx, test_idx in sss.split(X, y):
        X_train_val, y_train_val = X[train_idx], y[train_idx]

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    os.makedirs(save_pth, exist_ok=True)
    fold_metrics = []

    learning_rate = 1e-3
    batch_size = 32
    dropout = 0.3
    optimizer_name = "adam"

    for fold_to_run, (train_idx, val_idx) in enumerate(kf.split(X_train_val), 1):
        print(f"\n===== FOLD {fold_to_run}/10 =====")

        X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
        y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

        model = MySimpleRegressorNet(X_train.shape[1], dropout=dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
        criterion = nn.MSELoss() if loss_fn == 'mse' else nn.L1Loss()

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        joblib.dump(scaler, os.path.join(save_pth, f"scaler_fold_{fold_to_run}.pkl"))

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False)

        early_stopping = EarlyStopping(patience=20, verbose=True)
        best_mae = float("inf")
        best_metrics = {}
        model_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}.pth")

        for epoch in range(num_epochs):
            model.train()
            preds, labels = [], []

            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                preds.extend(outputs.detach().numpy().flatten())
                labels.extend(targets.numpy().flatten())

            train_mae = mean_absolute_error(labels, preds)

            model.eval()
            with torch.no_grad():
                val_preds, val_labels = [], []
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    val_preds.extend(outputs.numpy().flatten())
                    val_labels.extend(targets.numpy().flatten())

            val_mae = mean_absolute_error(val_labels, val_preds)
            val_rmse = np.sqrt(mean_squared_error(val_labels, val_preds))
            val_r2 = r2_score(val_labels, val_preds)

            print(f"Epoch {epoch+1}/{num_epochs} | Val MAE={val_mae:.3f} | RMSE={val_rmse:.3f} | R2={val_r2:.3f}")

            if val_mae < best_mae:
                best_mae = val_mae
                best_metrics = {"MAE": val_mae, "RMSE": val_rmse, "R2": val_r2, "Epoch": epoch + 1}
                torch.save(model.state_dict(), model_file)

            if early_stopping.early_stop(val_mae):
                print("Early stopping triggered.")
                break

        # Salva file JSON per singolo fold
        fold_results = {
            "Fold": fold_to_run,
            "Best_Validation": best_metrics,
            "Training_Settings": {
                "Epochs_Run": epoch + 1,
                "Learning_Rate": learning_rate,
                "Batch_Size": batch_size,
                "Dropout": dropout,
                "Loss_Function": loss_fn
            }
        }
        
        def to_native(obj):
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: to_native(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_native(v) for v in obj]
            return obj

        fold_results_native = to_native(fold_results)

        fold_json_path = os.path.join(save_pth, f"best_model_fold_{fold_to_run}_results.json")
        with open(fold_json_path, "w") as f:
            json.dump(fold_results_native, f, indent=4)

        fold_metrics.append(best_metrics)

    mae_vals = [m["MAE"] for m in fold_metrics]
    rmse_vals = [m["RMSE"] for m in fold_metrics]
    r2_vals = [m["R2"] for m in fold_metrics]

    summary = {
        "Mean_MAE": float(np.mean(mae_vals)),
        "Std_MAE": float(np.std(mae_vals)),
        "Mean_RMSE": float(np.mean(rmse_vals)),
        "Std_RMSE": float(np.std(rmse_vals)),
        "Mean_R2": float(np.mean(r2_vals)),
        "Std_R2": float(np.std(r2_vals)),
        "Fold_Result_Files": [f"best_model_fold_{i+1}_results.json" for i in range(len(fold_metrics))]
    }

    print("\n===== CROSS-VALIDATION SUMMARY =====")
    print(json.dumps(summary, indent=4))

    with open(os.path.join(save_pth, "crossval_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    print("Starting TIME-TO-ESKD regression training...")
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_minDateAssess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_MINDATA_REGRESSION_TIME_TO_ESKD_MIN_123/'
    os.makedirs(save_pth, exist_ok=True)

    df = pd.read_excel(data_path)
    num_epochs = 3000
    count_eskd_patients(df)
    train(df, num_epochs, save_pth, loss_fn='mse')
