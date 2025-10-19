import os
import json
import torch
import joblib
import tqdm
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import StandardScaler


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


# Preprocessing identico al training
def preprocess_data(df, target_column='TimeToESKD_years'):
    """Usa lo stesso schema del training con 'dateAssess'"""
    if 'dateAssess' not in df.columns:
        raise ValueError("Colonna 'dateAssess' mancante nel file Excel.")

    df['dateAssess'] = pd.to_numeric(df['dateAssess'], errors='coerce')

    # Conversione giorni→anni se serve
    if df['dateAssess'].max() > 50:
        print("'dateAssess' sembra essere in giorni → converto in anni")
        df['dateAssess'] = df['dateAssess'] / 365.25
    else:
        print("'dateAssess' interpretato come anni (nessuna conversione)")

    # Mantieni ultimo record per paziente se CODE presente
    if 'CODE' in df.columns:
        df = df.sort_values(['CODE', 'dateAssess'])
        df = df.groupby('CODE').tail(1)

    # Crea target
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

    return X, y


# Generazione test split (uguale al train) 
def generate_test_split(df, scaler):
    X, y = preprocess_data(df)
    sss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for _, test_idx in sss.split(X, y):
        X_test, y_test = X[test_idx], y[test_idx]

    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=32, shuffle=False)
    return test_loader, X_test_scaled, y_test

def eval_fold(df, save_pth, fold):
    scaler_file = os.path.join(save_pth, f"scaler_fold_{fold}.pkl")
    model_file = os.path.join(save_pth, f"best_model_fold_{fold}.pth")

    if not all(os.path.exists(f) for f in [scaler_file, model_file]):
        print(f"Missing files for fold {fold}, skipping.")
        return None

    scaler = joblib.load(scaler_file)
    test_loader, X_test_scaled, y_test = generate_test_split(df, scaler)

    model = MySimpleRegressorNet(X_test_scaled.shape[1], dropout=0.3)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.to(device)
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for inputs, target in tqdm.tqdm(test_loader, desc=f"Testing fold {fold}"):
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy().flatten())
            labels.extend(target.cpu().numpy().flatten())

    preds, labels = np.array(preds), np.array(labels)
    mse = mean_squared_error(labels, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)

    print(f"Fold {fold}: MAE={mae:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}")
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2), "MSE": float(mse)}


if __name__ == "__main__":
    print("Starting TIME-TO-ESKD regression evaluation...")

    torch.manual_seed(42)
    np.random.seed(42)

    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_minDateAssess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_MINDATA_REGRESSION_TIME_TO_ESKD/'
    save_res_file = os.path.join(save_pth, 'test_results_regression.json')

    df = pd.read_excel(data_path)

    all_results = {}
    for fold in range(1, 11):
        print(f"\n{'='*60}\nEvaluating Fold {fold}/10\n{'='*60}")
        res = eval_fold(df, save_pth, fold)
        if res:
            all_results[f"Fold_{fold}"] = res

    # Salvataggio risultati individuali
    with open(save_res_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nSaved per-fold results to {save_res_file}")

    # Media ± std
    if all_results:
        maes = [r["MAE"] for r in all_results.values()]
        rmses = [r["RMSE"] for r in all_results.values()]
        r2s = [r["R2"] for r in all_results.values()]

        summary = {
            "MAE": f"{np.mean(maes):.4f} ± {np.std(maes):.4f}",
            "RMSE": f"{np.mean(rmses):.4f} ± {np.std(rmses):.4f}",
            "R2": f"{np.mean(r2s):.4f} ± {np.std(r2s):.4f}"
        }

        print("\n===== FINAL TEST PERFORMANCE (across folds) =====")
        for k, v in summary.items():
            print(f"{k}: {v}")

        avg_file = os.path.join(save_pth, 'average_test_results_regression.json')
        with open(avg_file, 'w') as f:
            json.dump(summary, f, indent=4)
        print(f"\nSaved average results to {avg_file}")
