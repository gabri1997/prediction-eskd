import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
import os
import json
import tqdm
import joblib

"""Qua ricarico i valori di normalizzazione del training per fare eval sul test set"""

class MySimpleBinaryNet(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(MySimpleBinaryNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        return self.layers(x)


def preprocess_data(df):
    # Trasformazioni
    df['Gender'] = df['Gender'].replace({'M': 0, 'F': 1})
    # Voglio togliere anche la colonna Code che non serve dateAssess
    # Voglio combinare le tre colonne Antihypertensive, Immunosuppressive and FishOil in una sola colonna che indica se il paziente assume almeno uno di questi farmaci
    # Creo una nuova colonna 'Therapy' che è 1 se almeno una delle tre colonne è 1, altrimenti 0
    df['Therapy'] = df[['Antihypertensive', 'Immunosuppressants', 'FishOil']].max(axis=1)
    # Ora posso rimuovere le tre colonne originali
    cols_used = [c for c in df.columns if c not in ['Eskd', 'Code', 'dateAssess', 'Antihypertensive', 'Immunosuppressants', 'FishOil']]
    print("\n==============================")
    print("🧩 FEATURE ORDER USED FOR TRAINING / INFERENCE:")
    for i, col in enumerate(cols_used):
        print(f"{i+1:2d}. {col}")
    print("==============================\n")

    X = df[cols_used].values
    X = df.drop(columns=['Eskd', 'Code', 'dateAssess', 'Antihypertensive', 'Immunosuppressants', 'FishOil']).values
    y = df['Eskd'].values

    # Sostituisco NaN e Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y
    

def generate_splits(df, scaler, data, fold):

    print(f"Loaded scaler for fold {fold}")
    dropout = data['Config Parameters'].get('dropout', 0.1)
    batch_size = data['Config Parameters'].get('batch_size', 32)
    # Preprocessing
    X, y = preprocess_data(df)

    # Stesso split 80/20 usato in training
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_test = X[test_idx]
        y_test = y[test_idx]

    df_test = df.iloc[test_idx]  # subset del test set dal DataFrame originale

    mask_5Y = df_test['dateAssess'] >= 5*365
    mask_10Y = df_test['dateAssess'] >= 10*365
    print(f"Number of samples with dateAssess > 5 years: {mask_5Y.sum()}")
    print(f"Number of samples with dateAssess > 10 years: {mask_10Y.sum()}")

    X_test_5Y = X_test[mask_5Y.values]  # .values per convertire in numpy boolean array
    y_test_5Y = y_test[mask_5Y.values]

    X_test_10Y = X_test[mask_10Y.values]
    y_test_10Y = y_test[mask_10Y.values]

    print(f"Test shape with dateAssess > 5 years: {X_test_5Y.shape}, Labels: {y_test_5Y.shape}")
    print(f"Test shape with dateAssess > 10 years: {X_test_10Y.shape}, Labels: {y_test_10Y.shape}")
    
    print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

    # Normalizza il test set con lo scaler del training
    X_test_scaled = scaler.transform(X_test)
    X_test_scaled_5Y = scaler.transform(X_test_5Y)
    X_test_scaled_10Y = scaler.transform(X_test_10Y)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    X_test_tensor_5Y = torch.tensor(X_test_scaled_5Y, dtype=torch.float32)
    y_test_tensor_5Y = torch.tensor(y_test_5Y, dtype=torch.float32).view(-1, 1)
    test_dataset_5Y = TensorDataset(X_test_tensor_5Y, y_test_tensor_5Y)
    test_loader_5Y = DataLoader(test_dataset_5Y, batch_size=batch_size, shuffle=False, num_workers=2)

    X_test_tensor_10Y = torch.tensor(X_test_scaled_10Y, dtype=torch.float32)
    y_test_tensor_10Y = torch.tensor(y_test_10Y, dtype=torch.float32).view(-1, 1)
    test_dataset_10Y = TensorDataset(X_test_tensor_10Y, y_test_tensor_10Y)
    test_loader_10Y = DataLoader(test_dataset_10Y, batch_size=batch_size, shuffle=False, num_workers=2)

    return test_loader, test_loader_5Y, test_loader_10Y, X_test_scaled, y_test, X_test_scaled_5Y, y_test_5Y, X_test_scaled_10Y, y_test_10Y, dropout

def eval_fold(df, save_pth, fold, years=5):
    # Carica scaler e config
    scaler_file = os.path.join(save_pth, f"scaler_fold_{fold}.pkl")
    if not os.path.exists(scaler_file):
        print(f"Scaler file not found: {scaler_file}")
        return None
    scaler = joblib.load(scaler_file)

    config_file = os.path.join(save_pth, f'best_model_fold_{fold}_config.json')
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None
    with open(config_file, 'r') as f:
        data = json.load(f)

    # Ottieni split e dati
    test_loader, test_loader_5Y, test_loader_10Y, X_test_scaled, y_test, X_test_scaled_5Y, y_test_5Y, X_test_scaled_10Y, y_test_10Y, dropout = generate_splits(df, scaler, data, fold)

    # Colonne usate (servono per ricostruire il JSON leggibile)
    cols_used = [c for c in df.columns if c not in ['Eskd', 'Code', 'dateAssess', 'Antihypertensive', 'Immunosuppressants', 'FishOil']]

    # Carica modello
    model = MySimpleBinaryNet(input_size=X_test_scaled.shape[1], dropout=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model_path = os.path.join(save_pth, f'best_model_fold_{fold}.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_probs, all_labels = [], [], []
    positive_samples, negative_samples = [], []

    # Ricreo anche i valori non scalati
    X_unscaled = scaler.inverse_transform(X_test_scaled)

    with torch.no_grad():
        loader_to_use = test_loader
        for batch_idx, (inputs, labels) in enumerate(tqdm.tqdm(loader_to_use, desc=f"Testing fold {fold}")):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())

            # Estrai feature originali e scalate
            inputs_scaled = inputs.cpu().numpy()
            batch_start = batch_idx * loader_to_use.batch_size
            batch_end = batch_start + len(inputs_scaled)
            inputs_unscaled = X_unscaled[batch_start:batch_end]

            for i in range(len(preds)):
                sample_scaled = {col: float(inputs_scaled[i][j]) for j, col in enumerate(cols_used)}
                sample_unscaled = {col: float(inputs_unscaled[i][j]) for j, col in enumerate(cols_used)}

                sample_entry = {
                    "features_scaled": sample_scaled,
                    "features_original": sample_unscaled,
                    "probability": float(probs[i]),
                    "prediction": int(preds[i])
                }

                if preds[i] == 1:
                    positive_samples.append(sample_entry)
                else:
                    negative_samples.append(sample_entry)

    # === Salva solo per il fold 8 ===
    if fold == 8:
        pos_path = os.path.join(save_pth, f"fold_{fold}_positive_features_full.json")
        neg_path = os.path.join(save_pth, f"fold_{fold}_negative_features_full.json")

        with open(pos_path, "w") as f:
            json.dump(positive_samples, f, indent=4)
        with open(neg_path, "w") as f:
            json.dump(negative_samples, f, indent=4)

        print(f"Salvati: {pos_path} e {neg_path}")

    # === METRICHE ===
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = None
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        pass

    cm_values = confusion_matrix(all_labels, all_preds)
    cm_dict = {
        "labels": ["Negativo", "Positivo"],
        "matrix": [" - ".join(map(str, row)) for row in cm_values.tolist()]
    }

    result = {
        "Years Threshold": years,
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1),
        "Confusion Matrix": cm_dict,
        "AUC": float(auc) if auc is not None else None
    }

    return result


if __name__ == "__main__":
    print("Starting testing script...")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # Se vuoi usare il dataset in cui prendo la prima visita metti minDateAssess nel path, altrimenti metti maxDateAssess per prendere l'ultima visita
    # Di conseguenza cambia MAX e MIN nel save_pth per distinguere i due esperimenti
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAssess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_SWEEP_PARAM_ADAM_PROXYLOSS_SAMPLER_NO_ACCESS_SINGLE_SWEEP_THERAPY_CREATININE_SYS_DIAST_MAX_123/'
    save_res_file = os.path.join(save_pth, 'test_results.json')
    
    df = pd.read_excel(data_path)
    n_folds = 10
    
    all_results = {}
    
    # Evalua ogni fold
    for fold in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"Evaluating Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        years = 0 # Default evaluation on all test data
        fold_results = eval_fold(df, save_pth, fold, years)
        
        if fold_results is not None:
            all_results[f"Fold {fold}"] = fold_results
        else:
            print(f"Skipping fold {fold} due to missing files")
    
    # Salva risultati individuali
    with open(save_res_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\nIndividual results saved to {save_res_file}")
    
    # Calcola medie
    if all_results:
        accuracies = [r["Accuracy"] for r in all_results.values()]
        precisions = [r["Precision"] for r in all_results.values()]
        recalls = [r["Recall"] for r in all_results.values()]
        f1s = [r["F1 Score"] for r in all_results.values()]
        aucs = [r["AUC"] for r in all_results.values() if "AUC" in r]
        
        print(f"\n{'='*60}")
        print("AVERAGE RESULTS ACROSS FOLDS")
        print(f"{'='*60}")

        avg_results = {
            "Years Threshold": years,
            "Accuracy": f"{np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}",
            "Precision": f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
            "Recall": f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
            "F1 Score": f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        }
        
        if aucs:
            avg_results["AUC"] = f"{np.mean(aucs):.4f} ± {np.std(aucs):.4f}"

        print(avg_results)

        avg_file = os.path.join(save_pth, f'average_test_results_years_{years}.json')
        with open(avg_file, 'w') as f:
            json.dump(avg_results, f, indent=4, ensure_ascii=False)
        print(f"Average results saved to {avg_file}")
    else:
        print("No results to average - all folds failed")