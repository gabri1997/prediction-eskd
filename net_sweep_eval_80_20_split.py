import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import os
import json
import tqdm
import joblib

"""Qua ricarico i valori di normalizzazione del training per fare eval sul test set"""

class SimpleBinaryNN(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super(SimpleBinaryNN, self).__init__()
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
    df['Gender'] = df['Gender'].replace({'M': 0, 'F': 1})
    X = df.drop(columns=['Eskd', 'Code']).values
    y = df['Eskd'].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    return X, y


def eval_fold(df, save_pth, fold):
    
    # Carica lo scaler per questo fold
    scaler_file = os.path.join(save_pth, f"scaler_fold_{fold}.pkl")
    if not os.path.exists(scaler_file):
        print(f"Scaler file not found: {scaler_file}")
        return None
    
    scaler = joblib.load(scaler_file)
    print(f"Loaded scaler for fold {fold}")
    
    # Preprocessing
    X, y = preprocess_data(df)
    # TODO: Exclude dateAccess column if present ? Bha

    # Stesso split 80/20 usato in training
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]
    
    print(f"Test shape: {X_test.shape}, Labels: {y_test.shape}")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")
    
    # Normalizza il test set con lo scaler del training
    X_test_scaled = scaler.transform(X_test)
    
    # Carica configurazione del modello
    config_file = os.path.join(save_pth, f'best_model_fold_{fold}_config.json')
    if not os.path.exists(config_file):
        print(f"Config file not found: {config_file}")
        return None
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    dropout = data['Config Parameters'].get('dropout', 0.1)
    batch_size = data['Config Parameters'].get('batch_size', 32)
    
    # Prepara DataLoader
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Carica modello
    model = SimpleBinaryNN(input_size=X_test_scaled.shape[1], dropout=dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    model_path = os.path.join(save_pth, f'best_model_fold_{fold}.pth')
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded model for fold {fold}")
    print(f"First layer weights (sample): {model.layers[0].weight[0][:5]}")
    
    # Evaluation
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm.tqdm(test_loader, desc=f"Testing fold {fold}"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            all_preds.extend(preds.astype(int).flatten().tolist())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())
    
    # Metriche
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"Fold {fold} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
          f"Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        "Accuracy": float(accuracy),
        "Precision": float(precision),
        "Recall": float(recall),
        "F1 Score": float(f1)
    }


if __name__ == "__main__":
    print("Starting testing script...")
    # Nel train.py hai:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)

    # Nel test.py mancano! Aggiungi all'inizio di eval_fold():
    torch.manual_seed(42)
    np.random.seed(42)
    
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_SWEEP_PARAM/'
    save_res_file = os.path.join(save_pth, 'test_results.json')
    
    df = pd.read_excel(data_path)
    n_folds = 5
    
    all_results = {}
    
    # Evalua ogni fold
    for fold in range(1, n_folds + 1):
        print(f"\n{'='*60}")
        print(f"Evaluating Fold {fold}/{n_folds}")
        print(f"{'='*60}")
        
        fold_results = eval_fold(df, save_pth, fold)
        
        if fold_results is not None:
            all_results[f"Fold {fold}"] = fold_results
        else:
            print(f"Skipping fold {fold} due to missing files")
    
    # Salva risultati individuali
    with open(save_res_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n Individual results saved to {save_res_file}")
    
    # Calcola medie
    if all_results:
        accuracies = [r["Accuracy"] for r in all_results.values()]
        precisions = [r["Precision"] for r in all_results.values()]
        recalls = [r["Recall"] for r in all_results.values()]
        f1s = [r["F1 Score"] for r in all_results.values()]
        
        print(f"\n{'='*60}")
        print("AVERAGE RESULTS ACROSS FOLDS")
        print(f"{'='*60}")
        print(f"Accuracy:  {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
        print(f"Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
        print(f"Recall:    {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
        print(f"F1 Score:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        
        # Salva medie
        avg_results = {
            "Average Test Accuracy": float(np.mean(accuracies)),
            "Average Test Precision": float(np.mean(precisions)),
            "Average Test Recall": float(np.mean(recalls)),
            "Average Test F1 Score": float(np.mean(f1s)),
            "Std Test Accuracy": float(np.std(accuracies)),
            "Std Test Precision": float(np.std(precisions)),
            "Std Test Recall": float(np.std(recalls)),
            "Std Test F1 Score": float(np.std(f1s))
        }
        
        avg_file = os.path.join(save_pth, 'average_test_results.json')
        with open(avg_file, 'w') as f:
            json.dump(avg_results, f, indent=4)
        print(f"  Average results saved to {avg_file}")
    else:
        print(" No results to average - all folds failed")