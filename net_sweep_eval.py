import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
import os
import json
import tqdm

# --- Modello come lo indica nel paper anche se fa schifo ---
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
    # Trasformazioni
    df['Gender'] = df['Gender'].replace({'M': 0, 'F': 1})
    X = df.drop(columns=['Eskd', 'Code']).values
    y = df['Eskd'].values

    # Sostituisco NaN e Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y

def evaluate(df, save_pth):

    print("Starting test script...")
    X, y = preprocess_data(df)
    print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")

    # --- 5-fold stratified CV ---
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Per ogni fold devo aprire il json corrispondente nel path save_pth e leggere i parametri per inizializzare il modello
    for fold in range(1,6):
        print(f"Evaluating fold {fold}...")
        with open(os.path.join(save_pth, f'best_model_fold_{fold}_config.json'), 'r') as f:
            config = json.load(f)
        
        # Estraggo i parametri
        dropout = config['Config Parameters']['dropout']
        batch_size = config['Config Parameters']['batch_size']

        # Inizializzo il modello
        model = SimpleBinaryNN(input_size=X.shape[1], dropout=dropout)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model_path = os.path.join(save_pth, f'best_model_fold_{fold}.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded from {model_path}")

        splits = list(kf.split(X, y))
    
        train_idx, val_idx = splits[fold -1]
        _, X_val = X[train_idx], X[val_idx]
        _, y_val = y[train_idx], y[val_idx]

        # Z-score normalization
        scaler = StandardScaler()
        X_val = scaler.fit_transform(X_val)    
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        y_pred = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                y_pred.extend(torch.sigmoid(outputs).cpu().numpy())
        y_pred = (np.array(y_pred).flatten() > 0.5).astype(int)
        
        # Metriche fold
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        print(f"Fold {fold}: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        print("-"*50)
        # salvo le metriche in un file json
        metrics = {
            "Fold": fold,
            "Metrics": {
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1
            }           
        }
        with open(os.path.join(save_pth, f'best_model_fold_{fold}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4) 
if __name__ == "__main__":

    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models/'
    df = pd.read_excel(data_path)
    evaluate(df, save_pth)