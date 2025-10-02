"""

Questo script esegue il test della rete allenata con la 10 fold cross validation sull'80% dei dati, quindi il test viene eseguito sul 20% dei dati.
Il modello viene caricato da disco e i risultati vengono salvati in un file Json.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import os
import json
import tqdm


# --- Modello come lo indica nel paper anche se non si affronta ---
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


def eval(df, save_pth, save_res_file, n_folds):

    X, y = preprocess_data(df)

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in sss.split(X, y):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

    print(f"Training + validation shape: {X_train_val.shape}, Labels shape: {y_train_val.shape}")
    print(f"Test shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # Controllo distribuzione classi
    unique_train, counts_train = np.unique(y_train_val, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Training class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

    scaler = StandardScaler()
    scaler.fit(X_train_val)
    X_test = scaler.transform(X_test)

    # Ora devo prendere il 20 % dei dati che non ho usato per il training e validazione e fare il test
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1,1)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
             

    # Leggo il file json per vedere se c'è un best f1 salvato
    all_results = {}
    for fold in range(1, n_folds + 1):
        for folder in sorted(os.listdir(save_pth)):
        # Inizializzo il modello
        # Devo aprire il file json per vedere quale fold ha il best f1
            if folder.startswith(f'best_model_fold_{fold}_config.json'):
                with open(os.path.join(save_pth, folder), 'r') as f:
                    data = json.load(f)
                dropout = data['Config Parameters']['dropout']
                # So che posso usare un batch size diverso perchè qui non alleno ma vabè
                batch_size = data['Config Parameters'].get('batch_size', 32)
                # Carico il modello corrispondente
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)   
                model = SimpleBinaryNN(input_size=X.shape[1], dropout=dropout)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model.to(device)
                model_path = os.path.join(save_pth, f'best_model_fold_{fold}.pth')
                model.load_state_dict(torch.load(model_path, map_location=device))

                # Ora faccio il test sul 20% dei dati
                model.eval()
                all_preds = []      
                all_labels = []
                with torch.no_grad():
                    for inputs, labels in tqdm.tqdm(test_loader, desc="Testing"):
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        preds = torch.sigmoid(outputs).cpu().numpy() > 0.5
                        all_preds.extend(preds.astype(int).flatten().tolist())
                        all_labels.extend(labels.cpu().numpy().astype(int).flatten().tolist())
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds, zero_division=0)
                recall = recall_score(all_labels, all_preds, zero_division=0)
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                print(f"Test Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
                # Salvo i risultati in un file json
                all_results[f"Fold {fold}"] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                }
            else:
                continue
        
    with open(save_res_file, 'w') as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    # Path al file CSV
    print("Starting testing script...")
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_save_on_eval/'
    save_res_file = os.path.join(save_pth, 'test_results.json')
    df = pd.read_excel(data_path)
    n_folds = 5
    eval(df, save_pth, save_res_file, n_folds)
    # Calcolo la media dei risultati
    with open(save_res_file, 'r') as f:
        results = json.load(f)
    accuracies = [results[fold]["Accuracy"] for fold in results]
    precisions = [results[fold]["Precision"] for fold in results]
    recalls = [results[fold]["Recall"] for fold in results]
    f1s = [results[fold]["F1 Score"] for fold in results]
    print(f"Average Test Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
    print(f"Average Test Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
    print(f"Average Test Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    print(f"Average Test F1 Score: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}") 
    # Salvo la media dei risultati in un file json
    avg_results = {
        "Average Test Accuracy": np.mean(accuracies),
        "Average Test Precision": np.mean(precisions),
        "Average Test Recall": np.mean(recalls),
        "Average Test F1 Score": np.mean(f1s),
        "Std Test Accuracy": np.std(accuracies),
        "Std Test Precision": np.std(precisions),
        "Std Test Recall": np.std(recalls),
        "Std Test F1 Score": np.std(f1s)
    }
    with open(os.path.join(save_pth, 'average_test_results.json'), 'w') as f:
        json.dump(avg_results, f, indent=4) 
    