import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Caricamento dati ---
data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
df = pd.read_excel(data_path)

# Trasformazioni
df['Gender'] = df['Gender'].replace({'M': 1, 'F': 2})
X = df.drop(columns=['Eskd', 'Code']).values
y = df['Eskd'].values

# Sostituisco NaN e Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0)

# --- Modello ---
class SimpleBinaryNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleBinaryNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.2),
            
            nn.Linear(100, 1)
        )
        
    def forward(self, x):
        return self.layers(x)

# --- 10-fold stratified CV ---
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

fold_metrics = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n=== Fold {fold} ===")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Z-score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Modello, loss, optimizer, scheduler
    model = SimpleBinaryNN(X_train.shape[1])

    # Questa roba serve per l'ExponentialLR, il LR inizia a diminuire dopo 400 step e finisce di diminuire a 3200 step
    initial_lr = 0.2
    final_lr = 0.000001
    step_start = 400
    step_end = 3200
    N = step_end - step_start
    # Qua c'è la formula per calcolare il gamma
    gamma = (final_lr / initial_lr) ** (1 / N)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    
    # Training
    global_step = 0
    num_epochs = 150
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            if global_step >= step_start and global_step <= step_end:
                # Metto qui dentro lo step dello scheduler così si aggiorna ogni batch
                scheduler.step()
            global_step += 1
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            
        
    # Validazione
    model.eval()
    y_pred = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            outputs = model(inputs)
            y_pred.extend(torch.sigmoid(outputs).cpu().numpy())
    y_pred = (np.array(y_pred).flatten() > 0.5).astype(int)
    
    # Metriche fold
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    fold_metrics.append([acc, prec, rec, f1])

# Media dei 10 fold
fold_metrics = np.array(fold_metrics)
print("\n=== Media 10-fold ===")
print(f"Accuracy: {fold_metrics[:,0].mean():.4f}")
print(f"Precision: {fold_metrics[:,1].mean():.4f}")
print(f"Recall: {fold_metrics[:,2].mean():.4f}")
print(f"F1-score: {fold_metrics[:,3].mean():.4f}")
