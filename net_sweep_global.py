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
    df['Gender'] = df['Gender'].replace({'M': 1, 'F': 2})
    X = df.drop(columns=['Eskd', 'Code']).values
    y = df['Eskd'].values

    # Sostituisco NaN e Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)

    return X, y

def wandb_function(fold_to_run):
    # import time
    # fold = wandb.config.fold
    # run_name = f"ESKD_fold_{fold}_{int(time.time())}"
    # wandb.init(project="ESKD_NN", name=run_name, reinit=True)
    
    wandb.define_metric("epoch")
    wandb.define_metric("train_loss", step_metric="epoch")
    wandb.define_metric("train_accuracy", step_metric="epoch")
    wandb.define_metric("train_precision", step_metric="epoch")
    wandb.define_metric("train_recall", step_metric="epoch")
    wandb.define_metric("train_f1_score", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")


def model_initialization(X_train, y_train):

    config = wandb.config

    model = SimpleBinaryNN(X_train.shape[1],  dropout=config.dropout if 'dropout' in config else 0.1)

    # Questa roba serve per l'ExponentialLR, il LR inizia a diminuire dopo 400 step e finisce di diminuire a 3200 step
    initial_lr = 0.2
    final_lr = 0.000001
    step_start = 400
    step_end = 3200 
    N = step_end - step_start
    # Qua c'è la formula per calcolare il gamma
    gamma = (final_lr / initial_lr) ** (1 / N)

    if config.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9)

    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    return model, optimizer, scheduler, criterion, step_start, step_end

def read_json (save_file):
    if os.path.exists(save_file):
        try:
            with open(save_file, 'r') as f:
                data = json.load(f)
            best_f1 = data.get("Best Model Metrics", {}).get("Best F1", 0.0)
        except:
            best_f1 = 0.0
    else:
        best_f1 = 0.0
        with open(save_file, 'w') as f:
            json.dump({}, f)
    return best_f1


def train_and_evaluate(df, num_epochs, save_pth):

    X, y = preprocess_data(df)
    print(f"Feature shape: {X.shape}, Labels shape: {y.shape}")

    # --- 10-fold stratified CV ---
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_metrics = []

    splits = list(kf.split(X, y))
    print(f"Numero di split: {len(splits)}")
    #  Dal file sweep.py prendo il valore del fold che eseguo
    wandb.init()
    fold_to_run = wandb.config.fold

    train_idx, val_idx = splits[fold_to_run - 1]
    print(f" === Running fold {fold_to_run} with {len(train_idx)} training samples and {len(val_idx)} validation samples ===")
    
    wandb_function(fold_to_run)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model, optimizer, scheduler, criterion, step_start, step_end = model_initialization(X_train, y_train)

    # Z-score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    
    # Tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=2)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=2)
    
    # Training loop
    global_step = 0
    
    save_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}_config.json")
    model_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}.pth")

    for epoch in range(num_epochs):
        
        all_labels = []
        all_preds = []
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            if global_step >= step_start and global_step <= step_end:
                # Metto qui dentro lo step dello scheduler così si aggiorna ogni batch
                scheduler.step()
            global_step += 1
            
            running_loss += loss.item() * inputs.size(0)
            all_labels.append(labels)
            all_preds.append(torch.sigmoid(outputs).detach())

        all_labels = torch.cat(all_labels)
        all_preds = torch.cat(all_preds)
            
        all_preds_bin = (all_preds > 0.5).float()
        train_acc = (all_preds_bin == all_labels).float().mean()
        train_f1_score = f1_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0)
       
        best_f1 = read_json(save_file)
        print(f"Current best F1 from file: {best_f1}, Current epoch F1: {train_f1_score}")
        if train_f1_score > best_f1:
            best_f1 =  train_f1_score
            # Salvo il modello
            print(f"New best model with F1: {best_f1}, saving model...")
            torch.save(model.state_dict(), model_file)
            # In un file a parte salvo anche la configurazione di parametri usata e le metriche raggiunte con questa configurazione, voglio sovrascrivere il file ogni volta che trovo un modello migliore
            config_and_metrics = {
                "Config Parameters": dict(wandb.config),
                "Best Model Metrics": {
                    "Epoch": epoch + 1,
                    "Best F1": float(best_f1),
                    "Train Accuracy": float(train_acc),
                    "Train Precision": float(precision_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0)),
                    "Train Recall": float(recall_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0))
                }
            }
            with open(save_file, 'w') as f:
                json.dump(config_and_metrics, f, indent=4)

        epoch_loss = running_loss / len(train_loader.dataset)
        #print(f"Training Loss: {epoch_loss:.4f}")

        try :
            wandb.log({"train/loss" : epoch_loss, 'epoch': epoch}),
            wandb.log({"train/accuracy" : train_acc, 'epoch': epoch}),
            wandb.log({"train/precision": precision_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0), 'epoch' : epoch}),
            wandb.log({"train/recall-score" : recall_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0), 'epoch' : epoch}),
            wandb.log({"train/f1-score" : f1_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0), 'epoch' : epoch}),
            wandb.log({"learning_rate" : optimizer.param_groups[0]['lr'], 'epoch' : epoch})  
        except Exception as e:
            print(f'Errore nel log di wandb nel train: {e}')
        
    # Validazione
    # model.eval()
    # y_pred = []
    # with torch.no_grad():
    #     # Ricarico i pesi del miglior modello salvato per ogni fold specifico
    #     print("Loading best model for validation... for the fold ", fold_to_run)
    #     model.load_state_dict(torch.load(os.path.join(save_pth, f"best_model_fold_{fold_to_run}.pth")))
    #     for inputs, _ in val_loader:
    #         outputs = model(inputs)
    #         y_pred.extend(torch.sigmoid(outputs).cpu().numpy())
    # y_pred = (np.array(y_pred).flatten() > 0.5).astype(int)
    
    # # Metriche fold
    # acc = accuracy_score(y_val, y_pred)
    # prec = precision_score(y_val, y_pred)
    # rec = recall_score(y_val, y_pred)
    # f1 = f1_score(y_val, y_pred)
    # print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    # fold_metrics.append([acc, prec, rec, f1])

    wandb.finish()
    print("Training completato.")

    # Media dei 10 fold
    # fold_metrics = np.array(fold_metrics)
    # print("\n=== Media 10-fold ===")
    # print(f"Accuracy: {fold_metrics[:,0].mean():.4f}")
    # print(f"Precision: {fold_metrics[:,1].mean():.4f}")
    # print(f"Recall: {fold_metrics[:,2].mean():.4f}")
    # print(f"F1-score: {fold_metrics[:,3].mean():.4f}")


if __name__ == "__main__":
  
    print("Starting training script...")
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models/'
    df = pd.read_excel(data_path)
    num_epochs = 100
    train_and_evaluate(df, num_epochs, save_pth)