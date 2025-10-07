
import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- FLAG GLOBALE PER WANDB ---
USE_WANDB = False
if USE_WANDB:
    import wandb


class SimpleBinaryNN(nn.Module):
    def __init__(self, input_size, dropout=0.1):
        super().__init__()
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


def wandb_function():
    if not USE_WANDB:
        return
    wandb.define_metric("epoch")
    for m in ["train_loss","train_accuracy","train_precision","train_recall","train_f1_score",
              "learning_rate","val_loss","val_accuracy","val_precision","val_recall_score",
              "val_f1_score","val_auc_roc"]:
        wandb.define_metric(m, step_metric="epoch")


def model_initialization(X_train, y_train, config=None):
    dropout = getattr(config, 'dropout', 0.1) if config else 0.1
    optimizer_name = getattr(config, 'optimizer', 'adam') if config else 'adam'
    learning_rate = getattr(config, 'learning_rate', 0.001) if config else 0.001
    
    model = SimpleBinaryNN(X_train.shape[1], dropout=dropout)
    
    # ExponentialLR
    initial_lr, final_lr = 0.2, 1e-6
    step_start, step_end = 400, 3200
    gamma = (final_lr / initial_lr) ** (1 / (step_end - step_start))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) if optimizer_name=="adam" else torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    pos_weight = torch.tensor([len(y_train[y_train==0])/len(y_train[y_train==1])])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    return model, optimizer, scheduler, criterion, step_start, step_end


def read_json(save_file):
    if os.path.exists(save_file):
        try:
            with open(save_file, 'r') as f:
                data = json.load(f)
            return data.get("Best Model Metrics", {}).get("Best F1", 0.0)
        except:
            return 0.0
    else:
        with open(save_file, 'w') as f:
            json.dump({}, f)
        return 0.0


class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose: print(f"Validation loss improved to {val_loss:.6f}.")
            return False
        else:
            self.counter += 1
            if self.verbose: print(f"No improvement. Counter: {self.counter}/{self.patience}")
            return self.counter >= self.patience


def train(df, num_epochs, save_pth, save_on_evaluation=True, early_stop=True):
    torch.manual_seed(42)
    np.random.seed(42)

    X, y = preprocess_data(df)
    
    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in sss.split(X, y):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

    print(f"Training+val: {X_train_val.shape}, Test: {X_test.shape}")
    print(f"Training distribution: {dict(zip(*np.unique(y_train_val, return_counts=True)))}")
    print(f"Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # K-Fold
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_train_val, y_train_val))
    
    if USE_WANDB:
        wandb.init()
        config = wandb.config
        fold_to_run = getattr(config, 'fold', 1)
    else:
        config = None
        fold_to_run = 1

    train_idx, val_idx = splits[fold_to_run-1]
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    model, optimizer, scheduler, criterion, step_start, step_end = model_initialization(X_train, y_train, config=config)
    scaler = StandardScaler()
    X_train, X_val = scaler.fit_transform(X_train), scaler.transform(X_val)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                            torch.tensor(y_train, dtype=torch.float32).view(-1,1)),
                              batch_size=getattr(config,'batch_size',32) if config else 32,
                              shuffle=True)
    
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                          torch.tensor(y_val, dtype=torch.float32).view(-1,1)),
                            batch_size=getattr(config,'batch_size',32) if config else 32,
                            shuffle=False)
    
    early_stopping = EarlyStopping(verbose=True) if early_stop else None
    save_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}_config.json")
    model_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}.pth")
    
    wandb_function()
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        all_labels, all_preds, running_loss = [], [], 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if step_start <= global_step <= step_end:
                scheduler.step()
            global_step += 1
            running_loss += loss.item() * inputs.size(0)
            all_labels.append(labels)
            all_preds.append(torch.sigmoid(outputs).detach())

        all_labels, all_preds = torch.cat(all_labels), torch.cat(all_preds)
        all_preds_bin = (all_preds > 0.5).float()
        train_acc = (all_preds_bin == all_labels).float().mean()
        train_f1 = f1_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0)
        epoch_loss = running_loss / len(train_loader.dataset)

        if USE_WANDB:
            wandb.log({"train_loss": epoch_loss, "train_accuracy": train_acc, "train_f1_score": train_f1, "epoch": epoch})

        if save_on_evaluation:
            model.eval()
            val_labels, val_logits = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)       # logit grezzi
                    val_labels.append(labels)
                    val_logits.append(outputs)

            val_labels = torch.cat(val_labels)
            val_logits = torch.cat(val_logits)
            
            # Per le metriche trasformiamo in probabilità
            val_preds = torch.sigmoid(val_logits)
            val_preds_bin = (val_preds > 0.5).float()
            
            val_acc = (val_preds_bin == val_labels).float().mean()
            val_f1 = f1_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0)

            # Loss BCEWithLogitsLoss va calcolata sui logit, non sulle probabilità
            val_loss = criterion(val_logits, val_labels).item()

            
            if USE_WANDB:
                wandb.log({"val_loss": val_loss, "val_accuracy": val_acc, "val_f1_score": val_f1, "epoch": epoch})

            if early_stopping and early_stopping.early_stop(val_loss):
                print("Early stopping triggered")
                break

            if val_f1 > read_json(save_file):
                torch.save(model.state_dict(), model_file)
                if USE_WANDB:
                    wandb.log({"best_model_saved": True, "epoch": epoch})

    if USE_WANDB: wandb.finish()


if __name__ == "__main__":
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_SWEEP_PARAM/'
    df = pd.read_excel(data_path)
    train(df, num_epochs=120, save_pth=save_pth, save_on_evaluation=True, early_stop=True)

