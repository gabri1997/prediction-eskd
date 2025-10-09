import os
import json
import wandb
import torch
import joblib
import itertools 
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import WeightedRandomSampler
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# --- Modello come lo indica nel paper anche se non si affronta ---
class MySimpleRegressorNet(nn.Module):
    def __init__(self, input_size, dropout=0.1):
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
        return self.layers(x)

def preprocess_data(df, target_column='TimeToESKD_years'):
    # Calcola il target per TUTTI
    df['TimeToESKD_days'] = (df['DateLastVisit'] - df['DateFirstBiopsy']).dt.days
    df['TimeToESKD_years'] = df['TimeToESKD_days'] / 365.25
    
    # Filtra solo chi ha sviluppato ESKD (evento osservato)
    df_with_event = df[df['Eskd'] == 1].copy()
    
    print(f"Pazienti totali: {len(df)}")
    print(f"Pazienti con ESKD (per regressione): {len(df_with_event)}")
    
    # Preprocessing
    df_with_event['Gender'] = df_with_event['Gender'].replace({'M': 0, 'F': 1})
    X = df_with_event.drop(columns=[target_column, 'Eskd', 'Code', 
                                     'DateFirstBiopsy', 'DateLastVisit', 
                                     'TimeToESKD_days']).values
    y = df_with_event[target_column].values
    
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0)
    
    return X, y

# Proxy AUC Loss Function
# Io avevo messo la Auc normale, non è deffirenziabile quindi fuori dal grafo computazionale
# Ricordiamoci che l'Auc misura fondamentalmente la capacità del modello di separare i positivi dai negativi
# Cioè la probabilità che un positivo sia classificato con un punteggio più alto di un negativo
class ProxyAUCLoss(nn.Module):
    import torch
import torch.nn as nn

class ProxyAUCLoss(nn.Module):
    """
    Proxy AUC Loss — versione stabile per batch sbilanciati o con una sola classe.
    Approssima l'AUC con una sigmoide differenziabile e gestisce i casi limite.
    """
    def __init__(self, eps=1e-8):
        super(ProxyAUCLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.eps = eps

    def forward(self, outputs, targets):
        outputs = outputs.view(-1)
        targets = targets.view(-1)

        pos = outputs[targets == 1]
        neg = outputs[targets == 0]

        # Se in batch non ci sono sia positivi che negativi → ritorna loss neutra
        if len(pos) == 0 or len(neg) == 0:
            return torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device, requires_grad=True)

        # Calcolo differenze e media sigmoide
        diff = pos.unsqueeze(1) - neg.unsqueeze(0)
        auc_est = self.sigmoid(diff).mean()

        # Calcolo della loss (1 - AUC approssimata)
        loss = 1.0 - auc_est

        # Clamp per stabilità numerica (evita NaN o inf)
        loss = torch.clamp(loss, min=0.0, max=1.0)

        if torch.isnan(loss) or torch.isinf(loss):
            loss = torch.tensor(0.0, dtype=outputs.dtype, device=outputs.device, requires_grad=True)

        return loss


def model_initialization(X_train, y_train, loss_fn='proxy_auc'):

    config = wandb.config

    model = MySimpleRegressorNet(X_train.shape[1],  dropout=config.dropout if 'dropout' in config else 0.1)
    # Questa roba serve per l'ExponentialLR, il LR inizia a diminuire dopo 400 step e finisce di diminuire a 3200 step
    initial_lr = getattr(config, 'learning_rate', 0.001)
    final_lr = 1e-6
    # Qui da indicazioni da paper, però non serve complicare troppo
    step_start = 400
    step_end = 3200 
    N = step_end - step_start
    # Qua c'è la formula per calcolare il gamma
    gamma = (final_lr / initial_lr) ** (1 / N)
    
    optimizer_name = getattr(config, 'optimizer', 'adam')  # il default è Adam      

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    pos_weight = torch.tensor([len(y_train[y_train==0]) / len(y_train[y_train==1])])

    # Qui posso usare la Loss che voglio, la Proxy AUC-ROC
    if loss_fn == 'proxy_auc':
        print("Using Proxy AUC Loss Function")
        criterion = ProxyAUCLoss()
    else:
        # Loro non usano questa loss function, ma la proxy AUC-ROC che non è implementata in PyTorch
        print("Using BCEWithLogitsLoss Function")
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

class EarlyStopping:
    def __init__(self, patience = 2, min_delta = 0.0, verbose = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss + self.min_delta < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose == True:
                print(f"Validation loss improved to {val_loss:.6f}.")
            return False
        else:   
            self.counter += 1
            if self.verbose == True:
                print(f"No improvement in validation loss. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
            return False


def train(df, num_epochs, save_pth, save_on_evaluation=False, early_stop=None, loss_fn='proxy_auc', sampler=True):

    print("Starting training script...")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42) 

    X, y = preprocess_data(df)

    # Stratified 80/20 split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

    for train_idx, test_idx in sss.split(X, y):
        X_train_val, X_test = X[train_idx], X[test_idx]
        y_train_val, y_test = y[train_idx], y[test_idx]

    print(f"Training + validation shape: {X_train_val.shape}, Labels shape: {y_train_val.shape}")
    print(f"Test shape: {X_test.shape}, Labels shape: {y_test.shape}")

    # test_df = pd.DataFrame(X_test, columns=df.drop(columns=['Eskd', 'Code']).columns)
    # test_df['Eskd'] = y_test
    # test_df.to_csv(os.path.join(save_pth, 'test_train_data_20.csv'), index=False)

    # Controllo distribuzione classi
    #        Train
    # Classe 0 → 586 (~77.5%)
    # Classe 1 → 170 (~22.5%)
    #        Test
    # Classe 0 → 146 (~77.2%)
    # Classe 1 → 43  (~22.8%)
    unique_train, counts_train = np.unique(y_train_val, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print(f"Training class distribution: {dict(zip(unique_train, counts_train))}")
    print(f"Test class distribution: {dict(zip(unique_test, counts_test))}")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_train_val, y_train_val))
    print(f"Numero di split: {len(splits)}")
    #  Dal file sweep.py prendo il valore del fold che eseguo
    wandb.init()

    fold_to_run = getattr(wandb.config, 'fold', 1)  # default fold = 1

    train_idx, val_idx = splits[fold_to_run - 1]
    print(f"Running fold {fold_to_run} with {len(train_idx)} training samples and {len(val_idx)} validation samples.")
    
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val[train_idx], y_train_val[val_idx]

    model, optimizer, scheduler, criterion, step_start, step_end = model_initialization(X_train, y_train, loss_fn=loss_fn)

    # Z-score normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    # Non uso fit_transform ma solo transform per evitare data leakage, non ricalcolo la deviazione standard sui dati di validazione
    X_val = scaler.transform(X_val)
    # In un file a parte salvo anche la configurazione di parametri usata e le metriche raggiunte con questa configurazione, voglio sovrascrivere il file ogni volta che trovo un modello migliore
    scaler_file = os.path.join(save_pth, f"scaler_fold_{fold_to_run}.pkl")
    joblib.dump(scaler, scaler_file)
    # Tensori PyTorch
    batch_size = wandb.config.batch_size if 'batch_size' in wandb.config else 32

    # Qua trasformo il mio amico in tensore per la mia rete
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    # Qua come sempre devo cambiare shape perché BCEWithLogitsLoss vuole i target con shape (N, 1) e non (N,)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

    if sampler == True:
        # Implemeto il weighted random sampler per bilanciare le classi nel training
        class_sample_count = np.array([len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[int(t)] for t in y_train])
        samples_weight = torch.from_numpy(samples_weight).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=2)
        # --- Debug: controlla la proporzione per i primi 5 batch ---
        for i, (inputs, labels) in enumerate(itertools.islice(train_loader, 5)):
            num_pos = (labels == 1).sum().item()
            num_neg = (labels == 0).sum().item()
            print(f"Batch {i}: Positivi={num_pos}, Negativi={num_neg}")

    else:
    
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    # Anche qua cambio shape perchè il modello mi sputa (N, 1), un output per ogni campione, e questo Nik devo saperlo, grande Nik
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1,1)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Early stopping 
    early_stopping = EarlyStopping(patience=2, verbose=True)    
    # Training loop
    global_step = 0
    
    save_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}_config.json")
    model_file = os.path.join(save_pth, f"best_model_fold_{fold_to_run}.pth")

    wandb.define_metric("epoch")
    wandb.define_metric("train/loss", step_metric="epoch")
    wandb.define_metric("train/accuracy", step_metric="epoch")
    wandb.define_metric("train/precision", step_metric="epoch")
    wandb.define_metric("train/recall", step_metric="epoch")
    wandb.define_metric("train/f1-score", step_metric="epoch")
    wandb.define_metric("learning_rate", step_metric="epoch")
    wandb.define_metric("val/loss", step_metric="epoch")
    wandb.define_metric("val/accuracy", step_metric="epoch")
    wandb.define_metric("val/precision", step_metric="epoch")
    wandb.define_metric("val/recall-score", step_metric="epoch")
    wandb.define_metric("val/f1-score", step_metric="epoch")
    wandb.define_metric("val/auc_roc", step_metric="epoch")

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
        train_precision = precision_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0)
        train_recall = recall_score(all_labels.cpu(), all_preds_bin.cpu(), zero_division=0)
        epoch_loss = running_loss / len(train_loader.dataset)
        #print(f"Training Loss: {epoch_loss:.4f}")

        try :
            wandb.log({"train/loss" : epoch_loss, 'epoch': epoch})
            wandb.log({"train/accuracy" : train_acc, 'epoch': epoch})
            wandb.log({"train/precision": train_precision, 'epoch' : epoch})
            wandb.log({"train/recall" : train_recall, 'epoch' : epoch})
            wandb.log({"train/f1-score" : train_f1_score, 'epoch' : epoch})
            wandb.log({"learning_rate" : optimizer.param_groups[0]['lr'], 'epoch' : epoch})  
        except Exception as e:
            print(f'Error in wandb: {e}')

        
        # Qui faccio la validazione ad ogni epoca se richiesto e salvo il modello se migliora la F1 di validazione 
        # come suggerito dal grande e unico Nik
        if save_on_evaluation == True:
            print("Starting validation...")
            model.eval()
            val_labels = []
            val_preds = []
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    outputs = model(inputs)
                    val_labels.append(labels)
                    val_preds.append(torch.sigmoid(outputs).detach())
                    val_loss += criterion(outputs, labels).item() * inputs.size(0)

            val_labels = torch.cat(val_labels)
            val_preds = torch.cat(val_preds)
            val_preds_bin = (val_preds > 0.5).float()
            val_acc = (val_preds_bin == val_labels).float().mean()
            val_f1_score = f1_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0)

            # Calcolo anche l'AUC-ROC
            if len(torch.unique(val_labels)) > 1:  # Controllo che ci siano entrambe le classi
                val_auc_roc = roc_auc_score(val_labels.cpu(), val_preds.cpu())
                print(f"Validation AUC-ROC: {val_auc_roc:.4f}")
                try:
                    wandb.log({"val/auc_roc": val_auc_roc, 'epoch': epoch})
                except Exception as e:
                    print(f'Error in wandb while logging AUC-ROC: {e}')

            # Validation loss
            # Usiamo direttamente i logit del modello, senza sigmoid
            val_loss = val_loss / len(val_loader.dataset)
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation F1 Score: {val_f1_score:.4f}")

            # Log su wandb
            try:
                wandb.log({"val/loss": val_loss, 'epoch': epoch})
                wandb.log({"val/accuracy": val_acc, 'epoch': epoch})
                wandb.log({"val/precision": precision_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0), 'epoch': epoch})
                wandb.log({"val/recall-score": recall_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0), 'epoch': epoch})
                wandb.log({"val/f1-score": val_f1_score, 'epoch': epoch})
            except Exception as e:
                print(f'Error in wandb while logging validation metrics: {e}')

            # Early stopping check
            if early_stop is not None and (epoch + 1) % 5 == 0:
                if early_stopping.early_stop(val_loss):
                    print("Early stopping triggered. Stopping training.")
                    break
            else:
                print("Early stopping not enabled or not the right epoch for checking.  Continuing training.")

            if val_f1_score > read_json(save_file):
                # Salvo il modello
                print(f"New best model with Validation F1: {val_f1_score}, saving model...")
                torch.save(model.state_dict(), model_file)
                wandb.log({"best_val_f1": val_f1_score})
        
                config_and_metrics = {
                    "Run ID": wandb.run.id,
                    "Config Parameters": dict(wandb.config),
                    "Best Model Metrics": {
                        "Epoch": epoch + 1,
                        "Validation Loss": float(val_loss),
                        "Best F1": float(val_f1_score),
                        "Validation Accuracy": float(val_acc),
                        "Validation Precision": float(precision_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0)),
                        "Validation Recall": float(recall_score(val_labels.cpu(), val_preds_bin.cpu(), zero_division=0))
                    }
                }
                with open(save_file, 'w') as f:
                    json.dump(config_and_metrics, f, indent=4)

    wandb.finish()


if __name__ == "__main__":
  
    print("Starting training script...")
    # STEPS:
    """ 1. Split train/val
        2. Fit scaler on train
        3. Save scaler ← FATTO UNA VOLTA PERCHE FACCIO LA EVAL A PARTE COLPA MIA LO SO
        4. Training loop:
            - Each epochs: compute metrics train + val
            - If F1_val improves: save model + config
            - Each 5 epochs: check early stopping
    """
    early_stop = None
    data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
    save_pth = '/work/grana_far2023_fomo/ESKD/Models_SWEEP_PARAM_ADAM_PROXYLOSS_SAMPLER/'
    df = pd.read_excel(data_path)
    num_epochs = 120
    save_on_evaluation = True
    train(df, num_epochs, save_pth, save_on_evaluation, early_stop, loss_fn='proxy_auc', sampler=True)