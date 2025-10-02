import os
import pandas as pd
import torch.nn as nn
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
# Valutazione del modello
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Caricamento dati
data_path = '/work/grana_far2023_fomo/ESKD/Data/final_cleaned_maxDateAccess.xlsx'
save_pth = '/work/grana_far2023_fomo/ESKD/Models/best_model.pth'

df = pd.read_excel(data_path)

# Preprocessing
df['Gender'] = df['Gender'].replace({'M': 1, 'F': 2})
print(df['Gender'].unique())
X = df.drop(columns=['Eskd', 'Code']).values
y = df['Eskd'].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
y = np.nan_to_num(y, nan=0.0)

# Suddivisione in training e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizzazione
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)       

# Conversione in tensori PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)   
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Voglio un modello semplice con 4 layer densi a 100 neurono ciascuno, con funzione di attivazione ELU
class SimpleBinaryNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleBinaryNN, self).__init__()
        self.layers = nn.Sequential(
             
            nn.Linear(input_size, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(100, 1),
            
        )
        
    def forward(self, x):
            return self.layers(x)
            
model = SimpleBinaryNN(X_train.shape[1])
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # Decay del learning rate

# Training del modello
num_epochs = 150
best_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)

        preds = (torch.sigmoid(outputs) > 0.5).float()
        all_preds.append(preds)
        all_labels.append(labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()

    # Salvataggio best model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), save_pth)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(f'Learning Rate: {current_lr:.6f}')


# Valutazione del modello
model.eval()
y_pred = []
with torch.no_grad():
    for inputs, _ in test_loader:
        outputs = model(inputs)
        y_pred.extend(outputs.cpu().numpy())  # conversione sicura

# Converto a tensore e applico sigmoid
y_pred = torch.tensor(y_pred)
y_pred = torch.sigmoid(y_pred).numpy()
y_pred = (y_pred > 0.5).astype(int)

# Calcolo metriche
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")



