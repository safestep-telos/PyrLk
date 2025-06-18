import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.metrics import roc_auc_score

class FallTransformer(nn.Module):
    def __init__(self, feature_dim=12, seq_len=15, d_model=64, nhead=4, num_layers=2):
        super(FallTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)
        x = x[:, x.size(1) // 2, :]
        return self.classifier(x).squeeze()

def create_sliding_windows(X_seq, y_seq, window_size=15):
    X_win = []
    y_win = []
    half = window_size // 2
    for i in range(len(X_seq) - window_size + 1):
        window = X_seq[i:i+window_size]
        label = y_seq[i + half]
        X_win.append(window)
        y_win.append(label)
    X_np = np.array(X_win)
    y_np = np.array(y_win)
    return torch.from_numpy(X_np).float(), torch.from_numpy(y_np).float()

# 데이터 불러오기
X_seq = np.load("/content/drive/MyDrive/X_seq1.npy")
y_seq = np.load("/content/drive/MyDrive/y_seq1.npy")
scaler = StandardScaler()
X_seq = scaler.fit_transform(X_seq)
window_size = 15
X_win, y_win = create_sliding_windows(X_seq, y_seq, window_size)

X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, stratify=y_win, test_size=0.2)

batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallTransformer(feature_dim=12, seq_len=window_size).to(device)
#criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

best_f2 = 0.0
patience = 10
wait = 0

train_losses, val_losses, f2_scores = [], [], []

print("Using device:", device)
model.train()
for epoch in range(100):
    model.train()
    total_loss = 0
    num_batches = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    train_losses.append(avg_loss)

    # Validation loss + F2
    model.eval()
    val_loss = 0
    val_batches = 0
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            val_loss += criterion(outputs, yb).item()
            val_batches += 1
            preds = (outputs > 0.5).int().cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(yb.int().cpu().numpy())

    avg_val_loss = val_loss / val_batches
    f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
    val_losses.append(avg_val_loss)
    f2_scores.append(f2)

    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, F2: {f2:.4f}")

    if f2 > best_f2:
        best_f2 = f2
        wait = 0
        dir = f"/content/drive/MyDrive/vec_transformer/best_model2.pt"
        #dir = f"/content/drive/MyDrive/vec_transformer/best_model.pt"
        torch.save(model.state_dict(), dir)
        with open("best_scaler.pkl", "wb") as f:
          pickle.dump(scaler, f)
        print(" New best model saved.")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# 그래프 출력
plt.figure(figsize=(10,5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.plot(f2_scores, label='F2 Score')
plt.xlabel("Epoch")
plt.ylabel("Metric")
plt.legend()
plt.grid(True)
plt.title("Training vs Validation Loss and F2 Score")
plt.show()

from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score

# 모델 로드
model.load_state_dict(torch.load("/content/drive/MyDrive/vec_transformer/best_model2.pt"))
model.eval()

y_pred_all, y_true_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).int().cpu().numpy()
        y_pred_all.extend(preds)
        y_true_all.extend(yb.int().cpu().numpy())

# 평가
f1 = f1_score(y_true_all, y_pred_all)
f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
precision = precision_score(y_true_all, y_pred_all)
recall = recall_score(y_true_all, y_pred_all)
auc = roc_auc_score(y_true_all, y_pred_all)
print(len(y_pred_all))
print("\n Final Evaluation on Best Model:")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"F2 Score  : {f2:.4f}")
print("auc",auc)