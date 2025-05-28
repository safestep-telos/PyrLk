import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
from sklearn.preprocessing import StandardScaler

class FallTransformer(nn.Module):
    def __init__(self, feature_dim=12, seq_len=15, d_model=64, nhead=4, num_layers=2):
        super(FallTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, 15, 7]
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)  # [B, 15, d_model]
        x = x[:, x.size(1) // 2, :]  # 중심 프레임 추출
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

root_path = r"/content/drive/MyDrive/영상"
X_seq = np.load(r"/content/drive/MyDrive/X_seq.npy")
y_seq = np.load(r"/content/drive/MyDrive/y_seq.npy")
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
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

best_f2 = 0.0
patience = 5
wait = 0

print("Using device:", device)
model.train()
for epoch in range(50):  # 늘어난 세대 수
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

    # Validation
    model.eval()
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = (outputs > 0.5).int().cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(yb.int().cpu().numpy())
    avg_loss = total_loss/num_batches
    f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
    print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, F2: {f2:.4f}")

    # Early stopping
    if f2 > best_f2:
        best_f2 = f2
        wait = 0
        torch.save(model.state_dict(), "best_model.pt")
        print(" New best model saved.")
    else:
        wait += 1
        if wait >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

model.load_state_dict(torch.load("best_model.pt"))
model.eval()
y_pred_all, y_true_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(device), yb.to(device)
        outputs = model(xb)
        preds = (outputs > 0.5).int().cpu().numpy()
        y_pred_all.extend(preds)
        y_true_all.extend(yb.int().cpu().numpy())

f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
print(f"\n F2 Score: {f2:.4f}")