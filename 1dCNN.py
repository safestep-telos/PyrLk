import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os

# 1. CNN 모델 정의
class FallDetection1DCNN(nn.Module):
    def __init__(self, in_channels=12, window_size=15):
        super(FallDetection1DCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, 7, window_size]
        return self.net(x)

# 2. 슬라이딩 윈도우 함수 (X_seq: [N, 7], y_seq: [N])
def create_sliding_windows(X_seq, y_seq, window_size=15):
    X_win = []
    y_win = []
    half = window_size // 2
    for i in range(len(X_seq) - window_size + 1):
        window = X_seq[i:i+window_size]
        label = y_seq[i + half]
        X_win.append(window)
        y_win.append(label)
    return torch.tensor(X_win, dtype=torch.float32), torch.tensor(y_win, dtype=torch.float32)


X_seq = np.load("X_seq.npy")
y_seq = np.load("y_seq.npy")
# 4. 윈도우 생성 + train/test 분할
window_size = 15
X_win, y_win = create_sliding_windows(X_seq, y_seq, window_size)
X_win = X_win.permute(0, 2, 1)  # [B, 7, 15]

X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, stratify=y_win, test_size=0.2)

# 5. 모델 학습
model = FallDetection1DCNN(in_channels=7, window_size=window_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 학습 루프
model.train()
for epoch in range(10):
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 6. 평가 (F2-score)
model.eval()
with torch.no_grad():
    y_pred = model(X_test).squeeze()
    y_pred_bin = (y_pred > 0.5).int().numpy()
    y_true = y_test.int().numpy()
    f2 = fbeta_score(y_true, y_pred_bin, beta=2)
    print(f"\n F2 Score: {f2:.4f}")