import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os

class FallTransformer(nn.Module):
    def __init__(self, feature_dim=12, seq_len=15, d_model=64, nhead=4, num_layers=2):
        super(FallTransformer, self).__init__()
        self.embedding = nn.Linear(feature_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x):  # x: [B, 15, 7]
        x = self.embedding(x) + self.pos_embed[:, :x.size(1), :]
        x = self.encoder(x)  # [B, 15, d_model]
        x = x[:, x.size(1) // 2, :]  # 중심 프레임 추출
        return torch.sigmoid(self.classifier(x)).squeeze()

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

def load_sequences_from_json(data_json_path, label_root_path, max_frames=600):
    with open(data_json_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    X_seq = []
    y_seq = []

    for i,video in enumerate(data1):
        label_name = video["vid_name"].replace(".mp4", ".json")
        label_path = None
        for root, dirs, files in os.walk(label_root_path):
            if label_name in files:
                label_path = os.path.join(root, label_name)
                break
        if label_path is None:
            continue
        
        with open(label_path, "r", encoding="utf-8") as f:
            print(i+1,os.path.join(root_path,label_path))
            data2 = json.load(f)

        fall_start = data2["sensordata"]["fall_start_frame"]
        fall_end = data2["sensordata"]["fall_end_frame"]

        for j, frame in enumerate(video["frames"]):
            if j >= max_frames:
                break
            feature_dict = frame["features"]
            vec = [
                feature_dict["vec_num"],
                feature_dict["down_ratio"],
                feature_dict["speed_mean"],
                feature_dict["speed_std"],
                feature_dict["angle_std"],
                feature_dict["fastdown_num"],
                feature_dict["delta_vec_num"],
                feature_dict["delta_down_ratio"],
                feature_dict["delta_fastdown_num"],
                feature_dict["hip_accel"],
                feature_dict["shoulder_accel"],
                feature_dict["head_accel"]
            ]
            if all(v == 0.0 for v in vec):
                continue
            X_seq.append(vec)
            index = frame["frame_index"]
            y_seq.append(1 if fall_start <= index <= fall_end else 0)

    return np.array(X_seq), np.array(y_seq)

root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터\VL\영상"
X_seq, y_seq = load_sequences_from_json("data.json", root_path)
window_size = 15
X_win, y_win = create_sliding_windows(X_seq, y_seq, window_size)

X_train, X_test, y_train, y_test = train_test_split(X_win, y_win, stratify=y_win, test_size=0.2)

batch_size = 128
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

model = FallTransformer(feature_dim=12, seq_len=window_size)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()
for epoch in range(10):
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

model.eval()
y_pred_all, y_true_all = [], []

with torch.no_grad():
    for xb, yb in test_loader:
        outputs = model(xb)
        preds = (outputs > 0.5).int().numpy()
        y_pred_all.extend(preds)
        y_true_all.extend(yb.int().numpy())

f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
print(f"\n F2 Score: {f2:.4f}")

