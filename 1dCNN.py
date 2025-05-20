import torch
import torch.nn as nn
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os

# 1. CNN 모델 정의
class FallDetection1DCNN(nn.Module):
    def __init__(self, in_channels=7, window_size=15):
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

# 3. (예시) 데이터를 불러왔다고 가정하고 numpy 배열로 준비
def load_sequences_from_json(data_json_path, label_root_path, max_frames=600):
    with open(data_json_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    X_seq = []
    y_seq = []

    for i, video in enumerate(data1):
        # 라벨 경로 찾기
        label_name = video["vid_name"].replace(".mp4", ".json")
        label_path = None
        for root, dirs, files in os.walk(label_root_path):
            if label_name in files:
                label_path = os.path.join(root, label_name)
                break
        if label_path is None:
            continue  # 라벨 없으면 skip

        with open(label_path, "r", encoding="utf-8") as f:
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
                feature_dict["angle_mean"],
                feature_dict["angle_std"],
                feature_dict["fastdown_num"]
            ]

            # 모든 값이 0이면 의미 없는 프레임 → 제외
            if all(v == 0.0 for v in vec):
                continue

            X_seq.append(vec)

            index = frame["frame_index"]
            y_seq.append(1 if fall_start <= index <= fall_end else 0)

    return np.array(X_seq), np.array(y_seq)
root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터\VL\영상"
X_seq,y_seq = load_sequences_from_json("data.json",root_path)
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
    print(f"\n✅ F2 Score: {f2:.4f}")
