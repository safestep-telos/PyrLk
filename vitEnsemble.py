import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision import models, transforms
from PIL import Image
import cv2 as cv
import glob
import os
import pickle

# FallTransformer (벡터 기반)
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

# ViT 기반 이미지 모델 (pretrained)
class ImageTransformer(nn.Module):
    def __init__(self):
        super(ImageTransformer, self).__init__()
        self.backbone = models.vit_b_16(pretrained=True)
        self.backbone.heads = nn.Identity()  # Feature extractor
        self.fc = nn.Linear(768, 1)

    def forward(self, x):  # x: [B, 3, 224, 224]
        x = self.backbone(x)
        return self.fc(x).squeeze()

# 데이터 불러오기 (벡터)
X_seq = np.load("/content/drive/MyDrive/X_seq.npy")
y_seq = np.load("/content/drive/MyDrive/y_seq.npy")
scaler = StandardScaler()
X_seq = scaler.fit_transform(X_seq)

# 시퀀스 슬라이딩 윈도우
window_size = 15
X_win, y_win, image_indices = [], [], []
for i in range(len(X_seq) - window_size + 1):
    X_win.append(X_seq[i:i+window_size])
    y_win.append(y_seq[i + window_size // 2])
    center = i + window_size // 2
    image_indices.append(center)
X_win = torch.tensor(np.array(X_win)).float()
y_win = torch.tensor(np.array(y_win)).float()

# 영상에서 프레임 추출
"""root_path = r"/content/drive/MyDrive/영상"
all_videos = glob.glob(os.path.join(root_path, "**", "*.mp4"), recursive=True)"""
all_frames = []

# 중앙 프레임만 추출 (X_win과 길이 일치)
frames_dir = "/content/frames_parts"
all_frame_tensors = []
pt_files = sorted(glob.glob(os.path.join(frames_dir, "frames_part_*.pt")))
for pt in pt_files:
    all_frame_tensors.append(torch.load(pt))
all_frames = torch.cat(all_frame_tensors, dim=0)
image_tensor = torch.stack([all_frames[i] for i in image_indices])
assert len(image_tensor) == len(X_win), "벡터와 이미지 개수가 다릅니다"

# Train/Test split
X_train_vec, X_test_vec, y_train, y_test, X_train_img, X_test_img = train_test_split(
    X_win, y_win, image_tensor, stratify=y_win, test_size=0.2, random_state=42)

train_loader = DataLoader(TensorDataset(X_train_vec, X_train_img, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test_vec, X_test_img, y_test), batch_size=64)

# 모델 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vector_model = FallTransformer().to(device)
image_model = ImageTransformer().to(device)

# 손실 함수와 옵티마이저
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]).to(device))
optimizer = torch.optim.Adam(
    list(vector_model.parameters()) + list(image_model.parameters()),
    lr=5e-4
)

# 학습 루프
for epoch in range(20):
    vector_model.train()
    image_model.train()
    for xb_vec, xb_img, yb in train_loader:
        xb_vec, xb_img, yb = xb_vec.to(device), xb_img.to(device), yb.to(device)
        optimizer.zero_grad()
        out_vec = vector_model(xb_vec)
        out_img = image_model(xb_img)
        out = (out_vec + out_img) / 2
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

    # 검증
    vector_model.eval()
    image_model.eval()
    y_pred_all, y_true_all = [], []
    with torch.no_grad():
        for xb_vec, xb_img, yb in test_loader:
            xb_vec, xb_img = xb_vec.to(device), xb_img.to(device)
            out_vec = vector_model(xb_vec)
            out_img = image_model(xb_img)
            out = (out_vec + out_img) / 2
            preds = (out > 0.5).int().cpu().numpy()
            y_pred_all.extend(preds)
            y_true_all.extend(yb.int().cpu().numpy())

    f2 = fbeta_score(y_true_all, y_pred_all, beta=2)
    print(f"Epoch {epoch+1}, F2 Score: {f2:.4f}")