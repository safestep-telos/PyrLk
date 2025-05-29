import torch
from transformer import FallTransformer
from extract import OpticalFlowExtractor
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. 영상에서 특징 추출 (1개 영상만 가정)
video_path = "your_video.mp4"  # 실시간 영상을 대체할 샘플 영상
extractor = OpticalFlowExtractor(video_path)
extractor.run()
frames = extractor.data

# 2. 특징 벡터만 추출하여 시퀀스 구성
X_seq = [list(frame["features"].values()) for frame in frames]
X_seq = np.array(X_seq)

# 3. 전처리 (학습과 동일한 스케일링)
scaler = StandardScaler()
X_seq = scaler.fit_transform(X_seq)

# 4. 슬라이딩 윈도우 시퀀스 구성
def create_sliding_windows(X_seq, window_size=15):
    X_win = []
    half = window_size // 2
    for i in range(len(X_seq) - window_size + 1):
        window = X_seq[i:i + window_size]
        X_win.append(window)
    return torch.from_numpy(np.array(X_win)).float()

X_input = create_sliding_windows(X_seq)

# 5. 모델 불러오기
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FallTransformer(feature_dim=12, seq_len=15).to(device)
model.load_state_dict(torch.load("best_model.pt", map_location=device))
model.eval()

# 6. 예측 수행
with torch.no_grad():
    outputs = model(X_input.to(device))
    preds = (outputs > 0.5).int().cpu().numpy()

# 7. 결과 출력
for i, pred in enumerate(preds):
    print(f"Frame {i + 7} → 낙상 감지: {'Yes' if pred else 'No'}")
