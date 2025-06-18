import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import StandardScaler

import numpy as np
import cv2 as cv
import threading
from queue import Queue
import time
import collections
import pickle
import mediapipe as mp

mp_pose = mp.solutions.pose
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#out = cv.VideoWriter('output1.mp4', cv.VideoWriter_fourcc(*'XVID'), 30, (1280, 720))

class RealTimeFallDetector:
    def __init__(self, video_path, model_path, scaler_path):
        self.cap = cv.VideoCapture(video_path)
        #self.cap = cv.VideoCapture(0) #실시간 영상 적용시 사용
        self.pose = mp_pose.Pose(static_image_mode=False, model_complexity=0)
        self.prev_gray = None
        self.prev_landmarks = collections.deque(maxlen=3)
        self.pose_landmarks = None
        self.pose_next_try = 0
        self.frame_idx = 0

        self.model = FallTransformer().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.feature_queue = collections.deque(maxlen=15)

    def get_pose_landmarks(self, frame):
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            return [(lm.x, lm.y) for lm in results.pose_landmarks.landmark]
        return None

    def extract_features(self, vectors):
        if not vectors:
            return [0.0] * 12
        speeds = [v["t"] for v in vectors]
        angles = [v["a"] for v in vectors]
        isdowns = [v["isdown"] for v in vectors]
        fastdowns = [1 for v in vectors if v["isdown"] and v["t"] > 6]
        return [
            len(vectors),
            sum(isdowns) / len(vectors),
            np.mean(speeds),
            np.std(speeds),
            np.std(angles),
            len(fastdowns),
            0.0, 0.0, 0.0,  # delta features
            0.0, 0.0, 0.0   # accel
        ]

    def run(self):
        start_time = time.time()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frame_idx += 1
            #frame = frame[:, frame.shape[1]*4//7:, :]
            frame = cv.resize(frame, (640, 360))
            height, width = frame.shape[:2]
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # 포즈 업데이트
            if self.frame_idx >= self.pose_next_try:
                landmarks = self.get_pose_landmarks(frame)
                if landmarks and len(landmarks) >= 33:
                    self.pose_landmarks = [(int(x * width), int(y * height)) for x, y in landmarks]
                    self.prev_landmarks.append(self.pose_landmarks)
                    self.pose_next_try = self.frame_idx + 2
                else:
                    self.pose_next_try = self.frame_idx + 1

            # ROI 설정
            roi_points = []
            if self.pose_landmarks:
                lm_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]
                for i in lm_indices:
                    cx, cy = self.pose_landmarks[i]
                    for x in range(cx - 32, cx + 32, 16):
                        for y in range(cy - 32, cy + 32, 16):
                            if 0 <= x < width and 0 <= y < height:
                                roi_points.append([x, y])

            vectors = []
            if self.prev_gray is not None and roi_points:
                p0 = np.array(roi_points, dtype=np.float32).reshape(-1, 1, 2)
                p1, st, err = cv.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                      winSize=(15, 15), maxLevel=3,
                                                      criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    for new, old in zip(good_new, good_old):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        dx, dy = a - c, b - d
                        speed = np.sqrt(dx ** 2 + dy ** 2)
                        degree = -np.degrees(np.arctan2(dy, dx))
                        isDownwards = int(-135 < degree < -45)
                        if 1 <= speed <= 15:
                            vectors.append({"x": c, "y": d, "dx": dx, "dy": dy, "t": speed, "a": degree, "isdown": isDownwards})

            vec = self.extract_features(vectors)

            vec_scaled = self.scaler.transform(np.array(vec).reshape(1, -1))
            self.feature_queue.append(vec_scaled.flatten())
            self.prev_gray = gray
            text = ""
            if len(self.feature_queue) == 15:
                X_input = torch.tensor([list(self.feature_queue)], dtype=torch.float32).to(device)
                with torch.no_grad():
                    output = self.model(X_input)
                    pred = (output > 0.5).int().item()
                    text = f"Frame: {self.frame_idx} | isFall : {'Yes' if pred else 'No'}"
                    print(text)
            """frame = cv.resize(frame, (1280, 720))
            cv.putText(frame, text, (150, 70), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            cv.imshow("Real-Time Fall Detection", frame)
            #out.write(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break"""
            #위 주석은 시연용 영상 저장하는 코드
        end_time = time.time()  # 처리 종료 시간
        elapsed_time = end_time - start_time
        print(f"\n[INFO] 영상 처리에 걸린 시간 (초): {elapsed_time:.2f}")
        print(f"[INFO] 프레임당 평균 처리 시간: {elapsed_time / self.frame_idx:.4f} 초")
        
        self.cap.release()
        cv.destroyAllWindows()

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


# 1. 영상에서 특징 추출 (1개 영상만 가정)
video_path = r"" #falltrue.mp4  fallfalse4.mp4 slide.mp4  # 실시간 영상을 대체할 샘플 영상
realtime_detector = RealTimeFallDetector(video_path,"best_model2.pt", "best_scaler.pkl")
realtime_detector.run()
