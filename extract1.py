import numpy as np
import cv2 as cv
import threading
from queue import Queue
import time
import os
import glob
import json
from ultralytics import YOLO  # YOLOv8n 포즈

class OpticalFlowExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv.VideoCapture(video_path)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.frame_cnt = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.runtime = self.frame_cnt / self.fps

        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=3,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        self.pose_model = YOLO("yolov8n-pose.pt")  # 경량화된 YOLOv8n-pose 사용
        self.pose_next_try = 0
        self.frame_queue = Queue()
        self.data = []
        self.frame_idx = 0
        self.pose_landmarks = None
        self.prev_landmarks = []

    def get_pose_landmarks(self, frame):
        results = self.pose_model.predict(source=frame, conf=0.4, verbose=False)[0]
        for keypoints in results.keypoints.xy:
            if len(keypoints) >= 33:
                return [(int(x), int(y)) for x, y in keypoints.cpu().numpy()]
        return None

    def calc_optical_flow(self):
        prev = None
        step = 16
        roi_radius = 32
        start_time = time.time()

        while True:
            frame_info = self.frame_queue.get()
            if frame_info is None:
                break

            frame, frame_idx = frame_info
            frame = cv.resize(frame, (640, 360))
            height, width = frame.shape[:2]

            grid_points = [[x, y] for y in range(step // 2, height, step) for x in range(step // 2, width, step)]

            if prev is None:
                prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.data.append(self.empty_feature(frame_idx))
                continue

            if frame_idx >= self.pose_next_try:
                landmarks = self.get_pose_landmarks(frame)
                if landmarks:
                    self.pose_landmarks = landmarks
                    self.prev_landmarks.append(landmarks)
                    if len(self.prev_landmarks) > 3:
                        self.prev_landmarks.pop(0)
                    self.pose_next_try = frame_idx + 2
                else:
                    self.pose_next_try = frame_idx + 1

            roi_points = []
            if self.pose_landmarks:
                lm_indices = [0, 11, 12, 23, 24, 25, 26, 27, 28]
                for i in lm_indices:
                    cx, cy = self.pose_landmarks[i]
                    roi_points.extend([
                        [x, y] for x, y in grid_points
                        if abs(cx - x) <= roi_radius and abs(cy - y) <= roi_radius
                    ])

            if not roi_points:
                prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.data.append(self.empty_feature(frame_idx))
                continue

            p0 = np.array(roi_points, dtype=np.float32).reshape(-1, 1, 2)
            curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(prev, curr, p0, None, **self.lk_params)

            vectors = []
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

            self.data.append(self.extract_features(frame_idx, vectors))
            prev = curr


    def empty_feature(self, frame_idx):
        keys = ["vec_num", "down_ratio", "speed_mean", "speed_std", "angle_std", "fastdown_num",
                "delta_vec_num", "delta_down_ratio", "delta_fastdown_num",
                "hip_accel", "shoulder_accel", "head_accel"]
        return {
            "time": time.time(),
            "frame_index": frame_idx,
            "fall": 0,
            "features": {k: 0.0 for k in keys}
        }

    def extract_features(self, frame_idx, vectors):
        if not vectors:
            return self.empty_feature(frame_idx)
        speeds = [v["t"] for v in vectors]
        angles = [v["a"] for v in vectors]
        isdowns = [v["isdown"] for v in vectors]
        fastdowns = [1 for v in vectors if v["isdown"] and v["t"] > 6]

        feature = {
            "vec_num": float(len(vectors)),
            "down_ratio": float(sum(isdowns) / len(vectors)),
            "speed_mean": float(np.mean(speeds)),
            "speed_std": float(np.std(speeds)),
            "angle_std": float(np.std(angles)),
            "fastdown_num": float(len(fastdowns)),
            "delta_vec_num": 0.0,
            "delta_down_ratio": 0.0,
            "delta_fastdown_num": 0.0,
            "hip_accel": 0.0,
            "shoulder_accel": 0.0,
            "head_accel": 0.0
        }

        if len(self.data) >= 1:
            prev_f = self.data[-1]["features"]
            feature["delta_vec_num"] = feature["vec_num"] - prev_f["vec_num"]
            feature["delta_down_ratio"] = feature["down_ratio"] - prev_f["down_ratio"]
            feature["delta_fastdown_num"] = feature["fastdown_num"] - prev_f["fastdown_num"]

        if len(self.prev_landmarks) >= 3:
            p0, p1, p2 = [np.array(lms) for lms in self.prev_landmarks[-3:]]
            for name, idx in [("hip_accel", 23), ("shoulder_accel", 11), ("head_accel", 0)]:
                if np.all(p0[idx]) and np.all(p1[idx]) and np.all(p2[idx]):
                    accel = np.linalg.norm(p0[idx] - 2 * p1[idx] + p2[idx])
                    feature[name] = float(accel)

        return {
            "time": time.time(),
            "frame_index": frame_idx,
            "fall": 0,
            "features": feature
        }

    def run(self):
        thread = threading.Thread(target=self.calc_optical_flow)
        thread.start()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_idx += 1
            self.frame_queue.put((frame, self.frame_idx))

        thread.join()
        self.cap.release()

# 실행 부분
if __name__ == "__main__":
    root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터\VS\영상"
    all_videos = glob.glob(os.path.join(root_path, "**", "*.mp4"), recursive=True)

    part_index = 0
    merged_data = []

    for i, video in enumerate(all_videos):
        print(f"[{i+1}/{len(all_videos)}]")
        vid_id = i + 1
        start_time = time.time()
        extractor = OpticalFlowExtractor(video)
        extractor.run()
        elapsed = time.time() - start_time

        merged_data.append({
            "vid_id": vid_id,
            "vid_name": os.path.basename(video),
            "frames": extractor.data
        })

        if (i + 1) % 1000 == 0 or (i + 1) == len(all_videos):
            part_index += 1
            filename = f"data_part{part_index}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
            merged_data = []