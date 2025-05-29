import numpy as np
import cv2 as cv
import threading
import time
import os
import glob
import json
from queue import Queue
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
from mediapipe.framework.formats import landmark_pb2

# MediaPipe Landmarker 설정
model_path = "pose_landmarker_heavy.task"
options = PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=model_path),
    running_mode=mp_python.vision.RunningMode.VIDEO,
    output_segmentation_masks=False,
    num_poses=5  # 다중 포즈 검출
)
pose_model = PoseLandmarker.create_from_options(options)

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

        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.data = []
        self.frame_idx = 0
        self.prev = None

    def calc_optical_flow(self):
        step = 32
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

            if self.prev is None:
                self.prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.data.append(self._empty_feature(frame_idx))
                continue

            roi_points = []
            if frame_idx % 3 == 1:
                mp_image = mp_python.vision.Image(image_format=mp_python.vision.ImageFormat.SRGB, data=frame)
                detection_result = pose_model.detect_for_video(mp_image, int(frame_idx * (1000 / self.fps)))

                for person in detection_result.pose_landmarks:
                    for idx in [0, 23, 24, 15, 16, 27, 28]:
                        lm = person[idx]
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        for x, y in grid_points:
                            if abs(cx - x) <= roi_radius and abs(cy - y) <= roi_radius:
                                roi_points.append([x, y])

            if not roi_points:
                self.prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                self.data.append(self._empty_feature(frame_idx))
                continue

            p0 = np.array(roi_points, dtype=np.float32).reshape(-1, 1, 2)
            curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            p1, st, err = cv.calcOpticalFlowPyrLK(self.prev, curr, p0, None, **self.lk_params)

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

            self.data.append(self._compute_feature(frame_idx, vectors))
            self.prev = curr

        print(f"[INFO] Runtime {self.runtime:.2f} 처리 완료. 실행시간: {time.time() - start_time:.2f}초 프레임수 : {self.frame_cnt}")

    def _empty_feature(self, frame_idx):
        return {
            "time": time.time(),
            "frame_index": frame_idx,
            "fall": 0,
            "features": {k: 0.0 for k in ["vec_num", "down_ratio", "speed_mean", "speed_std", "angle_mean", "angle_std", "fastdown_num"]}
        }

    def _compute_feature(self, frame_idx, vectors):
        if not vectors:
            return self._empty_feature(frame_idx)
        speeds = [v["t"] for v in vectors]
        angles = [v["a"] for v in vectors]
        isdowns = [v["isdown"] for v in vectors]
        fastdowns = [1 for v in vectors if v["isdown"] and v["t"] > 6]
        return {
            "time": time.time(),
            "frame_index": frame_idx,
            "fall": 0,
            "features": {
                "vec_num": float(len(vectors)),
                "down_ratio": float(sum(isdowns) / len(vectors)),
                "speed_mean": float(np.mean(speeds)),
                "speed_std": float(np.std(speeds)),
                "angle_mean": float(np.mean(angles)),
                "angle_std": float(np.std(angles)),
                "fastdown_num": float(len(fastdowns))
            }
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

# 실행 코드
"""root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터\VS\영상"
all_videos = glob.glob(os.path.join(root_path, "**", "*.mp4"), recursive=True)
print(len(all_videos))
part_index = 0
merged_data = []
for i, video in enumerate(all_videos):
    print(i+1, os.path.basename(video))
    extractor = OpticalFlowExtractor(video)
    extractor.run()
    merged_data.append({
        "vid_id": i + 1,
        "vid_name": os.path.basename(video),
        "frames": extractor.data
    })
    if (i + 1) % 1000 == 0 or (i + 1) == len(all_videos):
        part_index += 1
        filename = f"data_part{part_index \}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
        merged_data = []"""
        
extractor = OpticalFlowExtractor("people.mp4")
extractor.run()