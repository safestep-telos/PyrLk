import numpy as np
import cv2 as cv
import mediapipe as mp
import threading
from queue import Queue
import time
import os
import glob
import json

pose_model = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

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

        self.pose = pose_model
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.data = []
        self.frame_idx = 0
        self.pose_landmarks = None

    def calc_optical_flow(self):
        prev = None
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

            grid_points = []
            for y in range(step // 2, height, step):
                for x in range(step // 2, width, step):
                    grid_points.append([x, y])
                    #cv.circle(frame, (x, y), 1, (0, 255, 255), -1) 

            if prev is None:
                prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                continue

            if frame_idx % 3 == 1:
                results = self.pose.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
                if results.pose_landmarks:
                    self.pose_landmarks = results.pose_landmarks

            roi_points = []
            if self.pose_landmarks:
                lm_indices = [0, 23, 24, 15, 16, 27, 28]
                for i in lm_indices:
                    lm = self.pose_landmarks.landmark[i]
                    cx, cy = int(lm.x * width), int(lm.y * height)
                    for x, y in grid_points:
                        if abs(cx - x) <= roi_radius and abs(cy - y) <= roi_radius:
                            roi_points.append([x, y])

            if not roi_points:
                prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
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
                        cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)

            if vectors:
                speeds = [v["t"] for v in vectors]
                angles = [v["a"] for v in vectors]
                isdowns = [v["isdown"] for v in vectors]
                fastdowns = [1 for v in vectors if v["isdown"] and v["t"] > 6]

                self.data.append({
                    "time": time.time(),
                    "frame_index": frame_idx,
                    "fall": 0,
                    "features": {
                        "vec_num": float(len(vectors)),
                        "down_ratio":  float(sum(isdowns) / len(vectors)) if len(vectors) else 0.0,
                        "speed_mean": float(np.mean(speeds)),
                        "speed_std": float(np.std(speeds)),
                        "angle_mean": float(np.mean(angles)),
                        "angle_std": float(np.std(angles)),
                        "fastdown_num": float(len(fastdowns))
                    }
                })

            prev = curr
            self.result_queue.put((frame, vectors))

        print(f"[INFO] Runtime {self.runtime:.2f} 처리 완료. 실행시간: {time.time() - start_time:.2f}초")

    def run(self):
        thread = threading.Thread(target=self.calc_optical_flow)
        thread.start()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_queue.put((frame, self.frame_idx))
            self.frame_idx += 1

            if not self.result_queue.empty():
                result_frame, _ = self.result_queue.get()
                """cv.imshow("Optical Flow", result_frame)

            if cv.waitKey(1) & 0xFF == 27:
                self.frame_queue.put(None)
                break"""

        thread.join()
        self.cap.release()
        #cv.destroyAllWindows()

root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터\VS\영상"
all_videos = glob.glob(os.path.join(root_path, "**", "*.mp4"), recursive=True)

merged_data = []
if len(all_videos) == 0:
    print("!!")
for i, video in enumerate(all_videos):
    print(video)
    vid_id = i + 1
    extractor = OpticalFlowExtractor(video)
    extractor.run()
    merged_data.append({
        "vid_id": vid_id,
        "vid_name": video,
        "frames": extractor.data
    })

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(merged_data, f, ensure_ascii=False, indent=4, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))