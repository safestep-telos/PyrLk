import numpy as np
import cv2 as cv
import threading
from queue import Queue
from collections import deque
import time
import json
import mediapipe as mp

import os
import glob

class OpticalFlowExtractor:
    def __init__(self, video_path, output_path):
        self.video_path = video_path
        self.output_path = output_path
        self.cap = cv.VideoCapture(video_path)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.data = []
        self.lk_params = dict(
            winSize=(25, 25),
            maxLevel=5,
            criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.pose = mp.solutions.pose.Pose()
        self.frame_queue = Queue()
        self.result_queue = Queue()
        self.fall_queue = deque()
        self.none_queue = deque()

    def calc_optical_flow(self):
        #total_calculation_time = 0.0    
        prev = None
        #old_speed = 0
        #dt = 1/fps

        vectors = list()
        vector = {}
        time_vector = {}

        while True :
            frame = self.frame_queue.get() #프레임 전달 받음
            if frame is None : break
            
            frame = frame[:, frame.shape[1]*4//7:, :] #falltrue, falltrue2, fallfalse, fallfalse2 사용하는 경우만 사용하는 코드
            
            points = list()

            if prev is None:
                step = 16
            
                for y in range(step // 2, frame.shape[0], step):
                    for x in range(step // 2, frame.shape[1], step):
                        points.append([x, y])
                prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                continue
            
            frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)
            new_x_min = None
            new_y_min = None
            new_x_max = None
            new_y_max = None  
            if results.pose_landmarks   :
                x_min = min([lm.x for lm in results.pose_landmarks.landmark]) * frame.shape[1]
                x_max = max([lm.x for lm in results.pose_landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in results.pose_landmarks.landmark]) * frame.shape[0]
                y_max = max([lm.y for lm in results.pose_landmarks.landmark]) * frame.shape[0]
                
                scale = 2
                w = x_max - x_min
                h = y_max - y_min
                
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2

                new_w = w * scale
                new_h = h * scale

                new_x_min = int(max(x_center - new_w/2, 0))
                new_y_min = int(max(y_center - new_h/2, 0))
                new_x_max = int(min(x_center + new_w/2, frame.shape[1]))
                new_y_max = int(min(y_center + new_h/2, frame.shape[0]))

                # 확대된 박스
                p3 = (new_x_min, new_y_min)
                p4 = (new_x_max, new_y_max)
                cv.rectangle(frame, p3, p4, (0, 0, 255), 2)  # 빨간색
            """prev_rgb = cv.cvtColor(prev,cv.COLOR_BGR2RGB)
            frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
            results_prev = pose.process(prev_rgb)
            results_cur = pose.process(frame_rgb)    
            heads_prev = []; heads_cur=[]
            #waists = []
            if results_prev.pose_landmarks and results_cur.pose_landmarks:
                landmark_prev = results_prev.pose_landmarks.landmark
                landmark_cur = results_cur.pose_landmarks.landmark
                
                for i in [2,5]:
                    lm_p = landmark_prev[i]
                    lm_c = landmark_cur[i]
                    if lm_p.visibility > 0.5:
                        heads_prev.append(lm_p.y*frame.shape[0])
                    if lm_c.visibility > 0.5:
                        heads_cur.append(lm_c.y*frame.shape[0])
                    
            heads_p_y = None
            heads_c_y = None
            waists_y = None
            
            if len(heads_prev) > 0 and len(heads_cur) > 0:
                heads_p_y = sum(heads_prev)/len(heads_prev)
                heads_c_y = sum(heads_cur)/len(heads_cur)
                if np.abs(heads_p_y-heads_c_y) > 10:
                    print("fall detect2")"""

            new_time = time.time()
            #old_time = time.time()
            #dt = old_time
            isfall = 0

            for y in range(step // 2, frame.shape[0], step):
                for x in range(step // 2, frame.shape[1], step):
                    points.append([x, y])
                    
            time_vector = {"time": new_time,"fall": isfall, "vectors": []}

            p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
            curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            p1, st, err = cv.calcOpticalFlowPyrLK(prev, curr, p0, None, **self.lk_params)

            # Select good points
            if p1 is not None:
                good_new = p1[st == 1]
                good_old = p0[st == 1]

            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()

                dx, dy = new.ravel() - old.ravel()

                speed = np.sqrt(dx ** 2 + dy ** 2)
                #acceleration = (speed - old_speed) / dt
                #old_speed = speed
                radian = np.arctan2(dy, dx)
                degree = - np.degrees(radian)
                isDownwards = 0

                if degree < -45 and degree > -135:
                    isDownwards = 1

                if speed > 2 and speed < 15:  # 큰 모션이 있는 곳 빨간색
                    if new_x_min is None and new_x_max is None and new_y_min is None and new_y_max is None:
                        vector = {"x": float(c), "y": float(d), "dx": float(dx), "dy": float(dy), "t": float(speed), "a": float(degree),
                            "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                        #if degree < -60 and degree > -120:
                            #cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                            #vectors.append(vector)
                        vectors.append(vector)
                    elif new_x_min<=c<=new_x_max and new_y_min<=d<=new_y_max:
                        vector = {"x": float(c), "y": float(d), "dx": float(dx), "dy": float(dy), "t": float(speed), "a": float(degree),
                            "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                        #if degree < -60 and degree > -120:
                            #cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                            #vectors.append(vector)
                        vectors.append(vector)

                    """vector = {"x": float(c), "y": float(d), "dx": float(dx), "dy": float(dy), "t": float(speed), "a": float(degree),
                            "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                    #if degree < -60 and degree > -120:
                        #cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                        #vectors.append(vector)
                    vectors.append(vector)"""
                
                """elif speed < 1:
                    cv.circle(frame, (int(c), int(d)),1, (0, 255, 0), -1)"""
            down_count = sum(1 for v in vectors if v.get("isdown") == 1 and v.get("t") >= 6)
            if down_count > 10:
                self.fall_queue.append(1)
                self.none_queue.clear()
            else:
                if len(self.fall_queue) >= 3:
                    self.fall_queue.append(1)
                self.none_queue.append(1)
                #fall_queue.clear()

            if len(self.none_queue) >= 2 and len(self.fall_queue) >= 3:
                self.fall_queue.pop()
                self.fall_queue.pop()           
                print(len(self.fall_queue))
                for k in range(1,len(self.fall_queue)+1):
                    self.data[len(self.data)-k-1]["fall"] = 1
                self.fall_queue.clear()
                #print(len(fall_queue))
            
            prev = curr
            #p0 = good_new.reshape(-1, 1, 2)
            self.result_queue.put((frame, vectors)) # optical flow 계산 결과를 전달할 큐

            time_vector["vectors"] = vectors
            if len(self.fall_queue) >= 3:
                #print(len(fall_queue))
                if len(self.none_queue) < 1:
                    print("fall detect",new_time)    # 이 코드는 디버깅용, 주석 혹은 삭제 처리 예정

            if vectors:
                self.data.append(time_vector)
                vectors = []

    def run(self):
        sub_thread = threading.Thread(target=self.calc_optical_flow)
        sub_thread.start()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)
                break
            self.frame_queue.put(frame)

            if not self.result_queue.empty():
                result_frame, _ = self.result_queue.get()
                cv.imshow("Optical Flow", result_frame)

            if cv.waitKey(30) & 0xFF == 27:
                self.frame_queue.put(None)
                break

        sub_thread.join()
        self.cap.release()
        cv.destroyAllWindows()

        with open(self.output_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\01.원천데이터\VS\영상"

# 하위 모든 폴더에서 .mp4 파일 검색
all_videos = glob.glob(os.path.join(root_path, "**", "*.mp4"), recursive=True)

print(f"[INFO] 총 {len(all_videos)}개의 영상 파일을 찾았습니다.")

video_list = ["falltrue1.mp4", "falltrue2.mp4", "fallfalse1.mp4"]

for video in video_list:
    #output = video.replace(".mp4", ".json")
    extractor = OpticalFlowExtractor(video, output)
    extractor.run()