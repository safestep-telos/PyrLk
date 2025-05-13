import numpy as np
import cv2 as cv
import threading
from queue import Queue
from collections import deque
import time
import json
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

########### falltrue, falltrue2, fallfalse, fallfalse2  영상 사용시 45줄 코드 주석 확인!!!
video_path = "falltrue3.mp4"  # humanVideo, people, testVideo, video, slide, falltrue, falltrue2, falltrue3, fallfalse, fallfalse2, fallfalse3, fallfalse4
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(25, 25),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# 큐를 이용해 데이터 전달
frame_queue = Queue()
result_queue = Queue()
fall_queue = deque()
none_queue = deque()

data = list()
frame_count = 0

def calc_optical_flow():
    total_calculation_time = 0.0
    prev = None
    old_speed = 0

    vectors = list()
    vector = {}

    while True:
        frame, frame_num = frame_queue.get()  # 프레임 전달 받음
        if frame is None: break

        frame = frame[:, frame.shape[1] * 4 // 7:, :]  # falltrue, falltrue2, fallfalse, fallfalse2 사용하는 경우만 사용하는 코드

        points = list()

        if prev is None:
            step = 16

            for y in range(step // 2, frame.shape[0], step):
                for x in range(step // 2, frame.shape[1], step):
                    points.append([x, y])
            prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            continue

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)
        new_x_min = None
        new_y_min = None
        new_x_max = None
        new_y_max = None
        if results.pose_landmarks:
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

            new_x_min = int(max(x_center - new_w / 2, 0))
            new_y_min = int(max(y_center - new_h / 2, 0))
            new_x_max = int(min(x_center + new_w / 2, frame.shape[1]))
            new_y_max = int(min(y_center + new_h / 2, frame.shape[0]))

            # 확대된 박스
            p3 = (new_x_min, new_y_min)
            p4 = (new_x_max, new_y_max)
            cv.rectangle(frame, p3, p4, (0, 0, 255), 2)  # 빨간색

        new_time = time.time()
        # old_time = time.time()
        # dt = old_time
        isfall = 0

        for y in range(step // 2, frame.shape[0], step):
            for x in range(step // 2, frame.shape[1], step):
                points.append([x, y])

        time_vector = {"frame" : frame_num, "time": new_time, "fall": isfall, "vectors": []}

        p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        p1, st, err = cv.calcOpticalFlowPyrLK(prev, curr, p0, None, **lk_params)

        # Select good points
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            dx, dy = new.ravel() - old.ravel()

            speed = np.sqrt(dx ** 2 + dy ** 2)
            radian = np.arctan2(dy, dx)
            degree = - np.degrees(radian)
            isDownwards = 0

            if degree < -45 and degree > -135:
                isDownwards = 1

            if speed > 2 and speed < 15:  # 큰 모션이 있는 곳 빨간색
                if new_x_min is None and new_x_max is None and new_y_min is None and new_y_max is None:
                    #mediapipe pose가 감지되지 않았을 때: 전체 화면 추적
                    vector = {"x": float(c), "y": float(d), "t": float(speed),
                              "a": float(degree),
                              "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                    if degree < -60 and degree > -120 and speed >= 6:
                        cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                        vectors.append(vector)
                    #vectors.append(vector)

                elif new_x_min <= c <= new_x_max and new_y_min <= d <= new_y_max:
                    #mediapipep pose가 감지되었을 때: 사람의 중심 영역 안에서만 추적
                    vector = {"x": float(c), "y": float(d), "t": float(speed),
                              "a": float(degree),
                              "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                    if degree < -60 and degree > -120 and speed >= 6:
                        cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                        vectors.append(vector)
                    #vectors.append(vector)

        down_count = sum(1 for v in vectors if v.get("isdown") == 1 and v.get("t") >= 6)
        if down_count > 10:
            fall_queue.append(1)
            none_queue.clear()
        else:
            if len(fall_queue) >= 3:
                fall_queue.append(1)
            none_queue.append(1)
            # fall_queue.clear()

        if len(none_queue) >= 2 and len(fall_queue) >= 3:
            fall_queue.pop()
            fall_queue.pop()
            print(len(fall_queue))
            for k in range(1, len(fall_queue) + 1): #앞뒤로 낙상 프레임인데 가운데 프레임만 아닌 경우 그 프레임도 낙상으로
                data[len(data) - k - 1]["fall"] = 1
            fall_queue.clear()
            # print(len(fall_queue))

        prev = curr
        result_queue.put((frame, vectors))  # optical flow 계산 결과를 전달할 큐

        time_vector["vectors"] = vectors
        if len(fall_queue) >= 3:
            # print(len(fall_queue))
            if len(none_queue) < 1:
                print("fall detect", new_time)  # 이 코드는 디버깅용, 주석 혹은 삭제 처리 예정

        if vectors:
            data.append(time_vector)
            vectors = []


sub_thread = threading.Thread(target=calc_optical_flow)  # 스레드 생성
sub_thread.start()  # 서브 스레드 시작

result_vectors_list = list()  # 프레임별 벡터값 저장
while True:
    ret, frame = cap.read()

    if not ret:
        print('프레임 끝')
        frame_queue.put(None)  # esc 처리 (none 전달 -> 서브 쓰레드 종료 유도)
        break

    frame_queue.put((frame, frame_count))  # 서브 스레드에 프레임 전달
    frame_count += 1

    if not result_queue.empty():
        result = result_queue.get()  # optical flow 계산 결과 받음
        result_frame = result[0]
        result_vectors = result[1]
        # result_vectors_list.append(result_vectors)

        for v in result_vectors:
            c = v["x"];
            d = v["y"];
            ang = v["a"]
            speed = v["t"]
            dx = speed * np.cos(-np.deg2rad(ang))
            dy = speed * np.sin(-np.deg2rad(ang))
            a = c + dx
            b = d + dy
            if ang < -60 and ang > -120:
                cv.arrowedLine(result_frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
        cv.imshow('Optical flow', result_frame)

    # esc키를 누르면 종료
    k = cv.waitKey(30) & 0xff

    if k == 27:
        frame_queue.put(None)
        break

cap.release()
cv.destroyAllWindows()

sub_thread.join()

# print(result_vectors_list)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

# json_data = json.dumps(vectors)
# print(json_data)