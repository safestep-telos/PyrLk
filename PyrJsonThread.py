import numpy as np
import cv2 as cv
import threading
from queue import Queue
from collections import deque
import time
import json
"""import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils"""

########### falltrue, falltrue2, fallfalse, fallfalse2  영상 사용시 45줄 코드 주석 확인!!! 
video_path = "falltrue.mp4" #humanVideo, people, testVideo, video, slide, falltrue, falltrue2, falltrue3, fallfalse, fallfalse2, fallfalse3, fallfalse4
cap = cv.VideoCapture(video_path)
fps = cap.get(cv.CAP_PROP_FPS)
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(25, 25),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

#dt = 1 / cap.get(cv.CAP_PROP_FPS)

#큐를 이용해 데이터 전달
frame_queue = Queue()
result_queue = Queue()
fall_queue = deque()
none_queue = deque()

data = list()

def calc_optical_flow():
    total_calculation_time = 0.0    
    prev = None
    old_speed = 0
    dt = 1/fps

    vectors = list()
    vector = {}
    time_vector = {}

    while True :
        frame = frame_queue.get() #프레임 전달 받음
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
            #acceleration = (speed - old_speed) / dt
            #old_speed = speed
            radian = np.arctan2(dy, dx)
            degree = - np.degrees(radian)
            isDownwards = 0

            if degree < -45 and degree > -135:
                isDownwards = 1

            if speed > 2 and speed < 15:  # 큰 모션이 있는 곳 빨간색
                
                #vectors.append((speed,acceleration, isDownwards, degree))

                vector = {"x": float(c), "y": float(d), "dx": float(dx), "dy": float(dy), "t": float(speed), "a": float(degree),
                          "isdown": isDownwards}  # x좌표, y좌표, 속도, 각도, 아래 방향 여부
                #if degree < -60 and degree > -120:
                    #cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                    #vectors.append(vector)
                vectors.append(vector)
            
            """elif speed < 1:
                cv.circle(frame, (int(c), int(d)),1, (0, 255, 0), -1)"""
        down_count = sum(1 for v in vectors if v.get("isdown") == 1 and v.get("t") > 4)
        if down_count > 10:
            fall_queue.append(1)
            none_queue.clear()
        else:
            if len(fall_queue) >= 6:
                fall_queue.append(1)
            none_queue.append(1)
            #fall_queue.clear()

        if len(none_queue) >= 2 and len(fall_queue) >= 6:
            fall_queue.pop()
            fall_queue.pop()           
            print(len(fall_queue))
            for k in range(1,len(fall_queue)+1):
                data[len(data)-k-1]["fall"] = 1
            fall_queue.clear()
            #print(len(fall_queue))
        
        prev = curr
        #p0 = good_new.reshape(-1, 1, 2)
        result_queue.put((frame, vectors)) # optical flow 계산 결과를 전달할 큐

        time_vector["vectors"] = vectors
        if len(fall_queue) >= 6:
            #print(len(fall_queue))
            if len(none_queue) < 1:
                print("fall detect",new_time)    # 이 코드는 디버깅용, 주석 혹은 삭제 처리 예정

        if vectors:
            data.append(time_vector)
            vectors = []
            
sub_thread = threading.Thread(target=calc_optical_flow)  # 스레드 생성
sub_thread.start() # 서브 스레드 시작

result_vectors_list = list() #프레임별 벡터값 저장
while True: 
    ret, frame = cap.read()

    if not ret:
        print('프레임 끝')
        frame_queue.put(None)   # esc 처리 (none 전달 -> 서브 쓰레드 종료 유도)
        break

    frame_queue.put(frame) #서브 스레드에 프레임 전달

    if not result_queue.empty():
        result = result_queue.get() # optical flow 계산 결과 받음
        result_frame = result[0]
        result_vectors = result[1]
        #result_vectors_list.append(result_vectors)
        
        for v in result_vectors:
            c = v["x"]; d = v["y"]; a = v["dx"]+c; b=v["dy"]+d
            ang = v["a"]
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

#print(result_vectors_list)

with open("data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

#json_data = json.dumps(vectors)
#print(json_data)