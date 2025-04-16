import numpy as np
import cv2 as cv
import threading
from queue import Queue
import time

video_path = "video.mp4" #humanVideo, people, testVideo, video, slide
cap = cv.VideoCapture(video_path)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

#dt = 1 / cap.get(cv.CAP_PROP_FPS)

#큐를 이용해 데이터 전달
frame_queue = Queue()
result_queue = Queue()
def calc_optical_flow():
    prev = None
    old_speed = 0
    old_time = time.time()
    while True :
        frame = frame_queue.get() #프레임 전달 받음
        if frame is None : break

        vectors = list()

        if prev is None:
            prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            continue

        step = 16
        points = list()

        for y in range(step // 2, frame.shape[0], step):
            for x in range(step // 2, frame.shape[1], step):
                points.append([x, y])

        p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        timestamp = time.time() - old_time
        old_time = time.time()
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
            acceleration = (speed - old_speed) / timestamp
            radian = np.arctan2(dy, dx)
            degree = - np.degrees(radian)
            isDownwards = False

            if degree < -45 and degree > -135:
                isDownwards = True

            if speed > 1 and speed < 20:  # 큰 모션이 있는 곳 빨간색
                cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)
                vectors.append((speed,acceleration, isDownwards, degree))

            elif speed < 1:
                cv.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)

        prev = curr
        #p0 = good_new.reshape(-1, 1, 2)
        result_queue.put((frame, vectors)) # optical flow 계산 결과를 전달할 큐

sub_thread = threading.Thread(target=calc_optical_flow)  # 스레드 생성
sub_thread.start() # 서브 스레드 시작

result_vectors_list = list() #프레임별 벡터값 저장

while True:
    ret, frame = cap.read()

    if not ret:
        print('프레임 끝')
        frame_queue.put(None)
        break

    frame_queue.put(frame) #서브 스레드에 프레임 전달

    if not result_queue.empty():
        result = result_queue.get() # ptical flow 계산 결과 받음
        result_frame = result[0]
        result_vectors = result[1]
        result_vectors_list.append(result_vectors)
        cv.imshow('Optical flow', result_frame)

    # esc키를 누르면 종료료
    k = cv.waitKey(30) & 0xff

    if k == 27:
        break

cap.release()
cv.destroyAllWindows()

print(result_vectors_list)