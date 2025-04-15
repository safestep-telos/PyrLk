import numpy as np
import cv2 as cv
import sys
import time

from collections import deque

q = deque(maxlen=5)

video_path = "fall.mp4" #humanVideo, people, testVideo, video, slide
cap = cv.VideoCapture(video_path)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


ret, old_frame = cap.read()

old_t = time.time()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

step = 16
h,w = old_frame.shape[:2]
idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.int64)
indices =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,2)

idx_y,idx_x = np.mgrid[step/2:h:step,step/2:w:step].astype(np.float32)
p0 =  np.stack( (idx_x,idx_y), axis =-1).reshape(-1,1,2)
p1= None

# 속도, 가속도, 아래로 이동했는지 여부
vectors = list()
speed = np.zeros((h, w), dtype=np.float32)
old_speed = np.zeros((h, w), dtype=np.float32)
idx = 0
while True:
    idx += 1
    ret, frame = cap.read()

    if not ret:
            print('프레임 획득 실패')
            sys.exit()
            
    new_t = time.time()
    
    dt = new_t - old_t
    
    if dt == 0:
        continue

    if old_gray is None:
        old_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if p1 is not None:
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, p1, **lk_params, flags=cv.OPTFLOW_USE_INITIAL_FLOW|cv.OPTFLOW_LK_GET_MIN_EIGENVALS)
    else:
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params,flags=cv.OPTFLOW_LK_GET_MIN_EIGENVALS)

    # Select good points
    if p1 is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]

    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()

        dx, dy = new.ravel() - old.ravel()

        speed[int(d),int(c)] = np.sqrt(dx ** 2 + dy ** 2)
        #acceleration = (speed - old_speed) / dt
        isDownwards = False

        radian = np.arctan2(dy, dx)
        degree = - np.degrees(radian)

        if degree < -45 and degree > -135:
            isDownwards = True

        if speed[int(d),int(c)] > 1 and speed[int(d),int(c)] < 20: #큰 모션이 있는 곳 빨간색
            # cv.line(img, (x, y), (x+dx, y+dy), (0,0,255), 2)
            cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)

            vectors.append((speed,isDownwards, degree))
        cnt = 0
        for v in vectors:
            if v[1] == True:
                cnt += 1
        
        if cnt > 3:
            q.append(idx)
            
        if len(q) == 5 and q[4]-q[0] == 4:
            print("!!")
            

    cv.imshow('Optical flow', frame)

    old_gray = frame_gray
    old_t = new_t

    # esc키를 누르면 종료
    k = cv.waitKey(1) & 0xff

    if k == 27:
         break

cap.release()
cv.destroyAllWindows()

#print(vectors)