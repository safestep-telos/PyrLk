import numpy as np
import cv2 as cv
import sys

video_path = "video.mp4" #humanVideo, people, testVideo, video, slide
cap = cv.VideoCapture(video_path)

# params for ShiTomasi corner detection
#feature_params = dict(maxCorners=100,
#                      qualityLevel=0.2,
#                      minDistance=5,
#                      blockSize=7)

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=5,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


prev = None
#ret, old_frame = cap.read()

#prev = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
#p0 = cv.goodFeaturesToTrack(prev, mask=None, **feature_params)


# 속도, 가속도, 아래로 이동했는지 여부
vectors = list()
old_speed = 0
dt = 1 / cap.get(cv.CAP_PROP_FPS)

while True:
    ret, frame = cap.read()

    if not ret:
            print('프레임 획득 실패')
            sys.exit()

    if prev is None:
        prev = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        continue

    step = 16
    points = list()

    for y in range(step // 2, frame.shape[0], step):
        for x in range(step // 2, frame.shape[1], step):
            points.append([x,y])

    p0 = np.array(points, dtype=np.float32).reshape(-1, 1, 2)


    # calculate optical flow
    curr = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #flow = cv.calcOpticalFlowFarneback(prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0) #farneback

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
        acceleration = (speed - old_speed) / dt
        old_speed = speed
        isDownwards = False

        radian = np.arctan2(dy, dx)
        degree = - np.degrees(radian)

        if degree < -45 and degree > -135:
            isDownwards = True

        if speed > 1 and speed < 20: #큰 모션이 있는 곳 빨간색
            # cv.line(img, (x, y), (x+dx, y+dy), (0,0,255), 2)
            cv.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 0, 255), 2, tipLength=0.3)

            vectors.append((speed, acceleration, isDownwards, degree))

            #if acceleration > 500 and isDownwards:  # 낙상판단..
                #print(vectors)

        elif speed < 1:
            cv.line(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2)



    cv.imshow('Optical flow', frame)

    prev = curr
    p0 = good_new.reshape(-1, 1, 2)

    #print(acceleration)

    # esc키를 누르면 종료료
    k = cv.waitKey(30) & 0xff

    if k == 27:
         break

cap.release()
cv.destroyAllWindows()

print(vectors)