import os
import torch
import glob
import cv2 as cv

# 영상 파일 열기
video_path = 'slide.mp4'
cap = cv.VideoCapture(video_path)

# 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: 비디오를 열 수 없습니다.")
    exit()

start_frame = 30
end_frame = 70
current_frame = 0

# 프레임 반복
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if current_frame >= start_frame and current_frame <= end_frame:
        cv.imshow('Frame', frame)
        if cv.waitKey(30) & 0xFF == ord('q'):
            break
    elif current_frame > end_frame:
        break

    current_frame += 1

# 리소스 해제
cap.release()
cv.destroyAllWindows()
