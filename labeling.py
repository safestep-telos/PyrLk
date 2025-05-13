import os
import json

x = []
y = []

#data폴더에 우리가 구한 json, label폴더에 데이터셋 json이 있다고 가정
data_file = "./data"
label_file = "./label"

for file_name in os.listdir(data_file):
    # 파일 이름 불러오기
    video_name = file_name.replace("_data.json","")

    #json불러오기
    with open(os.path.join(data_file, file_name), "r") as f:
        data1 = json.load(f)

    #label불러오기
    with open(os.path.join(label_file, video_name + "_label.json"), "r") as f:
        data2 = json.load(f)

    fall_start = data2["sensordata"][0]["fall_start_frame"]
    fall_end = data2["sensordata"][0]["fall_end_frame"]

    #data.json의 프레임별 데이터를 x에 넣기
    for frame in data1:
        x.append(frame["frames"])

        #0,1 구분해서 y에 넣기
        index = frame["frames"][0]["frame_index"]

        if index >= fall_start and index <= fall_end:
            frame["frames"][0]["fall"] = 1
            y.append(1)

        else:
            frame["frames"][0]["fall"] = 0
            y.append(0)
