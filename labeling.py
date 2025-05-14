import json

x = []
y = []

#json불러오기
with open("data.json", "r") as f:
    data1 = json.load(f)

for video in data1:
    # 영상 이름 불러오기
    video_name = video["vid_name"].replace(".mp4", "")

    #label불러오기
    with open(video_name + "_label.json", "r") as f:
        data2 = json.load(f)

    fall_start = data2["sensordata"][0]["fall_start_frame"]
    fall_end = data2["sensordata"][0]["fall_end_frame"]

    #data.json의 프레임별 데이터를 x에 넣기
    for frame in video["frames"]:
        x.append(frame["features"])

        #0,1 구분해서 y에 넣기
        index = frame["frame_index"]
        if index >= fall_start and index <= fall_end:
            frame["fall"] = 1
            y.append(1)

        else:
            frame["fall"] = 0
            y.append(0)
