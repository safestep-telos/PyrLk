from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score
import json
import os

X,y = None

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

    fall_start = data2["sensordata"]["fall_start_frame"]
    fall_end = data2["sensordata"]["fall_end_frame"]

    #data.json의 프레임별 데이터를 x에 넣기
    for frame in data1:
        X.append(frame["frames"][0]["features"])

        #0,1 구분해서 y에 넣기
        index = frame["frames"][0]["frame_index"]
        if index >= fall_start and index <= fall_end:
            frame["frames"][0]["fall"] = 1
            y.append(1)

        else:
            frame["frames"][0]["fall"] = 0
            y.append(0)

X_train, X_test, y_train, y_test = None #train_test_split(X, y, stratify=y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2 Score: {f2:.4f}")