from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score

import json
import numpy as np

root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터\VL\영상"
X = []; y = []

"""with open("data.json", "r",encoding="utf-8") as f:
    data1 = json.load(f)

for i,video in enumerate(data1):
    # 영상 이름 불러오기
    label_name = video["vid_name"].replace(".mp4", ".json")
    label_path = None
    for root, dirs, files in os.walk(root_path):
        if label_name in files:
            label_path = os.path.join(root, label_name)
    
    print(i+1,os.path.join(root_path,label_path))
    #label불러오기
    with open(os.path.join(root_path,label_path), "r",encoding="utf-8") as f:
        data2 = json.load(f)

    fall_start = data2["sensordata"]["fall_start_frame"]
    fall_end = data2["sensordata"]["fall_end_frame"]
    #data.json의 프레임별 데이터를 x에 넣기
    for i,frame in enumerate(video["frames"]):
        if i == 600:
            break
        feature_dict = frame["features"]
        vec = [
            feature_dict["vec_num"],
            feature_dict["down_ratio"],
            feature_dict["speed_mean"],
            feature_dict["speed_std"],
            feature_dict["angle_mean"],
            feature_dict["angle_std"],
            feature_dict["fastdown_num"]
        ]
        if all(v == 0.0 for v in vec):
            continue
        
        
        index = frame["frame_index"]
        if fall_start <= index <= fall_end:
            y.append(1)
        else:
            y.append(0)"""
X_seq = np.load("X_seq.npy")
y_seq = np.load("y_seq.npy")
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, stratify=y_seq, test_size=0.2)

model = RandomForestClassifier(n_estimators=10, class_weight="balanced")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f2 = fbeta_score(y_test, y_pred, beta=2)
print(f"F2 Score: {f2:.4f}")