import numpy as np
import os
import json

def load_sequences_from_json(data_json_path, label_root_path, max_frames=600):
    with open(data_json_path, "r", encoding="utf-8") as f:
        data1 = json.load(f)

    X_seq = []
    y_seq = []

    for i,video in enumerate(data1):
        label_name = video["vid_name"].replace(".mp4", ".json")
        label_path = None
        for root, dirs, files in os.walk(label_root_path):
            if label_name in files:
                label_path = os.path.join(root, label_name)
                break
        print(i+1,os.path.join(root_path,label_path))
        if label_path is None:
            continue

        with open(label_path, "r", encoding="utf-8") as f:
            data2 = json.load(f)

        fall_start = data2["sensordata"]["fall_start_frame"]
        fall_end = data2["sensordata"]["fall_end_frame"]

        for j, frame in enumerate(video["frames"]):
            if j >= max_frames:
                break
            feature_dict = frame["features"]
            vec = [
                feature_dict["vec_num"],
                feature_dict["down_ratio"],
                feature_dict["speed_mean"],
                feature_dict["speed_std"],
                feature_dict["angle_std"],
                feature_dict["fastdown_num"],
                feature_dict["delta_vec_num"],
                feature_dict["delta_down_ratio"],
                feature_dict["delta_fastdown_num"],
                feature_dict["hip_accel"],
                feature_dict["shoulder_accel"],
                feature_dict["head_accel"],
            ]
            """if all(v == 0.0 for v in vec):
                continue"""
            X_seq.append(vec)
            index = frame["frame_index"]
            y_seq.append(1 if fall_start <= index <= fall_end else 0)

    return np.array(X_seq), np.array(y_seq)

root_path = r"D:\041.낙상사고 위험동작 영상-센서 쌍 데이터\3.개방데이터\1.데이터\Validation\02.라벨링데이터\VL\영상"
X_seq,y_seq = load_sequences_from_json("data.json",root_path)
np.save("X_seq1.npy", X_seq)
np.save("y_seq1.npy", y_seq)
#np.save("X_seq.npy", X_seq)
#np.save("y_seq.npy", y_seq)