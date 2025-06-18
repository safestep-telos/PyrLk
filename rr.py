import os
import json

l = ["data_part1.json","data_part2.json","data_part3.json"]
merged = []
for i in l:
    with open(i, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            merged.extend(data)
        else:
            print(f" {i}는 리스트 형태가 아닙니다. 무시됨.")
            
with open("data.json","w",encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)