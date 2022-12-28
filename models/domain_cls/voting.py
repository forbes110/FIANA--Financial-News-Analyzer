import json
from typing import Dict
import csv
import os
from sklearn import metrics

def idx2label_save():
    intent_idx_path = "./cache/label2idx.json"
    with open(intent_idx_path) as f:
        data = json.load(f)
    return data

def label2index(_label2index, idx: int):  
    return _label2index[str(idx)]

def compute_accuracy(pred,truth):
    c = 0
    for p,t in zip(pred,truth):
        if p == t:
            c = c + 1
    return c/len(pred)

def main():
    _label2index = idx2label_save()


    groundtruth_path = './data/glue_valid.json'
    groundtruth_label = []
    groundtruth_data = []
    with open(groundtruth_path, 'r') as f:
        data = json.load(f)
        for doc in data:
            groundtruth_data.append(doc["id"])
            groundtruth_label.append(doc["label"])

    results = []
    for filename in os.listdir("./result"):
        with open(os.path.join("./result", filename), newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',')
            _ = next(rows)
            ids = []
            labels = []
            for row in rows:
                ids.append(int(row[0]))
                labels.append(label2index(_label2index,row[1]))
            results.append((ids,labels))
    print(len(results))

    for ins in results:
        print(compute_accuracy(ins[1],groundtruth_label))
    
    print("--voting result--")
    voted = []
    for cnt in range(len(results[0][1])):
        agg = []
        for ins in results:
            agg.append(ins[1][cnt])
        voted.append(results[4][1][cnt] if results[0][1][cnt]==results[1][1][cnt]==results[2][1][cnt]==results[4][1][cnt] else results[3][1][cnt])
    print(compute_accuracy(voted,groundtruth_label))
    #print(metrics.classification_report(y_validate, predicted_res))


if __name__ == "__main__":
    main()
