import json
import os


path = './source_json'
all_data = []
for filename in os.listdir(path):
    print(filename)
    with open(os.path.join(path, filename), 'r') as f:
        data = json.load(f)
        all_data.extend(data)

with open(os.path.join('./', 'all.json'),'w', newline='') as f:
    json.dump(all_data, f, indent=4)

label = []
for doc in all_data:
    label.append(doc["source_domain"])

from sklearn.model_selection import train_test_split
x_train_vec, x_validate_vec, y_train, y_validate = train_test_split(all_data, label, test_size=0.1, random_state=3, stratify=label)

with open(os.path.join('./', 'train.json'),'w', newline='') as f:
    json.dump(x_train_vec, f, indent=4)

with open(os.path.join('./', 'eval.json'),'w', newline='') as f:
    json.dump(x_validate_vec, f, indent=4)