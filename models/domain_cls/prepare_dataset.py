import json
import pprint
import random
from sklearn.model_selection import train_test_split
from collections import Counter


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


# if any(ext in url_string for ext in extensionsToCheck):
def clean_data(allData):
    print('Start processing data!')

    clean_file = []

    for i, data in enumerate(allData):

        if data['maintext'] == "" or data['source_domain'] == "" or data['source_domain'] not in ['stock', 'crypto', 'future', 'forex', 'gold', 'fund', 'system', 'material', 'bond', 'estate']:
            continue

        data['id'] = i
        clean_file.append(data)
    print(f"There are {len(clean_file)} clean file")

    return clean_file
        

def split_data(clean_file):

    label = []
    for doc in clean_file:
        label.append(doc["source_domain"])

    # print(Counter(label))

    train_file, valid_file, _, _ = train_test_split(clean_file, label, test_size=0.1, random_state=42, stratify=label)
        
    print(f'There are {len(train_file)} train_file data')
    print(f'There are {len(valid_file)} valid_file data')

    return train_file, valid_file

if __name__ == "__main__":
    # /Users/zhouchengkang/Desktop/WareHouse/Financial-News-Analyzer/models/summary/data/anue_crypto.json
    a = load_json_file('./data/anue_crypto.json') 
    b = load_json_file('./data/anue_estate.json') 
    c = load_json_file('./data/anue_forex.json') 
    d = load_json_file('./data/anue_stock(tw).json') 
    e = load_json_file('./data/anue_stock(wd).json') 
    f = load_json_file('./data/esunsec.json') 
    g = load_json_file('./data/jin10_1.json') 
    h = load_json_file('./data/jin10_2.json') 
    i = load_json_file('./data/jin10_3.json') 
    j = load_json_file('./data/jin10_4.json') 
    k = load_json_file('./data/nomura_bond.json') 
    l = load_json_file('./data/nomura_estate.json') 
    m = load_json_file('./data/nomura_forex.json') 
    n = load_json_file('./data/nomura_fund.json') 
    o = load_json_file('./data/nomura_future.json') 
    p = load_json_file('./data/nomura_gold.json') 
    q = load_json_file('./data/nomura_material.json') 
    r = load_json_file('./data/nomura_system.json') 
    s = load_json_file('./data/yahoo_fund.json') 
    t = load_json_file('./data/yahoo_international.json') 
    u = load_json_file('./data/yahoo_research.json') 
    v = load_json_file('./data/yahoo_tw.json') 
    
    

    # print(len(a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q+r+s+t+u+v))
    # print(a[0]['maintext'][12])
    all_data_list = [a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v]
    allData = []

    for group in all_data_list:
        for data in group:
            allData.append(data)

    print(f'There are {len(allData)} raw data')

    clean_file = clean_data(allData)
    
    train_data, valid_data = split_data(clean_file)

    save_json('./cache/raw_train_file.json', train_data)
    save_json('./cache/raw_valid_file.json', valid_data)

    print(f'File saved!')



