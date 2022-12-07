import json
import pprint
import random


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


# if any(ext in url_string for ext in extensionsToCheck):
def clean_data(allData):
    print('Start processing data!')

    clean_file = []
    ## for predict
    no_title_file = []

    for i, data in enumerate(allData):

        if data['maintext'] == "":
            pass

        if ("報告" in data['title']) or ("操作策略" in data['title']):
            data['id'] = i
            no_title_file.append(data)
        else:
            data['id'] = i
            clean_file.append(data)
    print(f"There are {len(clean_file)} clean file and {len(no_title_file)} no title files")

    return clean_file, no_title_file
        

def split_data(clean_file):
    random.seed(42)
    random.shuffle(clean_file)

    train_len = int(len(clean_file)*0.8)
    valid_len = int(len(clean_file)*0.1)
    train_file, valid_file, test_file = clean_file[:train_len], clean_file[train_len:train_len+valid_len], clean_file[train_len+valid_len:-1]
    
    print(f'There are {len(train_file)} train_file data')
    print(f'There are {len(valid_file)} valid_file data')
    print(f'There are {len(test_file)} test_file data')

    return train_file, valid_file, test_file

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

    clean_file, no_title_file = clean_data(allData)
    
    train_data, valid_data, test_data = split_data(clean_file)

    save_json('./data/train_file.json', train_data)
    save_json('./data/valid_file.json', valid_data)
    save_json('./data/test_file.json', test_data)

    print(f'file saved!')



