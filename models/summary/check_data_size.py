import json
import pprint
import random


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # /Users/zhouchengkang/Desktop/WareHouse/Financial-News-Analyzer/models/summary/data/anue_crypto.json
    a = load_json_file('./data/train_file.json') 
    b = load_json_file('./data/valid_file.json') 
    c = load_json_file('./data/test_file.json') 

    
    print("train size",len(a))
    print("valid size",len(b))
    print("test size",len(c))



