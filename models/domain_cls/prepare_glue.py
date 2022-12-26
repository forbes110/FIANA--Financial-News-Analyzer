import json
import pprint
from config import parse_args_news, same_seeds
from pathlib import Path
from typing import Dict, List
import logging

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def label_json(file):
    for instance in file:
        if 'maintext' in instance:
            instance['sentence'] = instance['maintext']
        if 'source_domain' in instance:
            instance['label'] = instance['source_domain']

    for instance in file:
        if 'maintext' in instance:
            instance.pop('maintext')
        if 'source_domain' in instance:
            instance.pop('source_domain') 

    return file

def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)


## make data without source_domain 2 prediction
def clean_data(allData):
    print('Start processing data!')

    clean_file = []
    ## for predict
    no_domain_file = []

    for i, data in enumerate(allData):

        if data['maintext'] == "":
            pass

        ## not in source domain
        if data['source_domain'] not in ['stock', 'crypto', 'future', 'forex', 'gold', 'fund', 'system', 'material', 'bond', 'estate']:
            no_domain_file.append(data)

        else:
            data['id'] = i
            clean_file.append(data)
    print(f"There are {len(clean_file)} clean file and {len(no_domain_file)} no title files")

    return clean_file, no_domain_file




'''
    save intent to index table
'''
def label2idx_save(args_pre):
    intents = set()
    for split in ["train", "eval"]:
        dataset_path = Path(f"./data/news/{split}.json")
        dataset = json.loads(dataset_path.read_text())

        intents.update({instance["source_domain"] for instance in dataset})

    intent2idx = {tag: i for i, tag in enumerate(intents)}
    intent_tag_path = args_pre.cache_dir / "label2idx.json"
    intent_tag_path.write_text(json.dumps(intent2idx, indent=4))
    logging.info(f"Intent label 2 index saved at {str(intent_tag_path.resolve())}")

    return intent2idx

'''
    encode the label to index
'''
def encode_label_save(file, news2idx):
    for instance in file:
        if 'label' in instance:
            instance['label'] = news2idx[instance['label']]
        if 'label' not in instance:
            instance['label'] = 0
    return file



if __name__ == "__main__":
    args_pre = parse_args_news()


    train_file, valid_file = load_json_file(args_pre.train_file), load_json_file(args_pre.valid_file)


    ## label json
    train_file_, valid_file_ = label_json(train_file), label_json(valid_file)#, label_json(test_file)

    news2idx = label2idx_save(args_pre)
    news_idx_path = args_pre.cache_dir / "label2idx.json"
    news2idx: Dict[str, int] = json.loads(news_idx_path.read_text())

    train_file_, valid_file_ = encode_label_save(train_file_, news2idx), \
        encode_label_save(valid_file_, news2idx)#, encode_label_save(test_file_, news2idx)



    save_json(args_pre.cache_dir/"glue_train.json", train_file_)
    save_json(args_pre.cache_dir/"glue_valid.json", valid_file_)
    #save_json(args_pre.cache_dir/"glue_test.json", test_file_)


    pp = pprint.PrettyPrinter(indent=4)

    print('glue_train example:\n')
    pp.pprint(train_file_[:1])

    print('glue_valid example:\n')
    pp.pprint(valid_file_[:1])

    # print('glue_test example:\n')
    # pp.pprint(test_file_[:1])
    

