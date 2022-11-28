import json
import pprint
from config import parse_args_prepare


def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args_pre = parse_args_prepare()

    