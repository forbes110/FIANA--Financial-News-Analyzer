'''
    raw data -- train:
    {   
        'date_publish': '2015-03-02 00:00:00',
        'id': '0',
        'maintext': 'str...',
        'source_domain': 'udn.com',
        'split': 'train',
        'title': '榜首進台大醫科卻休學 、27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教'
    }

    public(evaluation)
    {   
        'date_publish': '2015-03-02 00:00:00',
        'id': '0',
        'maintext': 'str...',
        'source_domain': 'udn.com',
        'split': 'dev',
        'title': '榜首進台大醫科卻休學 、27歲拿到法國天文博士 李悅寧跌破眾人眼鏡返台任教'
    }

    private(test)
    {   
        'date_publish': '2021-01-14 00:00:00',
        'id': '21719',
        'maintext': '',
        'source_domain': 'udn.com',
        'split': 'dev'
    }

    submission
    {
        "title": "t",
        "id": "21710"
    }

    need to be:
    cnn_dailymail dataset:
    {
        'id': '0054d6d30dbcad772e20b22771153a2a9cbeaf62',
        'article': '(CNN)...'
        'highlights': 'summary...'
    }

'''
'''
    only train/test this time:
    test: public/private, we get public
'''
import jsonlines, json
import pprint

def load_jsonlines_file(file_path):
    with jsonlines.open(file_path, 'r') as jsonl_f:
        return [obj for obj in jsonl_f]
        # return jsonl_f.read()

def load_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(path, data):
    json.dump(data, open(path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)

def clean_article(article):
    '''
        need to do something here, but not sure if it would be better.
        Maybe the cleaning step would make the model lose its robustness with less noise.
    '''
    return article





