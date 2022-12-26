import json
import pprint
from utils.config import parse_args_news, same_seeds
from pathlib import Path
from typing import Dict, List
import logging

def load_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
    # with open(file_path, 'r') as f:
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
        del data['date_publish']
        del data['url']
        data['id'] = i
        clean_file.append(data)
    print(f"There are {len(clean_file)} clean file")

    return clean_file


'''
    save intent to index table
'''
def label2idx_save(args_pre):
    intents = set()
    for split in ["raw_train_file", "raw_valid_file"]:
        dataset_path = Path(f"./cache/{split}.json")
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

    # news2idx = label2idx_save(args_pre)
    news_idx_path = args_pre.cache_dir / "label2idx.json"
    news2idx: Dict[str, int] = json.loads(news_idx_path.read_text(encoding='utf-8'))

    train_file_, valid_file_ = encode_label_save(train_file_, news2idx), \
        encode_label_save(valid_file_, news2idx)#, encode_label_save(test_file_, news2idx)
        
    train_file_, valid_file_ = clean_data(train_file_), clean_data(valid_file_)
    # f = open("data.txt","r",encoding="utf-8")


    save_json("./data/glue_train.json", train_file_)
    save_json("./data/glue_valid.json", valid_file_)
    #save_json(args_pre.cache_dir/"glue_test.json", test_file_)


    pp = pprint.PrettyPrinter(indent=4)

    # print('glue_train example:\n')
    # pp.pprint(train_file_[:1])

    # print('glue_valid example:\n')
    # pp.pprint(valid_file_[:1])

    # print('glue_test example:\n')
    # pp.pprint(test_file_[:1])
    

'''
    {
        "date_publish": "2022-11-22 23:46:11",
        "id": 13566,
        "maintext": "【時報記者林資傑台北報導】金管會推動「鼓勵境外基金深耕計畫」邁入第10年，為鼓勵境外基金機構長期在台經營資產管理業務、並促進永續發展，在參酌投信投顧公會及業者意見後，宣布將修正相關規定，放寬境外基金機構適用優惠措施的認可期間，有8家業者受惠。為鼓勵境外基金機構增加對台投入，共同參與發展台灣資產管理市場，金管會2013年起推動「鼓勵境外基金深耕計畫」，強化對台灣資產管理人才培育，符合深耕計畫一定條件的業者，可適用各項優惠措施。金管會近期參酌投信投顧公會及業者意見，將修正「鼓勵境外基金深耕計畫」及問答集，針對長期取得深耕計畫認可的業者，放寬適用優惠措施的認可期間，並在問答集明定鼓勵境外基金機構協助總代理人或在台據點發展ESG業務評估標準。證期局主秘黃厚銘說明，此次修正有2大重點。首先，為因應業者需求、鼓勵境外基金機構長期在台經營資產管理業務，若境外基金機構連3年獲金管會認可，可在第3次獲認可的次年，申請認可有效期間從1年延長為2年。黃厚銘表示，目前連3年獲金管會認可的境外基金業者，包括聯博、安聯、施羅德、景順、摩根、野村、NNIP及富蘭克林等8家業者。在此次完成規定修正後，上述8家業者將可在明年6月底前，提出深耕計畫適用2年優惠措施的認可期間申請。其次，配合3月發布的「證券期貨業永續發展轉型執行策略」，金管會此次在境外基金管理辦法問答集「鼓勵境外基金深耕計畫」增訂釋例，說明與ESG相關的「其他具體績效貢獻」事項的評估標準，以促進台灣資產管理業者的永續發展，為所有境外基金業者均適用。",
        "source_domain": "fund",
        "url": "https://tw.stock.yahoo.com/news/%E9%87%91%E8%9E%8D-%E5%A2%83%E5%A4%96%E5%9F%BA%E9%87%91%E6%B7%B1%E8%80%95%E8%A8%88%E7%95%AB%E5%8A%A0%E7%A2%BC-8%E6%A5%AD%E8%80%85%E5%84%AA%E6%83%A0%E5%8F%AF%E9%81%A9%E7%94%A82%E5%B9%B4-234611295.html",
        "title": "《金融》境外基金深耕計畫加碼 8業者優惠可適用2年"
    },


    {
        "id": 4945,
        "title": "太陽能迎併網高峰潮 元晶、安集Q3營收飆新高",
        "sentence": "太陽能模組廠茂迪 (6244-TW)、元晶 (6443-TW)、安集 (6477-TW) 今 (7) 日皆公布第三季營收，受惠政府政策加持、年底併網高峰潮到來，各家模組業者訂單皆已滿到明年，加上新品單價高，元晶、安集第三季營收皆衝上歷史新高。元晶 9 月營收改寫歷史第三高，達 7.73 億元，月增 3.78%，年增 62.48%，第三季營收創下歷史新高、約 22.76 億元，季增 20.36%，年增 61.88%，累計前三季營收更已超越去年全年的 61.6 億元，達 61.7 億元，年增 57.55%。元晶表示，目前大尺寸新品 M6、M10 電池模組產線都已建置完畢，尤其隨著大尺寸逐步取代 G1 成為國內案場主流，相關訂單已排至明年，訂單能見度高，且隨著先前無法轉嫁成本的訂單消化完畢，下半年獲利可望顯著好轉。安集此次包括 9 月、第三季與前三季營收全寫新猷，一共改寫三新高，9 月營收 4.03 億元，月增 11.98%，年增 190.62%，第三季營收 10.29 億元，季增，年增，累計前三季營收 24.26 億元，年增 193.53%。安集表示，隨著原物料價格趨於平穩，客戶建置電廠意願增加，公司為滿足客戶需求已進行擴產，以今年來看，模組產能已全產全銷，看好下半年營運將顯著優於上半年。茂迪 9 月營收 5.29 億元，月增 3.54%，年減 19.4%，第三季營收 13.88 億元，季增 19.35%，年減 11.77%，累計前三季營收 36.5 億元年減 15.31%。茂迪近年積極布局高技術門檻的差異化產品，首款 N 型 TOPCON 電池模組因發電效率高，可有效節省土地面積，獲客戶大量採用，加上政府積極衝刺太陽光電裝置量，不僅今年底前產能全產全銷，明年上半年訂單也已排滿。",
        "label": 5
    },
'''