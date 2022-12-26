# coding=utf8
'''
    {
        "id": {generate},
        "sentence": "太陽能模組廠茂迪 (6244-TW)、元晶 (6443-TW)、安集 (6477-TW) 今 (7) 日皆公布第三季營收，受惠政府政策加持、年底併網高峰潮到來，各家模組業者訂單皆已滿到明年，加上新品單價高，元晶、安集第三季營收皆衝上歷史新高。元晶 9 月營收改寫歷史第三高，達 7.73 億元，月增 3.78%，年增 62.48%，第三季營收創下歷史新高、約 22.76 億元，季增 20.36%，年增 61.88%，累計前三季營收更已超越去年全年的 61.6 億元，達 61.7 億元，年增 57.55%。元晶表示，目前大尺寸新品 M6、M10 電池模組產線都已建置完畢，尤其隨著大尺寸逐步取代 G1 成為國內案場主流，相關訂單已排至明年，訂單能見度高，且隨著先前無法轉嫁成本的訂單消化完畢，下半年獲利可望顯著好轉。安集此次包括 9 月、第三季與前三季營收全寫新猷，一共改寫三新高，9 月營收 4.03 億元，月增 11.98%，年增 190.62%，第三季營收 10.29 億元，季增，年增，累計前三季營收 24.26 億元，年增 193.53%。安集表示，隨著原物料價格趨於平穩，客戶建置電廠意願增加，公司為滿足客戶需求已進行擴產，以今年來看，模組產能已全產全銷，看好下半年營運將顯著優於上半年。茂迪 9 月營收 5.29 億元，月增 3.54%，年減 19.4%，第三季營收 13.88 億元，季增 19.35%，年減 11.77%，累計前三季營收 36.5 億元年減 15.31%。茂迪近年積極布局高技術門檻的差異化產品，首款 N 型 TOPCON 電池模組因發電效率高，可有效節省土地面積，獲客戶大量採用，加上政府積極衝刺太陽光電裝置量，不僅今年底前產能全產全銷，明年上半年訂單也已排滿。",
        "label": 0
    },
'''

import pandas as pd
from opencc import OpenCC
import json

converter = OpenCC('s2t')
raw_data = pd.read_csv("./data/raw_data.csv", encoding="utf-8")
text = raw_data["text"]
label = raw_data['label']

processed_data = []
for e, i in enumerate(text):
    dic = {
        "id": e,
        "sentence": i,
        "label": str(label[e])
    }
    processed_data.append(dic)
    
converted_json = json.dump(processed_data, open("./data/processed_data.json", 'w', encoding='utf-8'), indent=4, ensure_ascii=False)