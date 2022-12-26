'''
    Find news_tabs of each page and iterate through each article
    Then get the content of that article and push into results
'''
import requests
from bs4 import BeautifulSoup
from opencc import OpenCC
import json
import threading

def page_crawl(start, end, urls, results, source_domain):
    '''
        start: start index of pages
        end: end index of pages
        urls: {link, source_domain}
        results: []
        catCnt: now number of categories
    '''
    try:
        cat_threads = []
        for i, target_url in enumerate(urls["link"]):
            cat_threads.append(threading.Thread(target=cat_crawl, args=(start, end, source_domain, target_url, results)))
            cat_threads[i].start()
                
        for i in range(len(urls["link"])):
            cat_threads[i].join()

        with open(f"nomura_crawl/nomura_{source_domain}.json", 'w') as jsonFile:
            json.dump(results, jsonFile, indent=4)
    except: 
        pass

def cat_crawl(start, end, source_domain, target_url, results):
    for i in range(start, end):
        page = requests.get(target_url.format(i))
        soup = BeautifulSoup(page.content, 'html.parser')
        news_tabs = soup.find(id="MainContent_Contents_ArticleGridList1_gvList")
        links = news_tabs.find_all("tr")[1:]
        for link in links:
            converter = OpenCC("s2t")   
            url = "https://www.moneydj.com/" + link.find("a")["href"]
            publish_time = link.find("td").getText()
            publish_time = publish_time.replace("/", "-")
            publish_time = "2022-" + publish_time
            publish_time = publish_time.strip()
            publish_time = publish_time.replace("\n", "")
            publish_time = publish_time. replace("\r", "")

            article = requests.get(url)
            articleSoup = BeautifulSoup(article.content, 'html.parser')
            
            title = articleSoup.find("span", {"id": "MainContent_Contents_lbTitle"}).getText()
            title = title.strip()
            title = title.strip('\n')
            title = converter.convert(title)
        
            maintext = articleSoup.find("article").find_all("p")
            maintext = map(lambda x: x.getText(), maintext)
            maintext = "".join(maintext)
            maintext = converter.convert(maintext)
            
            info = {
                "date_publish": publish_time,
                "id": "",
                "maintext": maintext,
                "source_domain": source_domain,
                "url": url,
                "title": title
            }
            results.append(info)
            print(f"Now {len(results)} articles")

stock_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0100001',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0100012',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0100014',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0100009'
]

bond_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0500000'
]

future_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0700000'
]

material_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300001',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300002',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300003',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300004',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300007',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300008',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300009',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X0300010',
]

estate_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X1010000'
]

gold_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X1100000'
]

fund_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X1300000'
]

forex_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X1610000'
]

system_pages = [
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000001',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000002',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000003',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000004',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000005',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000006',
    'https://www.moneydj.com/kmdj/common/listnewarticles.aspx?index1={}&svc=NW&a=X2000007',
]

categories = [
    {"link": material_pages, "source_domain": "material"}, 
    {"link": future_pages, "source_domain": "future"}, 
    {"link": fund_pages, "source_domain": "fund"}, 
    {"link": gold_pages, "source_domain": "gold"}, 
    {"link": estate_pages, "source_domain": "estate"}, 
    {"link": bond_pages, "source_domain": "bond"}, 
    {"link": forex_pages, "source_domain": "forex"}, 
    {"link": system_pages, "source_domain": "system"}
]

page_threads = []   
for i in range(len(categories)):
    results = []
    page_threads.append(threading.Thread(target=page_crawl, args=(1, 6, categories[i], results, categories[i]["source_domain"])))
    page_threads[i].start()
        
for i in range(len(categories)):
    page_threads[i].join()
