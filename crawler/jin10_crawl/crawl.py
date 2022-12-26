'''
    Find news_tabs of each page and iterate through each article
    Then get the content of that article and push into results
'''
import requests
from bs4 import BeautifulSoup
from opencc import OpenCC
import json
import threading

def page_crawl(start, end, results):
    try:
        for i in range(start, end):
            page = requests.get(f"https://xnews.jin10.com/53/page/{i}")
            soup = BeautifulSoup(page.content, 'html.parser')
            news_tabs = soup.find("div", "jin10-news-tabs-body")
            links = news_tabs.find_all("div", "jin10-news-list-item news")
            article_threads = []
            for j, link in enumerate(links):
                article_threads.append(threading.Thread(target=link_crawl, args=(link, results)))
                article_threads[j].start()
            for j in range(len(article_threads)):
                article_threads[j].join()
    except:
        pass

def link_crawl(link, results):
    try:
        converter = OpenCC("s2t")   
        url = link.find("a")["href"]
        publish_time = link.find("span", "jin10-news-list-item-display_datetime").find("span").getText()
        if publish_time.count("-") == 1:
            publish_time = "2022-" + publish_time

        article = requests.get(url)
        articleSoup = BeautifulSoup(article.content, 'html.parser')
        
        title = articleSoup.find("p", "jin10-news-cdetails-title").getText()
        title = title.strip()
        title = title.strip('\n')
        title = converter.convert(title)
    
    
        maintext = articleSoup.find("div", "jin10vip-image-viewer setWebViewConentHeight upload-statics-links").find_all("p")
        maintext = map(lambda x: x.getText(), maintext)
        maintext = "".join(maintext)
        maintext = converter.convert(maintext)
        
        info = {
            "date_publish": publish_time,
            "id": "",
            "maintext": maintext,
            "source_domain": "",
            "url": url,
            "title": title
        }
        results.append(info)
        print(f"Now {len(results)} articles")
    except:
        pass


for x in range(1, 5):
    results = []
    page_threads = []   
    for i in range(1, 10): 
        page_threads.append(threading.Thread(target=page_crawl, args=((x-1)*100+i*10, (x-1)*100+i*10+10, results)))
        page_threads[i-1].start()
        
    for i in range(len(page_threads)):
        page_threads[i].join()
        
    with open(f"股票的東西/jin10_{x}.json", 'w') as jsonFile:
        json.dump(results, jsonFile, indent=4)
