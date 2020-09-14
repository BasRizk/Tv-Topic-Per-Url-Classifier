# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from bs4.element import Comment

tqdm.pandas()

def scrap_web_text(url):    
    
    def tag_visible(element):
        blacklist = ['[document]', 'script', 'header', 'html', 'meta',
                     'head', 'input', 'sript', 'style',]
        # gray = ['head']
        if element.parent.name in blacklist:
            return False
        # if element.parent.name in gray:
        #     print("gray")
        #     print(element)
        if isinstance(element, Comment):
            return False
        return True

    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64)\
            AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122\
                Safari/537.36", 
        "Accept-Encoding":"gzip, deflate", 
        'Accept': '*/*', 
        "DNT":"1",
        "Connection":"keep-alive", 
        "Upgrade-Insecure-Requests":"1"
    }
    
    requests_session = requests.Session()

    try:
        res = requests_session.get(url, allow_redirects=True,
                                    verify=True, headers=headers, timeout=20)
        
    except requests.exceptions.Timeout:
        print("\n$s : timed out" % url)
        return None
    except requests.exceptions.TooManyRedirects:
        print("\n%s : too many redirects" % url)
        # print("Redirects: " + str(seen))
        # print(res.history)
        return None
    except requests.exceptions.RequestException:
        # catastrophic error. bail.
        return None
    
    html_page = res.text
    soup = BeautifulSoup(html_page, 'html.parser')
    text = soup.find_all(text=True)
    visible_texts = filter(tag_visible, text) 
    
    return u" ".join(t.strip() for t in visible_texts)


def retrieve_webtexts(filename="Dataset.csv", save=False):
    df = pd.read_csv(filename)
    df["text"] = df.progress_apply(lambda row: scrap_web_text(row.link), axis=1)
    if save:
        df.to_csv("Dataset_with_text.csv", encoding='utf-8')
    return df

# text_data_1 = scrap_web_text("https://www.tvnachat.net/series/video/mosalsal-ra7im-ep-8")
# text_data_2 = scrap_web_text("https://www.rotana.video/watch.php?vid=a725098a5")
