# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
# import browser_cookie3

def scrap_web_text(url, requests_session = None):    
    
    def tag_visible(element):
        from bs4.element import Comment
        blacklist = ['[document]', 'script', 'header', 'html', 'meta',
                     'head', 'input', 'sript']
        # gray =  ['style', 'head', 'meta', 'title',]
        if element.parent.name in blacklist:
            return False
        if isinstance(element, Comment):
            return False
        return True

    if requests_session is None:
        headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36", 
            "Accept-Encoding":"gzip, deflate", 
            'Accept': '*/*', 
            "DNT":"1",
            "Connection":"keep-alive", 
            "Upgrade-Insecure-Requests":"1"
        }
        # domain_name = "." + ".".join(url.split(".")[-2:]).split("/")[0]
        # cookies = browser_cookie3.chrome(domain_name=domain_name)
        requests_session = requests.Session()
        # s.allow_redirects = 1
        # s.max_redirects = 100

    try:
        # res = requests_session.get(url, allow_redirects=True,
        #                              verify=True, headers=headers,
        #                              cookies=cookies, timeout=3)
        seen = set()
        while True:
            res = requests_session.get(url, allow_redirects=False,
                                        verify=True, headers=headers, timeout=3)
            url = res.headers['location']
            if url in seen:
                raise requests.exceptions.TooManyRedirects 
            seen.add(url)
            
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

# requests_session = requests.Session(cookies=get_cookies("https://www.tvnachat.net/series/video/mosalsal-ra7im-ep-8"))
# requests
df = pd.read_csv("Dataset.csv")
tqdm.pandas()
# df["text"] = df.progress_apply(lambda row: scrap_web_text(row.link, requests_session=requests_session), axis=1)

text_data_1 = scrap_web_text("https://www.tvnachat.net/series/video/mosalsal-ra7im-ep-8")
text_data_2 = scrap_page("https://www.rotana.video/watch.php?vid=a725098a5")































