#!/usr/bin/python
import requests
import sys
import shutil
import re
import os
import threading
from bs4 import BeautifulSoup

THREAD_COUNTER = 0
THREAD_MAX     = 5
IND=1
def get_source( link ):
    r = requests.get( link )
    if r.status_code == 200:
        return BeautifulSoup( r.text , 'lxml')
    else:
        sys.exit( "[~] Invalid Response Received." )

def filter( html ):
    imgs = html.findAll( "img" )
    if imgs:
        return imgs
    else:
        sys.exit("[~] No images detected on the page.")

def requesthandler( link, name ):
    global THREAD_COUNTER
    global IND
    THREAD_COUNTER += 1
    try:
        r = requests.get(link, stream=True)
        if r.status_code == 200:
            r.raw.decode_content = True
            f = open('table_'+ str(IND)+ '.jpg', "wb")
            IND += 1
            shutil.copyfileobj(r.raw, f)
            f.close()
            print ("[*] Downloaded Image: %s" % name)
    except Exception as error:
        print("[~] Error Occured with %s : %s" % (name, error))
    THREAD_COUNTER -= 1

links = ['https://www.ikea.com/in/en/cat/dining-tables-21825/', 'https://www.ikea.com/in/en/cat/dining-tables-21825/page-2/', 'https://www.ikea.com/gb/en/search/?k=Caf%C3%A9%20tables'
,'https://www.ikea.com/in/en/cat/dining-tables-21825/page-3/','https://www.ikea.com/in/en/cat/bar-tables-20862/','https://www.ikea.com/in/en/ikea-business/coffee-and-side-tables-pub6566923b','https://www.ikea.com/in/en/ikea-business/desks-for-stationary-computers-pub98736d31']
for l in links:
    html = get_source(l)
    imgs = filter(html)
    for img in imgs:
        src = img.get("src")
        if src:
            src = re.match( r"((?:https?:\/\/.*)?\/(.*\.(?:png|JPG|jpg)))", src )
            if src:
                (link, name) = src.groups()
                if not link.startswith("http"):
                    link = "https://www.ikea.com/in/en/cat/" + link
                _t = threading.Thread( target=requesthandler, args=(link, name.split("/")[-1]) )
                _t.daemon = True
                _t.start()

                while THREAD_COUNTER >= THREAD_MAX:
                    pass

    while THREAD_COUNTER > 0:
        pass
