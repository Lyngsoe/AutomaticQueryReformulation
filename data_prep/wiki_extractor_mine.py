import xml.etree.ElementTree as etree
import json
import time
import os


def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def extract_wiki_pages(path_wiki_xml = "/home/jonas/data/wiki/dawiki-20190801-pages-articles.xml",language_code="da",debug=False):

    def strip_tag_name(t):
        t = elem.tag
        idx = k = t.rfind("}")
        if idx != -1:
            t = t[idx + 1:]
        return t

    pages = []

    totalCount = 0
    articleCount = 0
    title = None
    start_time = time.time()
    txt_len = 0



    for event, elem in etree.iterparse(path_wiki_xml, events=('start', 'end')):
        tname = strip_tag_name(elem.tag)

        if event == 'start':
            if tname == 'page':
                title = ''
                id = -1
                redirect = ''
                ns = 0
                text=""
        else:
            if tname == 'title':
                title = elem.text
            elif tname == 'id':
                id = int(elem.text)
            elif tname == 'text':
                text = elem.text
            elif tname == 'ns':
                ns = int(elem.text)
            elif tname == 'page':
                totalCount += 1

                if ns == 0 and "#REDIRECT" not in text:
                    articleCount += 1
                    txt_len+=len(text)
                    if len(text) == "" or None:
                        raise Exception("nothing in text")

                    link = language_code+".wikipedia.org/wiki/"+title.replace(" ","_")
                    pages.append({"title":title,"text":text,"url":link})
                if debug and totalCount > 100:
                    break

                if totalCount > 1 and (totalCount % 100000) == 0:
                    print("{:,} pages parsed".format(totalCount))

            elem.clear()

    elapsed_time = time.time() - start_time

    print("Total pages: {:,}".format(totalCount))
    print("Article pages: {:,}".format(articleCount))
    print("mean text length: {}".format(txt_len/articleCount))
    print("Elapsed time: {}".format(hms_string(elapsed_time)))

    return pages