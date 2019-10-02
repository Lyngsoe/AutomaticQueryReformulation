from data_prep.wikipedia_parsing.topic_classification.bfs_engine import BFS
import json
from data_prep.wikipedia_parsing.topic_classification.utils import get_wikidata_id
from tqdm import tqdm
import html
import concurrent.futures
import multiprocessing
import urllib.parse

def get_topic(title,language):
    categories_id = {'Q6642719': 'Category:Academic_disciplines', 'Q6353120': 'Category:Business', 'Q5550686': 'Category:Concepts', 'Q6478924': 'Category:Crime', 'Q2944929': 'Category:Culture', 'Q9715089': 'Category:Economy', 'Q4103249': 'Category:Education', 'Q8413436': 'Category:Energy', 'Q6337045': 'Category:Entertainment', 'Q7214908': 'Category:Events', 'Q5645580': 'Category:Food_and_drink', 'Q1457673': 'Category:Geography', 'Q54070': 'Category:Government', 'Q7486603': 'Category:Health', 'Q1457595': 'Category:History', 'Q6697416': 'Category:Human_behavior', 'Q6172603': 'Category:Humanities', 'Q2945448': 'Category:Knowledge', 'Q1458484': 'Category:Language', 'Q4026563': 'Category:Law', 'Q5550747': 'Category:Life', 'Q4619': 'Category:Mathematics', 'Q5850187': 'Category:Military', 'Q6643238': 'Category:Mind', 'Q8255': 'Category:Music', 'Q4049293': 'Category:Nature', 'Q6576895': 'Category:Objects', 'Q5613113': 'Category:Organizations', 'Q4047087': 'Category:People', 'Q1983674': 'Category:Philosophy', 'Q4103183': 'Category:Politics', 'Q1457903': 'Category:Religion', 'Q1458083': 'Category:Science', 'Q1457756': 'Category:Society', 'Q1457982': 'Category:Sports', 'Q4884546': 'Category:Technology', 'Q52075235': 'Category:Universe', 'Q7386634': 'Category:World'}


    bfs = BFS(categories_id,language)

    topic = bfs.search(title)
    print(topic," : ",title)
    return topic


def add_wiki_id_to_info(base_path,language):
    url2wiki = json.load(open(base_path+"url2wiki.json",'r'))
    url2para =json.load(open(base_path + "url2para.json", 'r'))
    wiki_info = json.load(open(base_path + "wiki_info.json", 'r'))

    id2para = {}
    title2para = {}

    id2wiki = {}
    title2wiki = {}

    pbar = tqdm(total=int(wiki_info["annotated_wikis"]),desc="adding wikidata id's")

    for url in url2para.keys():
        page_info = url2wiki.get(url)
        title = page_info["title"]

        identifier = get_wikidata_id(title,language)

        page_info.update({"wikidata_id":identifier,"url":url})

        id2wiki.update({identifier:page_info})
        title2wiki.update({title:page_info})
        url2wiki.update({url:page_info})

        id2para.update({identifier:url2para.get(url)})
        title2para.update({title:url2para.get(url)})

        pbar.update()

    json.dump(id2wiki, open(base_path + "id2wiki.json", 'w'))
    json.dump(title2wiki, open(base_path + "title2wiki.json", 'w'))
    json.dump(url2wiki, open(base_path + "url2wiki.json", 'w'))
    json.dump(id2para, open(base_path + "id2para.json", 'w'))
    json.dump(title2para, open(base_path + "title2para.json", 'w'))


def add_wiki_id_to_info_concurrent(base_path,language):
    url2wiki = json.load(open(base_path+"url2wiki.json",'r'))
    url2para =json.load(open(base_path + "url2para.json", 'r'))
    wiki_info = json.load(open(base_path + "wiki_info.json", 'r'))

    id2para = {}
    title2para = {}

    id2wiki = {}
    title2wiki = {}

    error_log = []

    pbar = tqdm(total=int(wiki_info["annotated_wikis"]),desc="adding wikidata id's")

    urls = list(url2para.keys())

    with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
        future_to_page = {executor.submit(_update_page_info,url2wiki.get(url),url2para.get(url),url,language): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_page):
            results = future_to_page[future]
            try:
                id2wiki_page,title2wiki_page,url2wiki_page,id2para_page,title2para_page = future.result()

                id2wiki.update(id2wiki_page)
                title2wiki.update(title2wiki_page)
                url2wiki.update(url2wiki_page)

                id2para.update(id2para_page)
                title2para.update(title2para_page)

            except Exception as exc:
                #print('%r generated an exception: %s' % (results, exc))
                error_log.append({"exc":str(exc),"results":str(results)})

            pbar.update()



    json.dump(id2wiki, open(base_path + "id2wiki.json", 'w'))
    json.dump(title2wiki, open(base_path + "title2wiki.json", 'w'))
    json.dump(url2wiki, open(base_path + "url2wiki.json", 'w'))
    json.dump(id2para, open(base_path + "id2para.json", 'w'))
    json.dump(title2para, open(base_path + "title2para.json", 'w'))
    json.dump(error_log,open(base_path+"error_log.json",'w'))

def _update_page_info(page_info,paras,url,language):
    title = page_info["title"]
    title = html.unescape(title)
    title = urllib.parse.quote(title)

    identifier = get_wikidata_id(title, language)

    page_info.update({"wikidata_id": identifier, "url": url})

    id2wiki = {identifier: page_info}
    title2wiki = {title: page_info}
    url2wiki = {url: page_info}

    id2para = {identifier: paras}
    title2para = {title: paras}

    return id2wiki,title2wiki,url2wiki,id2para,title2para


if __name__ == '__main__':

    wiki_lang = "da"

    drive_path = "/media/jonas/archive/"

    base_path = drive_path+"master/data/raffle_wiki/{}/".format(wiki_lang)
    add_wiki_id_to_info_concurrent(base_path,wiki_lang)