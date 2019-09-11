import json
import requests
from tqdm import tqdm
import jsonlines
import concurrent.futures
import multiprocessing


class WikiIDCreator:
    def __init__(self,drive_path,language,debug):
        self.drive_path = drive_path
        self.language = language
        self.debug = debug
        self.wiki = {}
        self.para2wiki = {}
        self.url2id = {}
        self.wiki_no_id = []
        self.number_of_queries = 0

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(self.language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)

        self.info = json.load(open(self.base_path+"wiki_info.json",'r'))

        ### RUN LOGIC ###
        self.create_wiki()
        self.create_queries()

        tqdm.write("#queries: {}".format(self.number_of_queries))
        tqdm.write("#wikis: {}".format(len(self.wiki.keys())))
        tqdm.write("#wikis_no_id: {}".format(len(self.wiki_no_id)))

        self.info.update({
            "wikis":len(self.wiki.keys()),
            "wikis_no_id":len(self.wiki_no_id),
            "queries":self.number_of_queries
        })

        json.dump(self.wiki,open(self.base_path+"wiki.json",'w'))
        json.dump(self.para2wiki,open(self.base_path+"para2wiki.json",'w'))
        json.dump(self.wiki_no_id,open(self.base_path+"wikiurlnoid.json",'w'))
        json.dump(self.info,open(self.base_path+"wiki_info.json",'w'))



    def create_wiki(self):

        urlwikis = json.load(open(self.base_path+"url2wiki.json",'r'))
        pbar = tqdm(total=self.info["urlwikis"], desc="getting wiki id's")
        with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            future_to_id = {executor.submit(self.get_id,url,info): (url,info) for (url,info) in urlwikis.items()}
            for future in concurrent.futures.as_completed(future_to_id):
                results = future_to_id[future]
                try:
                    wiki_id,url,info = future.result()

                    if wiki_id is not None:
                        info.update({"url": url})
                        self.wiki.update({wiki_id: info})
                        self.url2id.update({url: wiki_id})
                        for p in info["paragraphs"]:
                            self.para2wiki.update({p: wiki_id})
                except Exception as exc:
                    print('%r generated an exception: %s' % (results, exc))
                    raise exc

                pbar.update()

        pbar.close()

    def get_id(self,url,info):
        wiki_id = self.request_id(url,info["title"])

        return wiki_id,url,info

    def create_queries(self):

        query_writer = jsonlines.open(self.base_path+"queries.jsonl",'w')
        pbar = tqdm(total=self.info["urlqueries"],desc="creating queries")

        for query in jsonlines.open(self.base_path+"urlqueries.jsonl",'r'):

            wiki_id = self.url2id.get(query["url"],None)

            if wiki_id is not None:
                query.update({"wiki_id":wiki_id})
                query_writer.write(query)
                self.number_of_queries+=1

            pbar.update()

        pbar.close()


    def request_id(self,wiki_url,title):
        url = "https://{}.wikipedia.org/w/api.php?action=query&format=json&prop=pageprops&ppprop=wikibase_item&redirects=1&titles={}".format(self.language, title)
        try:
            resp = requests.get(url)
            body = resp.json()

            query = body["query"]
            pages = query["pages"]
            if len(list(pages.keys())) != 1:
                raise Exception("multiple pages returned: {}".format(body))

            key = list(pages.keys())[0]
            page = pages[key]
            pageprops = page["pageprops"]
            wikidata_id = pageprops["wikibase_item"]
        except:
            self.wiki_no_id.append({"url":wiki_url,"title":title})
            return None

        return wikidata_id