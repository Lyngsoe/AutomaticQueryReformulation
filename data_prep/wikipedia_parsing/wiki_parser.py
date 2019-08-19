from data_prep.wikipedia_parsing.wiki_dataloader import WikiDataloader
from data_prep.wikipedia_parsing.wiki_page_parser import parse_queries,extract_paragraphs_and_annotations
from data_prep.lsh import remove_duplicates
from data_prep.wikipedia_parsing.utils import get_wikidata_id
import concurrent.futures
import multiprocessing
import jsonlines
import json
import os
from tqdm import tqdm

class WikiParser:
    def __init__(self,load_path,save_path,language,debug=False):
        self.load_path = load_path
        self.save_path = save_path

        os.makedirs(self.save_path,exist_ok=True)

        self.language = language
        self.debug = debug
        self.max_pages = 100 if self.debug else -1
        self.batch_size = 10 if self.debug else 100

        self.wiki_d = WikiDataloader(max_pages=self.max_pages,base_path=self.load_path)

        self.paragraphs = []
        self.annotations = {}
        self.queries = []
        self.page_info = {}

        #### RUN LOGIC ####
        self.parse_wiki()
        self._remove_duplicates()
        self._save()

    def parse_wiki(self):

        while True:

            batch = self.wiki_d.get_next_batch()

            if batch is None:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                future_to_min_hash = {executor.submit(self._process_page, page, self.language): page for page in batch}
                for future in concurrent.futures.as_completed(future_to_min_hash):
                    results = future_to_min_hash[future]
                    try:
                        p, a, q, info = future.result()

                        self.paragraphs.extend(p)
                        self.annotations.update(a)
                        self.queries.extend(q)
                        self.page_info.update(info)

                    except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))
                        raise exc


    def _remove_duplicates(self):
        self.paragraphs, self.annotations = remove_duplicates(self.paragraphs, self.annotations,language=self.language,batch_size=self.batch_size)


    def _save(self):
        count_sum = 0
        for v in self.annotations.values():
            count_sum += len(v)
        avg_paras_per_article = count_sum / len(self.annotations.keys())

        count_query_paras = 0
        for q in self.queries:
            count_query_paras += len(self.annotations.get(q["url"]))
        avg_paras_per_query = count_query_paras / len(self.queries)

        tqdm.write("#paragraphs: {}".format(len(self.paragraphs)))
        tqdm.write("#queries: {}".format(len(self.queries)))
        tqdm.write("mean #paragraphs per article: {}".format(avg_paras_per_article))
        tqdm.write("mean #paragraphs per query: {}".format(avg_paras_per_query))

        writer = jsonlines.open(self.save_path + "paragraphs.json", 'w')
        for para in self.paragraphs:
            writer.write(para)

        writer = jsonlines.open(self.save_path + "annotations.json", 'w')
        for wiki, paras in self.annotations.items():
            writer.write({wiki: paras})

        writer = jsonlines.open(self.save_path + "queries.json", 'w')
        for q in self.queries:
            writer.write(q)

        wiki_info = {
            "language": self.language,
            "paragraphs": len(self.paragraphs),
            "queries": len(self.queries),
            "paragraph_per_art": avg_paras_per_article,
            "paragraph_per_query": avg_paras_per_query
        }

        json.dump(wiki_info, open(self.save_path + "wiki_info.json", 'w'))
        json.dump(self.page_info, open(self.save_path + "page_info.json", 'w'))

    def _process_page(self,page,language):
        p, a = extract_paragraphs_and_annotations(page)
        q = parse_queries(page)

        try:
            wiki_data_id = get_wikidata_id(page["title"],language)
        except:
            wiki_data_id = ""

        page_info = {page["url"]:{"title":page["title"],"wikidata_id":wiki_data_id}}
        return p,a,q,page_info



if __name__ == '__main__':

    wiki_lang = "da"

    load_path = "/home/jonas/data/wiki/extracted_wiki/{}/".format(wiki_lang)
    save_path = "/home/jonas/data/wiki/raffle_wiki/{}/".format(wiki_lang)

    WikiParser(load_path,save_path,wiki_lang,debug=True)