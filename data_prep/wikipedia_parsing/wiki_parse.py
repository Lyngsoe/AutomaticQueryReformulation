from data_prep.wikipedia_parsing.wiki_dataloader import WikiDataloader
from data_prep.wikipedia_parsing.wiki_page_parser import parse_queries, extract_paragraphs_and_annotations
import concurrent.futures
import multiprocessing
import jsonlines
import json
import os
from tqdm import tqdm

class WikiParser:
    def __init__(self, drive_path,language, debug=False):
        self.language = language
        self.debug = debug

        self.save_path = drive_path+"raffle_wiki/{}/debug/".format(language) if debug else drive_path+"raffle_wiki/{}/".format(language)
        os.makedirs(self.save_path, exist_ok=True)

        self.max_pages = 100 if self.debug else -1

        self.number_of_paras = 0
        self.number_of_wikis = 0
        self.number_of_queries = 0

        self.wiki_d = WikiDataloader(max_pages=self.max_pages, base_path=drive_path+"extracted_wiki/{}/".format(language))

        #### RUN LOGIC ####
        self.parse_wiki()

    def parse_wiki(self):

        paragraphs = []
        queries = []
        url2wiki = {}

        while True:

            batch = self.wiki_d.get_next_batch()

            if batch is None:
                break

            with concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                future_to_min_hash = {executor.submit(self._process_page, page): page for page in batch}
                for future in concurrent.futures.as_completed(future_to_min_hash):
                    results = future_to_min_hash[future]
                    try:
                        p, q, info = future.result()

                        paragraphs.extend(p)
                        queries.extend(q)
                        url2wiki.update(info)

                    except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))
                        raise exc

        wiki_info = {
            "language": self.language,
            "paragraphs": len(paragraphs),
            "urlqueries": len(queries),
            "urlwikis": len(url2wiki.keys())
        }

        self.number_of_paras = len(paragraphs)
        self.number_of_wikis = len(url2wiki.keys())
        self.number_of_queries = len(queries)

        self._save_paragraphs(self.save_path, paragraphs)
        self._save_queries(self.save_path, queries)

        json.dump(wiki_info, open(self.save_path + "wiki_info.json", 'w'))
        json.dump(url2wiki, open(self.save_path + "url2wiki.json", 'w'))

        tqdm.write("#paragraphs: {}".format(len(paragraphs)))
        tqdm.write("#queries: {}".format(len(queries)))
        tqdm.write("#wikis: {}".format(len(url2wiki.keys())))

    def _process_page(self, page):
        p, a = extract_paragraphs_and_annotations(page)
        if a is not []:
            q = parse_queries(page)
            page_info = {page["url"]: {"title": page["title"],"paragraphs":a}}

            if len(q) > len(a):
                return p,q[:len(a)],page_info
            else:
                return p, q, page_info
        else:
            return [],[],{}

    def _save_paragraphs(self, path, paragraphs):
        pbar = tqdm(total=self.number_of_paras, desc="saving paragraphs")
        writer = jsonlines.open(path + "paragraphs.jsonl", 'w')
        for para in paragraphs:
            writer.write(para)
            pbar.update()

        pbar.close()

    def _save_annotations(self, path, url2para):
        json.dump(url2para, open(path + "url2para.json", 'w'))

    def _save_queries(self, path, queries):
        pbar = tqdm(total=self.number_of_queries, desc="saving queries")
        writer = jsonlines.open(path + "urlqueries.jsonl", 'w')
        for q in queries:
            writer.write(q)
            pbar.update()

        pbar.close()

