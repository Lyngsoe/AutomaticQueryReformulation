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
from laserembeddings import Laser
import numpy as np

class WikiParser:
    def __init__(self,load_path,save_path,language,debug=False,wiki_data=False):
        self.load_path = load_path
        self.save_path = save_path
        self.embedding_save_path = save_path+"embeddings/"

        os.makedirs(self.save_path,exist_ok=True)

        self.language = language
        self.debug = debug
        self.wiki_data = wiki_data
        self.max_pages = 1000 if self.debug else -1
        self.batch_size = 10 if self.debug else 50

        self.wiki_d = WikiDataloader(max_pages=self.max_pages,base_path=self.load_path)

        self.embeddings = {}

        self.paragraphs = []
        self.queries = []

        self.url2wiki = {}
        self.url2paras = {}
        self.para2emb = {}

        #### RUN LOGIC ####
        self.parse_wiki()
        self._get_embeddings()
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
                        self.queries.extend(q)

                        self.url2paras.update(a)
                        self.url2wiki.update(info)

                    except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))
                        raise exc

    def _get_embeddings(self):
        if self._correct_embeddings():
            tqdm.write("loading existing embeddings")
            self._load_embeddings()
        else:
            tqdm.write("creating new embeddings")
            self._create_embeddings()
            self._save_embeddings()

    def _create_embeddings(self):
        para_text = [para["text"] for para in self.paragraphs]

        embedder = Laser(embedding_options={"max_sentences": self.batch_size})
        embeddings = embedder.embed_sentences(para_text, lang=self.language)

        for i,emb in enumerate(embeddings):
            self.embeddings.update({self.paragraphs[i]["id"]:emb})

    def _save_embeddings(self):
        os.makedirs(self.embedding_save_path,exist_ok=True)
        np.savez(self.embedding_save_path+"paragraph_emb",self.embeddings.update)
        info = {"paragraphs":len(self.paragraphs)}
        json.dump(info,open(self.embedding_save_path+"info.json",'w'))

    def _correct_embeddings(self):
        if os.path.isfile(self.embedding_save_path+"info.json"):
            info = json.load(open(self.embedding_save_path+"info.json",'r'))
            return int(info["paragraphs"]) == len(self.paragraphs)
        else:
            return False

    def _load_embeddings(self):
        self.embeddings = np.load(self.embedding_save_path+"paragraph_emb.npz")

    def _remove_duplicates(self):
        self.paragraphs, self.url2paras = remove_duplicates(self.embeddings,self.paragraphs,self.url2paras)


    def _save(self):
        count_sum = 0
        for v in self.url2paras.values():
            count_sum += len(v)
        avg_paras_per_article = count_sum / len(self.url2paras.keys())

        count_query_paras = 0
        for q in self.queries:
            count_query_paras += len(self.url2paras.get(q["url"]))
        avg_paras_per_query = count_query_paras / len(self.queries)

        tqdm.write("#paragraphs: {}".format(len(self.paragraphs)))
        tqdm.write("#queries: {}".format(len(self.queries)))
        tqdm.write("mean #paragraphs per article: {}".format(avg_paras_per_article))
        tqdm.write("mean #paragraphs per query: {}".format(avg_paras_per_query))

        writer = jsonlines.open(self.save_path + "paragraphs.jsonl", 'w')
        for para in self.paragraphs:
            writer.write(para)

        writer = jsonlines.open(self.save_path + "queries.jsonl", 'w')
        for q in self.queries:
            writer.write(q)

        wiki_info = {
            "language": self.language,
            "paragraphs": len(self.paragraphs),
            "queries": len(self.queries),
            "wikis":len(self.url2wiki.keys()),
            "annotated_wikis":len(self.url2paras.keys()),
            "paragraph_per_art": avg_paras_per_article,
            "paragraph_per_query": avg_paras_per_query
        }

        json.dump(self.url2paras,open(self.save_path + "url2para.json", 'w'))
        json.dump(wiki_info, open(self.save_path + "wiki_info.json", 'w'))
        json.dump(self.url2wiki, open(self.save_path + "url2wiki.json", 'w'))

    def _process_page(self,page,language):
        p, a = extract_paragraphs_and_annotations(page)
        q = parse_queries(page)

        if self.wiki_data:
            try:
                wiki_data_id = get_wikidata_id(page["title"],language)
            except:
                wiki_data_id = ""
        else:
            wiki_data_id = ""

        page_info = {page["url"]:{"title":page["title"],"wikidata_id":wiki_data_id}}
        return p,a,q,page_info



if __name__ == '__main__':

    wiki_lang = "en"

    drive_path = "/media/jonas/archive/"

    load_path = drive_path+"master/data/extracted_wiki/{}/".format(wiki_lang)
    save_path = drive_path+"master/data/raffle_wiki/{}/".format(wiki_lang)

    WikiParser(load_path,save_path,wiki_lang,debug=False)