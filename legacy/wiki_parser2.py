from data_prep.wikipedia_parsing.wiki_dataloader import WikiDataloader
from data_prep.wikipedia_parsing.wiki_page_parser import parse_queries,extract_paragraphs_and_annotations
from legacy.lsh import find_duplicates,delete_para_from_annotations,delete_paras_from_paragraphs
from data_prep.wikipedia_parsing.topic_classification.utils import get_wikidata_id
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
        self.temp_save_path = self.save_path + "temp/"
        self.embedding_save_path = save_path+"embeddings/"

        os.makedirs(self.temp_save_path,exist_ok=True)
        os.makedirs(self.embedding_save_path, exist_ok=True)

        self.language = language
        self.debug = debug
        self.wiki_data = wiki_data
        self.max_pages = 100 if self.debug else -1
        self.batch_size = 10 if self.debug else 300

        self.number_of_paras = 0
        self.number_of_wikis = 0
        self.number_of_queries = 0

        self.wiki_d = WikiDataloader(max_pages=self.max_pages,base_path=self.load_path)

        #### RUN LOGIC ####
        self.parse_wiki()
        embeddings = self._get_embeddings()
        self._remove_duplicates(embeddings)
        self._save_final_info()

    def parse_wiki(self):

        paragraphs = []
        queries = []

        url2para = {}
        url2wiki = {}

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

                        paragraphs.extend(p)
                        queries.extend(q)

                        url2para.update(a)
                        url2wiki.update(info)

                    except Exception as exc:
                        print('%r generated an exception: %s' % (results, exc))
                        raise exc



        wiki_info = {
            "language": self.language,
            "paragraphs": len(paragraphs),
            "queries": len(queries),
            "wikis": len(url2wiki.keys()),
            "annotated_wikis": len(url2para.keys()),
        }

        self.number_of_paras = len(paragraphs)
        self.number_of_wikis = len(url2wiki.keys())
        self.number_of_queries = len(queries)

        ## Temp save
        self._save_paragraphs(self.temp_save_path,paragraphs)
        self._save_annotations(self.temp_save_path,url2para)


        ## Final save
        self._save_queries(self.save_path,queries)
        json.dump(wiki_info, open(self.temp_save_path + "wiki_info.json", 'w'))
        json.dump(url2wiki, open(self.save_path + "url2wiki.json", 'w'))

        tqdm.write("#paragraphs: {}".format(len(paragraphs)))
        tqdm.write("#queries: {}".format(len(queries)))
        tqdm.write("#wikis: {}".format(len(url2wiki.keys())))

    def _get_embeddings(self):
        if self._correct_embeddings():
            tqdm.write("loading existing embeddings")
            embeddings = self._load_embeddings()
        else:
            tqdm.write("creating new embeddings")
            embeddings = self._create_embeddings()
            self._save_embeddings(embeddings)

        return embeddings

    def _create_embeddings(self):
        reader = jsonlines.open(self.temp_save_path+"paragraphs.jsonl", 'r')
        embedder = Laser(embedding_options={"max_sentences": self.batch_size})
        embeddings = {}
        paras = []
        amount_of_paras = self.number_of_paras
        pbar = tqdm(total=int(amount_of_paras/self.batch_size),desc="embedding paragraph batches")
        for para in reader:
            paras.append(para)

            if len(paras) >= self.batch_size:
                para_text = [p["text"] for p in paras]
                emb = embedder.embed_sentences(para_text, lang=self.language)

                for i,e in enumerate(emb):
                    embeddings.update({paras[i]["id"]:emb})

                paras = []
                pbar.update()
        pbar.close()

        if len(paras) != 0:
            para_text = [p["text"] for p in paras]
            emb = embedder.embed_sentences(para_text, lang=self.language)

            for i, e in enumerate(emb):
                embeddings.update({paras[i]["id"]: emb})

        return embeddings

    def _save_embeddings(self,embeddings):
        np.savez(self.embedding_save_path+"paragraph_emb",embeddings)
        info = {"paragraphs":self.number_of_paras}
        json.dump(info,open(self.embedding_save_path+"info.json",'w'))

    def _correct_embeddings(self):
        if os.path.isfile(self.embedding_save_path+"info.json"):
            info = json.load(open(self.embedding_save_path+"info.json",'r'))
            return self.number_of_paras == info["paragraphs"]
        else:
            return False

    def _remove_duplicates(self,embeddings):
        paras_to_delete = find_duplicates(embeddings)

        paragraphs = self._load_paragraphs(self.temp_save_path)
        paragraphs = delete_paras_from_paragraphs(paras_to_delete,paragraphs)

        tqdm.write("#duplicate paragraphs deleted: {}".format(len(paras_to_delete)))

        self.number_of_paras = len(paragraphs)
        self._save_paragraphs(self.save_path,paragraphs)

        url2para = self._load_annotations(self.temp_save_path)
        url2para = delete_para_from_annotations(paras_to_delete,url2para)
        self.number_of_annotated_wiki = len(url2para.keys())
        self._save_annotations(self.save_path,url2para)


    def _save_final_info(self):

        wiki_info = {
            "language": self.language,
            "paragraphs": self.number_of_paras,
            "queries": self.number_of_queries,
            "wikis":self.number_of_wikis,
            "annotated_wikis":self.number_of_annotated_wiki,
        }

        json.dump(wiki_info, open(self.save_path + "wiki_info.json", 'w'))

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

    def _load_embeddings(self):
        return np.load(self.embedding_save_path+"paragraph_emb.npz",allow_pickle=True)

    def _load_paragraphs(self,path):
        paragraphs = []
        pbar = tqdm(total=self.number_of_paras,desc="loading paragraphs")
        for para in jsonlines.open(path+"paragraphs.jsonl",'r'):
            paragraphs.append(para)
            pbar.update()
        pbar.close()
        return paragraphs

    def _load_annotations(self,path):
        return json.load(open(path+"url2para.json",'r'))

    def _load_queries(self,path):
        queries = []
        pbar = tqdm(total=self.number_of_queries, desc="loading queries")
        for q in jsonlines.open(path+"queries.jsonl",'r'):
            queries.append(q)
            pbar.update()

        pbar.close()
        self.number_of_queries = len(queries)
        return queries

    def _save_paragraphs(self,path,paragraphs):
        pbar = tqdm(total=self.number_of_paras, desc="saving paragraphs")
        writer = jsonlines.open(path+"paragraphs.jsonl",'w')
        for para in paragraphs:
            writer.write(para)
            pbar.update()

        pbar.close()

    def _save_annotations(self,path,url2para):
        json.dump(url2para,open(path+"url2para.json",'w'))

    def _save_queries(self,path,queries):
        pbar = tqdm(total=self.number_of_paras, desc="saving queries")
        writer = jsonlines.open(path+"queries.jsonl",'w')
        for q in queries:
            writer.write(q)
            pbar.update()
            
        pbar.close()


if __name__ == '__main__':

    wiki_lang = "en"

    drive_path = "/media/jonas/archive/"

    load_path = drive_path+"master/data/extracted_wiki/{}/".format(wiki_lang)
    save_path = "/home/jonas/data/raffle_wiki/{}/".format(wiki_lang)

    WikiParser(load_path,save_path,wiki_lang,debug=False)