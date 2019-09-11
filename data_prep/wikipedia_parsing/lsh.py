from embedding_method.mapper import MemFetcher
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from datasketch import MinHash,MinHashLSH
import jsonlines
import os
import json

class LSH:
    def __init__(self,drive_path,language,debug=False):
        self.drive_path = drive_path
        self.language = language
        self.debug = debug
        self.embedding_methods = ["laser"]

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.emb_path = self.base_path + "laser/"

        self.embeddings = MemFetcher(lookup_path=self.emb_path+"para2emb.json",data_path="paragraph_emb.jsonl")
        self.min_hashes = []
        self.num_perm = 128
        self.lsh = None
        self.threshold = 0.9
        self.paras_to_delete = {}
        self.number_of_paragraphs = 0
        self.info = json.load()

        self.create_min_hash()
        self.create_LSH()
        self.find_duplicates()

    def create_min_hash(self):


        number_of_cpus = multiprocessing.cpu_count()

        with tqdm(total=len(self.embeddings.lookup.keys()), desc="min hashing paragraphs") as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=number_of_cpus) as executor:
                future_to_min_hash = {
                    executor.submit(self.multi_create_min_hash, self.embeddings(para_id), para_id, num_perm=self.num_perm): (para_id, fp) for
                    para_id, fp in self.embeddings.lookup.items()}
                for future in concurrent.futures.as_completed(future_to_min_hash):
                    results = future_to_min_hash[future]
                    try:
                        self.min_hashes.append(future.result())
                    except Exception as exc:
                        tqdm.write('%r generated an exception: %s' % (results, exc))
                        raise exc

                    pbar.update()
            pbar.close()

    def create_LSH(self):
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

        for id, mh in tqdm(self.min_hashes, desc="adding MinHash to LSH index"):
            if id not in self.lsh.keys:
                self.lsh.insert("{}".format(id), mh)

    def find_duplicates(self):
        for id, mh in tqdm(self.min_hashes, desc="finding similar paragraphs"):
            result = self.lsh.query(mh)
            cur_para = id
            try:
                result.remove(id)
            except ValueError:
                pass

            if len(result) > 0:
                for res in result:
                    res_para = res
                    if (cur_para, res_para) not in self.paras_to_delete and (res_para, cur_para) not in self.paras_to_delete:
                        # print("replace:",delete,"with:",cur_para)
                        self.paras_to_delete.update({res_para:cur_para})

    @staticmethod
    def multi_create_min_hash(embedding, id, num_perm):
        mh = MinHash(num_perm=num_perm)
        mh.update(embedding)
        return id, mh

    def delete_in_paragraphs(self):

        current = self.base_path+"paragraphs.jsonl"
        temp = self.base_path+"paragraphs_temp.jsonl"

        temp_writer = jsonlines.open(temp,'w')
        for paragraph in jsonlines.open(current,'r'):
            if paragraph["id"] not in self.paras_to_delete.keys():
                temp_writer.write(paragraph)

        final_writer = jsonlines.open(current, 'w')
        for paragraph in jsonlines.open(temp, 'r'):
            final_writer.write(paragraph)
            self.number_of_paragraphs +=1

        os.remove(temp)

    def replace_in_embeddings(self):

        for emb in self.embedding_methods:

            current = self.base_path + "{}/paragraphs_emb.jsonl".format(emb)
            temp = self.base_path + "{}/paragraphs_emb_temp.jsonl".format(emb)

            temp_writer = jsonlines.open(temp, 'w')
            for embedding in jsonlines.open(current, 'r'):
                if embedding["id"] not in self.paras_to_delete.keys():
                    temp_writer.write(embedding)

            final_writer = jsonlines.open(current, 'w')
            for paragraph in jsonlines.open(temp, 'r'):
                final_writer.write(paragraph)

            os.remove(temp)

    def replace_in_url_wiki(self):
        url2wiki = json.load(open(self.base_path+"url2wiki.json",'r'))

        for k in url2wiki.keys():
            wiki = url2wiki.get(k)
            paragraphs = wiki["paragraphs"]

            to_replace = {}
            for i,para_id in enumerate(paragraphs):
                if para_id in self.paras_to_delete.keys():
                    to_replace.update({i:self.paras_to_delete.get(para_id)})

            for k,v in to_replace.items():
                paragraphs[k] = v

            wiki["paragraphs"] = paragraphs
            url2wiki.update(wiki)

        json.dump(url2wiki,open(self.base_path+"url2wiki.json",'w'))


    def replace_in_id_wiki(self):
        id2wiki = json.load(open(self.base_path+"wiki.json",'r'))

        for k in id2wiki.keys():
            wiki = id2wiki.get(k)
            paragraphs = wiki["paragraphs"]

            to_replace = {}
            for i,para_id in enumerate(paragraphs):
                if para_id in self.paras_to_delete.keys():
                    to_replace.update({i:self.paras_to_delete.get(para_id)})

            for k,v in to_replace.items():
                paragraphs[k] = v

            wiki["paragraphs"] = paragraphs
            id2wiki.update(wiki)

        json.dump(id2wiki,open(self.base_path+"wiki.json",'w'))


    def create_new_emb_lookup(self):

        for emb_meth in self.embedding_methods:

            lookup = {}

            emb_path = self.base_path + "{}/paragraphs_emb.jsonl".format(emb_meth)

            fp = 0
            reader = jsonlines.open(emb_path, 'r')
            for embedding in reader:
                lookup.update({embedding["id"]:fp})
                fp = reader._fp.tell()

            json.dump(lookup,open(emb_path,'w'))
