from embedding_method.mapper import MemFetcher
import concurrent.futures
import multiprocessing
from tqdm import tqdm
from datasketch import MinHash,MinHashLSH
import jsonlines
import os
import json

class LSH:
    def __init__(self,drive_path,language,embedding_methods,debug=False):
        self.drive_path = drive_path
        self.language = language
        self.debug = debug
        self.embedding_methods = embedding_methods

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.emb_path = self.base_path + "laser/"

        self.embeddings = MemFetcher(lookup_path=self.emb_path+"para2emb.json",data_path=self.emb_path+"paragraph_emb.jsonl")
        self.min_hashes = []
        self.num_perm = 128
        self.lsh = None
        self.threshold = 0.9
        self.paras_to_delete = {}
        self.ids = set()
        self.number_of_paragraphs = 0
        self.info = json.load(open(self.base_path+"wiki_info.json",'r'))

        self._create_min_hash()
        self.create_LSH()
        self.find_duplicates()
        tqdm.write("duplicate pairs found: {}".format(len(self.paras_to_delete.keys())))
        self.delete_in_paragraphs()
        self.replace_in_embeddings()
        self.replace_in_url_wiki()
        self.replace_in_id_wiki()
        self.create_new_emb_lookup()


    def _create_min_hash(self):
        pbar = tqdm(total=len(self.embeddings.lookup.keys()), desc="min hashing paragraphs")
        for para_id,fp in self.embeddings.lookup.items():
            try:
                minh = self.multi_create_min_hash(self.embeddings(para_id), para_id, self.num_perm)
                self.min_hashes.append(minh)
            except:
                pass
            pbar.update()
        pbar.close()


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
                        self.paras_to_delete.update({res_para:cur_para})

    @staticmethod
    def multi_create_min_hash(embedding, id, num_perm):
        mh = MinHash(num_perm=num_perm)
        mh.update(embedding)
        return id, mh

    def delete_in_paragraphs(self):
        tqdm.write("starting on paragraphs")
        current = self.base_path+"paragraphs.jsonl"
        temp = self.base_path+"paragraphs_temp.jsonl"
        temp_writer = jsonlines.open(temp,'w',flush=True)
        tmp_l = 0
        for paragraph in jsonlines.open(current,'r'):
            tmp_l+=1

            if paragraph["id"] in self.paras_to_delete.keys() or paragraph["id"] in self.ids:
                continue
            else:
                temp_writer.write(paragraph)
                self.ids.update([paragraph["id"]])

        final_writer = jsonlines.open(current, 'w',flush=True)
        for paragraph in jsonlines.open(temp, 'r'):
            final_writer.write(paragraph)
            self.number_of_paragraphs +=1

        tqdm.write("number of paras: before:{} after:{}".format(tmp_l,self.number_of_paragraphs))
        tqdm.write("done with paragraphs")
        os.remove(temp)

    def replace_in_embeddings(self):
        tqdm.write("starting on embeddings")
        for emb in self.embedding_methods:

            current = self.base_path + "{}/paragraph_emb.jsonl".format(emb)
            temp = self.base_path + "{}/paragraph_emb_temp.jsonl".format(emb)

            temp_writer = jsonlines.open(temp, 'w',flush=True)
            for embedding in jsonlines.open(current, 'r'):
                if embedding["id"] not in self.paras_to_delete.keys() and embedding["id"] in self.ids:
                    temp_writer.write(embedding)

            final_writer = jsonlines.open(current, 'w',flush=True)
            for paragraph in jsonlines.open(temp, 'r'):
                final_writer.write(paragraph)

            os.remove(temp)
        tqdm.write("done with embeddings")

    def replace_in_url_wiki(self):
        url2wiki = json.load(open(self.base_path+"url2wiki.json",'r'))
        new_url2wiki = {}
        for k in url2wiki.keys():
            wiki = url2wiki.get(k)
            paragraphs = wiki["paragraphs"]

            new_p = []
            for para_id in paragraphs:
                if para_id in self.paras_to_delete.keys():
                    new_p.append(self.paras_to_delete.get(para_id))
                else:
                    new_p.append(para_id)

            wiki.update({"paragraphs":new_p})
            new_url2wiki.update({k:wiki})

        json.dump(new_url2wiki,open(self.base_path+"url2wiki.json",'w'))


    def replace_in_id_wiki(self):
        id2wiki = json.load(open(self.base_path+"wiki.json",'r'))
        new_id2wiki = {}
        for k in id2wiki.keys():
            wiki = id2wiki.get(k)
            paragraphs = wiki["paragraphs"]

            new_p = []
            for para_id in paragraphs:
                if para_id in self.paras_to_delete.keys():
                    new_p.append(self.paras_to_delete.get(para_id))
                else:
                    new_p.append(para_id)

            wiki.update({"paragraphs":new_p})
            new_id2wiki.update({k:wiki})

        json.dump(new_id2wiki,open(self.base_path+"wiki.json",'w'))


    def create_new_emb_lookup(self):

        for emb_meth in self.embedding_methods:

            lookup = {}

            emb_path = self.base_path + "{}/paragraph_emb.jsonl".format(emb_meth)
            temp_emb_path = self.base_path + "{}/paragraph_temp_emb.jsonl".format(emb_meth)
            temp_writer = jsonlines.open(temp_emb_path,'w',flush=True)
            for embedding in jsonlines.open(emb_path, 'r'):
                fp = temp_writer._fp.tell()
                lookup.update({embedding["id"]:fp})
                temp_writer.write(embedding)

            os.remove(temp_emb_path)
            json.dump(lookup,open(self.base_path+"para2emb.json",'w'))

if __name__ == '__main__':
    drive_path = "/media/jonas/archive/master/data/"
    debug = True
    language = "da"
    LSH(drive_path=drive_path,language=language,debug=debug)