import jsonlines
from laserembeddings import Laser
from tqdm import tqdm
import json
import os
import mmap
import numpy as np

class Embedder:
    def __init__(self,drive_path,language,method="laser",debug=False):
        self.method = method
        self.drive_path = drive_path
        self.debug = debug
        self.language = language

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.save_path = self.base_path + "{}/".format(self.method)
        os.makedirs(self.save_path,exist_ok=True)

        self.batch_size = 10 if self.debug else 10


        self.paragraph_path = self.base_path+"paragraphs.jsonl"
        self.query_path = self.base_path+"queries.jsonl"
        self.info = json.load(open(self.base_path+"wiki_info.json",'r'))
        self.embedder = Laser() #Laser(embedding_options={"max_sentences": self.batch_size})

        self.para2emb = {}
        self.query2emb = {}

        self.create_paragraph_embeddings()
        self.create_query_embeddings()

        json.dump(self.para2emb,open(self.save_path + "para2emb.json",'w'))
        json.dump(self.query2emb,open(self.save_path + "query2emb.json",'w'))

    def create_paragraph_embeddings(self):
        writer = jsonlines.open(self.save_path + "paragraph_emb.jsonl","w")
        #laser =



        paras = []
        pbar = tqdm(total=int(self.info["paragraphs"] / self.batch_size), desc="embedding paragraph batches")

        for para in jsonlines.open(self.paragraph_path,'r'):
            paras.append(para)

            if len(paras) >= self.batch_size:
                para_text = [p["text"] for p in paras]
                embeddings = self.embedder.embed_sentences(para_text, lang=self.language)

                for i, emb in enumerate(embeddings):
                    self.para2emb.update({paras[i]["id"]:writer._fp.tell()})
                    writer.write({"id":paras[i]["id"],"emb":emb.tolist()})

                paras = []
                pbar.update()

        pbar.close()

        if len(paras) != 0:
            para_text = [p["text"] for p in paras]
            embeddings = self.embedder.embed_sentences(para_text, lang=self.language)

            for i, emb in enumerate(embeddings):
                self.para2emb.update({paras[i]["id"]: writer._fp.tell()})
                writer.write({"id": paras[i]["id"], "emb": emb.tolist()})


    def create_query_embeddings(self):
        writer = jsonlines.open(self.save_path + "query_emb.jsonl","w")

        queries = []
        pbar = tqdm(total=int(self.info["queries"] / self.batch_size), desc="embedding queries batches")

        for query in jsonlines.open(self.query_path,'r'):
            queries.append(query)

            if len(queries) >= self.batch_size:
                query_txt = [q["text"] for q in queries]
                embeddings = self.embedder.embed_sentences(query_txt, lang=self.language)

                for i, emb in enumerate(embeddings):
                    self.query2emb.update({queries[i]["id"]:writer._fp.tell()})
                    writer.write({"id":queries[i]["id"],"emb":emb.tolist()})

                queries = []
                pbar.update()

        pbar.close()

        if len(queries) != 0:
            query_txt = [q["text"] for q in queries]
            embeddings = self.embedder.embed_sentences(query_txt, lang=self.language)

            for i, emb in enumerate(embeddings):
                self.query2emb.update({queries[i]["id"]: writer._fp.tell()})
                writer.write({"id": queries[i]["id"], "emb": emb.tolist()})