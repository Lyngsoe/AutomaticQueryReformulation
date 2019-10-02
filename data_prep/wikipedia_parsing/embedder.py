import jsonlines
from tqdm import tqdm
import json
import os
from embedding_method.embedders import get_embedder

class Embedder:
    def __init__(self,drive_path,language,debug=False):
        #self.methods = ["laser","bert"]
        self.methods = ["bert"]
        self.drive_path = drive_path
        self.debug = debug
        self.language = language

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)


        self.batch_size = 10 if self.debug else 200


        self.paragraph_path = self.base_path+"paragraphs.jsonl"
        self.query_path = self.base_path+"queries.jsonl"
        self.info = json.load(open(self.base_path+"wiki_info.json",'r'))

        self.create_embeddings()


    def create_embeddings(self):

        for method in self.methods:
            save_path = self.base_path + "{}/".format(method)
            os.makedirs(save_path, exist_ok=True)
            embedder = get_embedder(method,self.language)
            self.create_paragraph_embeddings(method,save_path,embedder)
            self.create_query_embeddings(method,save_path,embedder)


    def create_paragraph_embeddings(self,method,save_path,embedder):

        writer = jsonlines.open(save_path + "paragraph_emb.jsonl","w")
        para2emb = {}
        paras = []
        pbar = tqdm(total=int(self.info["paragraphs"] / self.batch_size), desc="embedding paragraph batches with {}".format(method))

        for para in jsonlines.open(self.paragraph_path,'r'):
            paras.append(para)

            if len(paras) >= self.batch_size:
                para_text = [p["text"] for p in paras]
                embeddings = embedder(para_text)

                for i, emb in enumerate(embeddings):
                    para2emb.update({paras[i]["id"]:writer._fp.tell()})
                    writer.write({"id":paras[i]["id"],"emb":emb.tolist()})

                paras = []
                pbar.update()

        pbar.close()

        if len(paras) != 0:
            para_text = [p["text"] for p in paras]
            embeddings = embedder(para_text)

            for i, emb in enumerate(embeddings):
                para2emb.update({paras[i]["id"]: writer._fp.tell()})
                writer.write({"id": paras[i]["id"], "emb": emb.tolist()})

        json.dump(para2emb, open(save_path + "para2emb.json", 'w'))


    def create_query_embeddings(self,method,save_path,embedder):
        query2emb = {}
        writer = jsonlines.open(save_path + "query_emb.jsonl","w")

        queries = []
        pbar = tqdm(total=int(self.info["queries"] / self.batch_size), desc="embedding queries batches with {}".format(method))

        for query in jsonlines.open(self.query_path,'r'):
            queries.append(query)

            if len(queries) >= self.batch_size:
                query_txt = [q["text"] for q in queries]
                embeddings = embedder(query_txt)

                for i, emb in enumerate(embeddings):
                    query2emb.update({queries[i]["id"]:writer._fp.tell()})
                    writer.write({"id":queries[i]["id"],"emb":emb.tolist()})

                queries = []
                pbar.update()

        pbar.close()

        if len(queries) != 0:
            query_txt = [q["text"] for q in queries]
            embeddings = embedder(query_txt)

            for i, emb in enumerate(embeddings):
                query2emb.update({queries[i]["id"]: writer._fp.tell()})
                writer.write({"id": queries[i]["id"], "emb": emb.tolist()})


        json.dump(query2emb,open(save_path + "query2emb.json",'w'))