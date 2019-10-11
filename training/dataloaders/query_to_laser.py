import re
from embedding_method.mapper import MemFetcher
import numpy as np

class QueryToLaser:
    def __init__(self,base_path,embedder,max_length):
        self.base_path = base_path
        self.max_length = max_length
        self.embedder = embedder
        self.embedding_method = "laser"
        self.wordmmap = MemFetcher(self.base_path + self.embedding_method + "/word2emb.json",self.base_path + self.embedding_method + "/word_emb.jsonl")

    def get_embedding(self, query):
        query_text = query["text"]
        q_emb = []
        query_text = self.clean(query_text)
        for word in query_text.split(" "):
            if word != '':
                try:
                    emb = self.wordmmap(word)

                    q_emb.append(emb)
                except:
                    # print("ofv: ",word)
                    emb = self.embedder(word)[0]
                    q_emb.append(emb)

                assert emb.shape == (1024,), "got: {}".format(emb.shape)

        return q_emb


    def clean(self,text):
        text = re.sub("[/$&+,:;=?@#|'<>.\"^*()%!-]", ' ', text)
        text = text.lower()
        return text

    def padding(self,query,max_len):
        while len(query) < max_len:
            query.append(np.zeros(1024))

        return query[:self.max_length]