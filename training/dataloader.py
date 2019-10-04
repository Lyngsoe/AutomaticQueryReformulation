import json
from embedding_method.mapper import MemFetcher
import jsonlines
import numpy as np
from embedding_method.embedders import get_embedder
import re

class Dataloader:
    def __init__(self,data_base_path,embedding_method,language):
        self.language = language
        self.data_base_path = data_base_path
        self.embedding_method = embedding_method

        self.embedder = get_embedder(method=self.embedding_method, language=self.language)
        self.mem_map = MemFetcher(self.data_base_path+self.embedding_method+"/query2emb.json",self.data_base_path+self.embedding_method+"/query_emb.jsonl")
        self.max = len(self.mem_map.lookup.keys())
        self.reader = jsonlines.open(self.data_base_path+"queries.jsonl")
        self.tfidf = json.load(open(self.data_base_path+"wiki2tfidf.json"))
        self.word2id = json.load(open(self.data_base_path+"/word2id.json"))
        self.wordmmap = MemFetcher(self.data_base_path+embedding_method+"/word2emb.json",self.data_base_path+embedding_method+"/word_emb.jsonl")
        self.batch_size = 10
        self.vocab_size = len(self.word2id.keys())
        print(self.max)
        self.n=0
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        q_batch = []
        y_batch = []
        if self.n <= self.max:
            for i in range(self.batch_size):
                if self.n > self.max:
                    return q_batch,y_batch
                else:
                    query = self.reader.read()

                    # X
                    q_batch.append(self.get_query_embedding(query["text"]))
                    #q_batch.append(self.mem_map(query["id"]))

                    #Y
                    top_10 = self.tfidf.get(query["wiki_id"])
                    y_batch.append(self.get_annotation([word[0] for word in top_10]))
                self.n+=1
            return q_batch,y_batch
        else:
            raise StopIteration


    def get_query_embedding(self,query_text):
        q_emb = []
        query_text = re.sub("[/$&+,:;=?@#|'<>.\"^*()%!-]",' ', query_text)
        query_text = query_text.lower()
        for word in query_text.split(" "):

            if word != '':
                try:
                    q_emb.append(self.wordmmap(word))
                except:
                    print("ofv: ",word)
                    q_emb.append(self.embedder([word]))

        return q_emb

    def get_annotation(self,tf_idf):
        annotations = []
        for word in tf_idf:
            w = np.zeros(self.vocab_size)
            w[self.word2id.get(word)] = 1
            annotations.append(w)

        return annotations