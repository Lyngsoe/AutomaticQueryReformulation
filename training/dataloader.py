import json
from embedding_method.mapper import MemFetcher
import jsonlines

class Dataloader:
    def __init__(self,data_base_path,embedding_method):
        self.data_base_path = data_base_path
        self.embedding_method = embedding_method
        self.mem_map = MemFetcher(self.data_base_path+self.embedding_method+"/query2emb.json",self.data_base_path+self.embedding_method+"/query_emb.jsonl")
        self.max = len(self.mem_map.lookup.keys())
        self.reader = jsonlines.open(self.data_base_path+"queries.jsonl")
        self.batch_size = 10
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        q_batch = []
        if self.n <= self.max:
            for i in range(self.batch_size):
                if self.n <= self.max:
                    return q_batch
                else:
                    q_id = self.reader.read()["id"]
                    q_batch.append(self.mem_map(q_id))
        else:
            raise StopIteration