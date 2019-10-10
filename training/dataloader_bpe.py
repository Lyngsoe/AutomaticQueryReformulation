import json
from embedding_method.mapper import MemFetcher
import jsonlines
import numpy as np
from embedding_method.embedders import get_embedder
import re

class DataloaderBPE:
    def __init__(self,drive_path,language,embedder,embedding_method,batch_size=1,fold=0,debug=False,eval=False):
        self.language = language
        self.base_path = drive_path+"raffle_wiki/{}/debug/".format(language) if debug else drive_path+"raffle_wiki/{}/".format(language)
        self.embedding_method = embedding_method
        self.embedder = embedder
        self.fold = fold
        self.eval = eval
        self.mem_map = MemFetcher(self.base_path+self.embedding_method+"/query2emb.json",self.base_path+self.embedding_method+"/query_emb.jsonl")
        self.max = len(self.mem_map.lookup.keys())
        self.reader = jsonlines.open(self.base_path+"{}-queries.jsonl".format(self.fold))
        self.tfidf = json.load(open(self.base_path+"wiki2tfidf.json"))
        self.bpe2id = json.load(open(self.base_path+"bpe2id.json"))
        self.id2bpe = json.load(open(self.base_path + "id2bpe.json"))
        self.wordmmap = MemFetcher(self.base_path+embedding_method+"/word2emb.json",self.base_path+embedding_method+"/word_emb.jsonl")
        self.batch_size = batch_size
        self.vocab_size = len(self.bpe2id.keys())
        #print(self.vocab_size)
        self.max_length = 20
        self.n=0
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        q_batch = []
        y_batch = []
        queries = []
        targets = []
        for i in range(self.batch_size):
            try:
                query = self.reader.read()
            except EOFError:
                if len(q_batch) < 1:
                    raise StopIteration
                else:
                    return self.on_return(q_batch,y_batch,queries,targets)

            # X
            q_batch.append(self.get_query_embedding(query["text"]))
            #q_batch.append(self.mem_map(query["id"]))

            #Y
            top_10 = self.tfidf.get(query["wiki_id"])
            bpe_anno = self.get_bpe_annotation([word[0] for word in top_10[:5]])
            y_batch.append(bpe_anno)
            queries.append(query)
            targets.append([word[0] for word in top_10[:5]])

        return self.on_return(q_batch,y_batch,queries,targets)

    def on_return(self,q_batch,y_batch,queries,targets):
        max_seq_len = 0

        for q in q_batch:
            seq_len = len(q)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        for y in y_batch:
            seq_len = len(y)
            if seq_len > max_seq_len:
                max_seq_len = seq_len

        new_q_batch = []
        for q in q_batch:
            new_q_batch.append(self.padding(q,max_seq_len+1))

        new_y_batch = []
        for y in y_batch:
            new_y_batch.append(self.bpe_padding(y,max_seq_len+1))


        y_batch = np.stack(new_y_batch)
        q_batch = np.stack(new_q_batch)

        if self.eval:
            return q_batch, y_batch,queries,targets

        return q_batch,y_batch

    def get_query_embedding(self,query_text):
        q_emb = []
        query_text = re.sub("[/$&+,:;=?@#|'<>.\"^*()%!-]",' ', query_text)
        query_text = query_text.lower()
        for word in query_text.split(" "):

            if word != '':
                try:
                    emb = self.wordmmap(word)

                    q_emb.append(emb)
                except:
                    #print("ofv: ",word)
                    emb = self.embedder(word)[0]
                    q_emb.append(emb)

                assert emb.shape == (1024,),"got: {}".format(emb.shape)

        #q_emb = np.stack(q_emb)
        return q_emb

    def get_bpe_annotation(self,tf_idf):
        ranked_sentence = " ".join(tf_idf)
        bpes = self.embedder.laser.embed_sentences_bpe([ranked_sentence],lang=self.language)
        bpes = bpes[0].split(" ")
        annotations = []
        for word in bpes:
            w = np.zeros(self.vocab_size)
            w[self.bpe2id.get(word)] = 1
            annotations.append(w)

        return annotations

    def get_sentence_annotation(self,tf_idf):
        emb = [" ".join(tf_idf)]
        return self.embedder.embed_sentences_bpe([emb])

    def padding(self,query,max_len):
        while len(query) < max_len:
            query.append(np.zeros(1024))

        return query[:self.max_length]

    def bpe_padding(self,bpes,max_len):

        w = np.zeros(self.vocab_size)
        w[1] = 1
        bpes.append(w)

        while len(bpes) < max_len:
            w = np.zeros(self.vocab_size)
            w[0] = 1
            bpes.append(w)
        return bpes[:self.max_length]
