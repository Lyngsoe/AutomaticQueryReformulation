import numpy as np
import json

class TargetToLaserBPE:
    def __init__(self,base_path,max_length,embedder,language):
        self.base_path = base_path
        self.language = language
        self.max_length = max_length
        self.embedder = embedder
        self.bpe2id = json.load(open(self.base_path+"bpe2id.json"))
        self.id2bpe = json.load(open(self.base_path + "id2bpe.json"))
        self.vocab_size = len(self.bpe2id.keys())

    def get_embedding(self,query):
        text = query["question"]
        bpes = self.embedder.laser.embed_sentences_bpe([text],lang=self.language)
        bpes = bpes[0].split(" ")
        annotations = []
        for token in bpes:
            w = np.zeros(self.vocab_size)
            w[self.bpe2id.get(token)] = 1
            annotations.append(w)

        return annotations,text

    def padding(self, bpes, max_len):
        w = np.zeros(self.vocab_size)
        w[1] = 1
        bpes.append(w)

        while len(bpes) < max_len:
            w = np.zeros(self.vocab_size)
            w[0] = 1
            bpes.append(w)
        return bpes[:self.max_length]