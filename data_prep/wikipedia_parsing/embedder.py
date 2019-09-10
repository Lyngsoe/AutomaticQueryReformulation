import jsonlines
from laserembeddings import Laser
from tqdm import tqdm
import json
import os
import mmap
import numpy as np

class LaserEmbeddingCreator:
    def __init__(self,path,batch_size,paragraph_path,number_of_paras,language):
        self.method = "LASER"
        self.path = path
        self.batch_size = batch_size
        self.ready = False
        self.para2emb = None
        self.number_of_paras = number_of_paras
        self.language = language
        self.paragraph_path = paragraph_path

        if self._correct_embeddings():
            self._load()
        else:
            self.create_embeddings()

    def _correct_embeddings(self):
        if os.path.isfile(self.path+"info.json"):
            info = json.load(open(self.path+"info.json",'r'))
            return self.number_of_paras == info["number_of_paras"]
        else:
            return False

    def create_embeddings(self):
        reader = jsonlines.open(self.paragraph_path,'r')
        writer = jsonlines.open(self.path + "embeddings.jsonl","w")
        laser = Laser(embedding_options={"max_sentences": self.batch_size})
        para2emb = {}

        paras = []
        pbar = tqdm(total=int(self.number_of_paras / self.batch_size), desc="embedding paragraph batches")

        for para in reader:
            paras.append(para)

            if len(paras) >= self.batch_size:
                para_text = [p["text"] for p in paras]
                emb = laser.embed_sentences(para_text, lang=self.language)

                for i, e in enumerate(emb):
                    para2emb.update({paras[i]["id"]:writer._fp.tell()})
                    writer.write({paras[i]["id"]: emb.tolist()})

                paras = []
                pbar.update()

        pbar.close()

        if len(paras) != 0:
            para_text = [p["text"] for p in paras]
            emb = laser.embed_sentences(para_text, lang=self.language)

            for i, e in enumerate(emb):
                para2emb.update({paras[i]["id"]: writer._fp.tell()})
                writer.write({paras[i]["id"]: emb.tolist()})

        json.dump(para2emb,open(self.path + "para2emb.json",'w'))

        info = {"number_of_paras":self.number_of_paras,
                "method":self.method}

        json.dump(info,open(self.path+"info.json",'w'))

        self._load()

    def _load(self):
        self.para2emb = json.load(open(self.path + "para2emb.json",'r'))
        mfd = os.open(self.path + "embeddings.jsonl", os.O_RDONLY)
        self.mmap = mmap.mmap(mfd,0,access=mmap.PROT_READ)
        self.ready = True

    def delete_paras(self,paras):

        for _,id in paras:
            self.para2emb.pop(id)

        json.dump(self.para2emb,open(self.path + "para2emb.json", 'w'))

    def __call__(self,id):
        fp = self.para2emb[id]
        self.mmap.seek(fp)
        line = self.mmap.readline()
        return np.array(line)