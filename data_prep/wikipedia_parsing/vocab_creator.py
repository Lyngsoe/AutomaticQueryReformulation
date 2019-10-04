import jsonlines
from laserembeddings import Laser
from tqdm import tqdm
import json
import os
from embedding_method.embedders import get_embedder
import re

class VocabCreator:
    def __init__(self,drive_path,language,embedding_methods,debug=False):

        self.drive_path = drive_path
        self.debug = debug
        self.methods = embedding_methods

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.queries_path = self.base_path +"urlqueries.jsonl"
        self.paras_path = self.base_path +"paragraphs.jsonl"

        self.language = language
        self.batch_size = 10 if self.debug else 10
        self.tokenizer = Laser()._get_tokenizer(self.language)
        self.words = set()
        self.lookup = {}
        self.word2id = {}
        self.paragraphs = []
        self.info = json.load(open(self.base_path + "wiki_info.json", 'r'))

        ### RUN LOGIC ###
        self.get_words()
        self.number_of_words = len(self.words)
        tqdm.write("vocabulary size: {}".format(self.number_of_words))
        self.create_vocab()
        self.create_word2id()
        self.update_info()

    def create_word2id(self):
        for i,word in enumerate(self.words):
            self.word2id.update({word:i})

        json.dump(self.word2id,open(self.base_path+"word2id.json",'w'))
    def create_vocab(self):

        for method in self.methods:
            save_path = self.base_path + "{}/".format(method)
            embedder = get_embedder(method, self.language)
            os.makedirs(save_path, exist_ok=True)
            self.create_vocab_for_method(method,save_path,embedder)

    def create_vocab_for_method(self,method,save_path,embedder):

        writer = jsonlines.open(save_path + "word_emb.jsonl", "w")

        word_batch = []
        pbar = tqdm(total=int(self.number_of_words / self.batch_size), desc="embedding word batches for {}".format(method))

        for word in self.words:
            word_batch.append(word)

            if len(word_batch) >= self.batch_size:
                emb = embedder(word_batch)

                for i, e in enumerate(emb):
                    self.lookup.update({word_batch[i]: writer._fp.tell()})
                    writer.write({"word": word_batch[i], "emb": e.tolist()})

                word_batch = []
                pbar.update()

        pbar.close()

        if len(word_batch) != 0:
            emb = embedder(word_batch)

            for i, e in enumerate(emb):
                self.lookup.update({word_batch[i]: writer._fp.tell()})
                writer.write({"word": word_batch[i], "emb": e.tolist()})

        json.dump(self.lookup, open(save_path + "word2emb.json", 'w'))

    def get_words(self):

        total = self.info["paragraphs"] + self.info["urlqueries"]
        pbar = tqdm(total=total,desc="tokenizing")

        for para in jsonlines.open(self.paras_path,'r'):
            text = para["text"]
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens.split()
            self.words.update(set(tokens))
            pbar.update()

        for query in jsonlines.open(self.queries_path, 'r'):
            text = query["text"]
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens.split()
            self.words.update(set(tokens))
            pbar.update()

        pbar.close()

    def update_info(self):

        self.info.update({"vocabulary_size":self.number_of_words})

        json.dump(self.info,open(self.base_path+"wiki_info.json",'w'))



    def _load_paragraphs(self):

        jsonlines.open(self.paras_path, 'r')