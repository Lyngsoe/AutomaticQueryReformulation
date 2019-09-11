import jsonlines
from laserembeddings import Laser
from tqdm import tqdm
import json
import os

class VocabCreator:
    def __init__(self,drive_path,language,method="laser",debug=False):

        self.drive_path = drive_path
        self.debug = debug
        self.method = method

        self.base_path = self.drive_path + "raffle_wiki/{}/debug/".format(language) if self.debug else self.drive_path + "raffle_wiki/{}/".format(language)
        self.save_path = self.base_path + "{}/".format(self.method)
        os.makedirs(self.save_path, exist_ok=True)

        self.queries_path = self.base_path +"urlqueries.jsonl"
        self.paras_path = self.base_path +"paragraphs.jsonl"

        self.language = language
        self.batch_size = 10 if self.debug else 10
        self.embedder = Laser()
        self.tokenizer = self.embedder._get_tokenizer(self.language)
        self.words = set()
        self.lookup = {}
        self.info = json.load(open(self.base_path + "wiki_info.json", 'r'))

        ### RUN LOGIC ###
        self.get_words()
        self.number_of_words = len(self.words)
        tqdm.write("vocabulary size: {}".format(self.number_of_words))
        self.create_vocab()
        self.update_info()

    def create_vocab(self):

        writer = jsonlines.open(self.save_path + "word_emb.jsonl", "w")

        word_batch = []
        pbar = tqdm(total=int(self.number_of_words / self.batch_size), desc="embedding word batches")

        for word in self.words:
            word_batch.append(word)

            if len(word_batch) >= self.batch_size:
                emb = self.embedder.embed_sentences(word_batch, lang=self.language)

                for i, e in enumerate(emb):
                    self.lookup.update({word_batch[i]: writer._fp.tell()})
                    writer.write({"word":word_batch[i],"emb":e.tolist()})

                word_batch = []
                pbar.update()

        pbar.close()

        if len(word_batch) != 0:
            emb = self.embedder.embed_sentences(word_batch, lang=self.language)

            for i, e in enumerate(emb):
                self.lookup.update({word_batch[i]: writer._fp.tell()})
                writer.write({"word": word_batch[i], "emb": e.tolist()})

        json.dump(self.lookup, open(self.save_path + "word2emb.json", 'w'))


    def get_words(self):
        word_lst = []

        total = self.info["paragraphs"] + self.info["urlqueries"]
        pbar = tqdm(total=total,desc="tokenizing")

        for para in jsonlines.open(self.paras_path,'r'):
            text = para["text"]
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens.split()
            word_lst.extend(tokens)
            pbar.update()

        for query in jsonlines.open(self.queries_path, 'r'):
            text = query["text"]

            tokens = self.tokenizer.tokenize(text)
            tokens = tokens.split()
            word_lst.extend(tokens)
            pbar.update()

        pbar.close()
        self.words = set(word_lst)

    def update_info(self):

        self.info.update({"vocabulary_size":self.number_of_words})

        json.dump(self.info,open(self.base_path+"wiki_info.json",'w'))