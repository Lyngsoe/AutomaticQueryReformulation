import jsonlines
from laserembeddings import Laser
from tqdm import tqdm

class VocabCreator:
    def __init__(self,save_path,paras_path,queries_path,language,batch_size):
        self.queries_path = queries_path
        self.paras_path = paras_path
        self.save_path = save_path
        self.language = language
        self.batch_size = batch_size
        self.embedder = Laser()
        self.tokenizer = self.embedder._get_tokenizer(self.language)
        self.words = set()
        self.lookup = {}
    def create_vocab(self):

        self.get_words()
        writer = jsonlines.open(self.save_path + "vocab.jsonl", "w")

        paras = []
        pbar = tqdm(total=int(self.number_of_words / self.batch_size), desc="embedding word batches")

        for word in self.words:
            paras.append(para)

            if len(paras) >= self.batch_size:
                para_text = [p["text"] for p in paras]
                emb = laser.embed_sentences(para_text, lang=self.language)

                for i, e in enumerate(emb):
                    para2emb.update({paras[i]["id"]: writer._fp.tell()})
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

        json.dump(para2emb, open(self.path + "para2emb.json", 'w'))


def get_words(self):
        reader = jsonlines.open(self.paras_path,'r')
        word_lst = []
        for para in reader.read():
            para_id = para.keys()[0]
            text = para.get(para_id)

            tokens = self.tokenizer.tokenize(text)
            word_lst.extend(tokens)

        self.words.union(word_lst)
        self.number_of_words = len(self.words)
