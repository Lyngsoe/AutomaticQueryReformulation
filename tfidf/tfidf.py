import sklearn.feature_extraction.text as skl
import json
from embedding_method.mapper import MemFetcher
from mem_map.mem_map import MemoryMapLookup
import numpy as np
import os

class WikiTfIdf:
    def __init__(self,base_path):
        self.base_path = base_path
        self.vectorizer = skl.TfidfVectorizer()

        if not os.path.isfile(self.base_path+"paralookup.json"):
            MemoryMapLookup(self.base_path+"paragraphs.jsonl",self.base_path+"paralookup.json")

        self.wikis = json.load(open(base_path + "wiki.json", 'r'))
        self.memmap = MemFetcher(self.base_path+"paralookup.json",self.base_path+"paragraphs.jsonl",key="text",type="json")
        self.wiki_text = []
        self.wiki_tfidf_lookup = {}
        self.feature_array = []
        self.tf_matrix = []

        self.gather_wikis()
        self.create_tf_idf()

    def create_tf_idf(self):
        self.tf_matrix = self.vectorizer.fit_transform(self.wiki_text)
        self.feature_array = np.array(self.vectorizer.get_feature_names())

    def gather_wikis(self):
        for i, (wiki_id, data) in enumerate(self.wikis.items()):
            paragraphs = data.get("paragraphs", [])
            text = ""
            for p in paragraphs:
                try:
                    t = self.memmap(p)
                    text += " " + t
                except KeyError as e:
                    print(e)
                    continue

            self.wiki_tfidf_lookup.update({wiki_id: i})
            self.wiki_text.append(text)

    def __call__(self,wiki_id,n=10):
        index = self.wiki_tfidf_lookup[wiki_id]
        tfidf_sorting = np.argsort(self.tf_matrix[index].toarray()).flatten()[::-1]
        top_n = self.feature_array[tfidf_sorting][:n]
        return top_n

if __name__ == '__main__':
    base_path = "/home/jonas/data/raffle_wiki/da/debug/"

    tfidf_lookup = WikiTfIdf(base_path)
    print(tfidf_lookup("Q357984"))

