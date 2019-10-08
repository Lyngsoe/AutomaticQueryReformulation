import sklearn.feature_extraction.text as skl
import json
from embedding_method.mapper import MemFetcher
from mem_map.mem_map import MemoryMapLookup
import numpy as np
from tqdm import tqdm


class WikiTfIdf:
    def __init__(self,drive_path,language,debug=False):

        self.base_path = drive_path + "raffle_wiki/{}/debug/".format(language) if debug else drive_path + "raffle_wiki/{}/".format(language)
        self.vectorizer = skl.TfidfVectorizer(max_df=0.3)

        #if not os.path.isfile(self.base_path+"paralookup.json"):
        MemoryMapLookup(self.base_path+"paragraphs.jsonl",self.base_path+"paralookup.json")

        self.wikis = json.load(open(self.base_path + "wiki.json", 'r'))
        self.memmap = MemFetcher(self.base_path+"paralookup.json",self.base_path+"paragraphs.jsonl",key="text",type="json")
        self.wiki_text = []
        self.wiki_tfidf_lookup = {}
        self.feature_array = []
        self.tf_matrix = []

        self.gather_wikis()
        self.create_tf_idf()
        self.save_top_n()

    def save_top_n(self):
        wiki2tfidf = {}
        pbar = tqdm(total=len(self.wikis.keys()),desc="finding top ranked tf-idf for each wiki")
        for wiki in self.wikis.keys():

            top_10 = self.__call__(wiki)

            if top_10 is None:
                raise Exception("top_10 is None for wiki: {}".format(wiki))

            wiki2tfidf.update({wiki: top_10})
            pbar.update()

        pbar.close()
        json.dump(wiki2tfidf, open(self.base_path + "wiki2tfidf.json", 'w'))

    def create_tf_idf(self):
        tqdm.write("starting fit transform")
        self.tf_matrix = self.vectorizer.fit_transform(self.wiki_text)
        tqdm.write("done fit transform")
        self.feature_array = np.array(self.vectorizer.get_feature_names())

    def gather_wikis(self):
        for i, (wiki_id, data) in enumerate(self.wikis.items()):
            paragraphs = data.get("paragraphs", [])
            text = ""
            for p in paragraphs:
                try:
                    t = self.memmap(p)
                except:
                    print("!!")
                    continue
                text += " " + t

            self.wiki_tfidf_lookup.update({wiki_id: i})
            self.wiki_text.append(text)
        tqdm.write("done gathering wikis")

    def __call__(self,wiki_id,n=10):
        index = self.wiki_tfidf_lookup[wiki_id]
        tfidf_sorting = np.argsort(self.tf_matrix[index].toarray()).flatten()[::-1]
        top_score = np.sort(self.tf_matrix[index].toarray()).flatten()[::-1]
        top_words = self.feature_array[tfidf_sorting][:n]
        return list(zip(top_words,top_score))

if __name__ == '__main__':
    #base_path = "/home/jonas/data/raffle_wiki/da/debug/"
    drive_path = "/media/jonas/archive/master/data/"
    language = "da"
    debug = True
    tfidf_lookup = WikiTfIdf(drive_path,language,debug)
    print(tfidf_lookup("Q741430"))

