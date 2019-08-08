from lupyne import engine
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm

class LUCENE:
    def __init__(self,index_dir="index",debug_index_dir="index_debug",debug=False,ram=False):
        self.index_dir = debug_index_dir if debug else index_dir
        self.indexer = engine.Indexer() if ram else engine.Indexer(directory=self.index_dir)

    def index(self,paras):
        for txt, id in tqdm(paras, desc="indexing paragraphs"):
            self.indexer.set("text", engine.Field.Text, stored=True)
            self.indexer.set("id", engine.Field.Text, stored=True)
            self.indexer.add(text=txt,id=id)
        self.indexer.commit()

    def search(self,query):
        #print(query)
        query = re.escape(query)
        query = query.replace("\\","")
        query = query.replace("/"," ")
        query = re.sub(r'[^\w]', ' ', query)
        #print(query)

        #print(query)
        # Now search the index:
        hits = self.indexer.search(query, field='text')  # parsing handled if necessary

        results = []
        for rank,hit in enumerate(hits):

            txt = hit["text"]
            id = hit["id"]
            results.append({"txt":txt,"para_id":id,"rank":rank+1})
        return results
