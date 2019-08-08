import json
from car import CAR
from tqdm import tqdm
import numpy as np

class Evaluator:
    def __init__(self,result_path,dataset):
        self.result_path = result_path
        self.dataset = dataset
        self.results = json.load(open(result_path,'r'))

        self.para2wiki, self.wiki2para = self.dataset.load_annotation()

        self.eval()

    def eval(self):
        recall3 = []
        recall5 = []
        recall10 = []
        recall20 = []
        recall30 = []
        recall40 = []

        for result in tqdm(self.results, desc="calculating results"):
            search_results = result.get("results")
            retrieved_doc_id = []
            for i, sr in enumerate(search_results):
                retrieved_doc_id.append(sr.get("_source").get("para_id"))

            relevant_docs_id = self.wiki2para.get(result["wiki_id"])
            if len(relevant_docs_id) >= 3:
                recall3.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 3))
            if len(relevant_docs_id) >= 5:
                recall5.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 5))
            if len(relevant_docs_id) >= 10:
                recall10.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 10))
            if len(relevant_docs_id) >= 20:
                recall20.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 20))
            if len(relevant_docs_id) >= 30:
                recall30.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 30))
            if len(relevant_docs_id) >= 40:
                recall40.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 40))

        print("\n\nrecall 3:", np.mean(recall3))
        print("recall 5:", np.mean(recall5))
        print("recall 10:", np.mean(recall10))
        print("recall 20:", np.mean(recall20))
        print("recall 30:", np.mean(recall30))
        print("recall 40:", np.mean(recall40))


    @staticmethod
    def calc_recall(relevant_docs,retrieved_docs,rank):
        retrieved_docs = retrieved_docs[0:rank]
        retrieved_docs = set(retrieved_docs)
        if len(relevant_docs) < rank:
            raise Exception("recall rank larger than number of relevant docs")

        intersec = retrieved_docs.intersection(relevant_docs)
        return len(intersec) / rank




if __name__ == "__main__":


    Evaluator("/results/identity_model/",CAR(debug=True))