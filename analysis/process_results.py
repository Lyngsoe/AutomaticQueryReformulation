from tqdm import tqdm
import json
import numpy as np
import jsonlines


class ResultProcessor:
    def __init__(self,result_path,base_path,fold=4):
        self.result_path = result_path
        self.base_path = base_path

        self.results_amount = json.load(open(self.base_path+"wiki_info.json",'r'))["fold-{}".format(fold)]["queries"]
        self.wikis = json.load(open(self.base_path+"wiki.json",'r'))

        self.eval()

    def eval(self):
        recall3 = []
        recall5 = []
        recall10 = []
        recall20 = []
        recall30 = []
        recall40 = []

        precision3 = []
        precision5 = []
        precision10 = []
        precision20 = []
        precision30 = []
        precision40 = []

        mrr = []
        map10 = []

        pbar = tqdm(total=self.results_amount, desc="calculating results")
        for result in jsonlines.open(self.result_path,'r'):
            try:
                search_results = result.get("result").get("hits").get("hits")
                relevant_docs_id = self.wikis.get(result.get("query").get("wiki_id"))["paragraphs"]
                retrieved_doc_id = []
                for i, sr in enumerate(search_results):
                    retrieved_doc_id.append(sr.get("_id"))

                mrr.append(self.calc_mrr(relevant_docs_id,retrieved_doc_id))

                if len(relevant_docs_id) >= 3:
                    recall3.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 3))
                    precision3.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 3))
                if len(relevant_docs_id) >= 5:
                    recall5.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 5))
                    precision5.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 5))
                if len(relevant_docs_id) >= 10:
                    recall10.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 10))
                    precision10.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 10))
                    map10.append(self.calc_avp(relevant_docs_id,retrieved_doc_id,10))
                if len(relevant_docs_id) >= 20:
                    recall20.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 20))
                    precision20.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 20))
                if len(relevant_docs_id) >= 30:
                    recall30.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 30))
                    precision30.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 30))
                if len(relevant_docs_id) >= 40:
                    recall40.append(self.calc_recall(relevant_docs_id, retrieved_doc_id, 40))
                    precision40.append(self.calc_precision(relevant_docs_id, retrieved_doc_id, 40))
                pbar.update()
            except:
                print(result)
                print(result.get("query").get("wiki_id"))
                print(self.wikis.get(result.get("query").get("wiki_id")))
                raise Exception("")

        pbar.close()

        print("\n\nR@3:", np.mean(recall3))
        print("R@5:", np.mean(recall5))
        print("R@10:", np.mean(recall10))
        print("R@20:", np.mean(recall20))
        print("R@30:", np.mean(recall30))
        print("R@40:", np.mean(recall40))

        print("\n\nP@3:", np.mean(precision3))
        print("P@5:", np.mean(precision5))
        print("P@10:", np.mean(precision10))
        print("P@20:", np.mean(precision20))
        print("P@30:", np.mean(precision30))
        print("P@40:", np.mean(precision40))

        print("\n")
        print("MRR:",np.mean(mrr))
        print("MAP@10:", np.mean(map10))

    @staticmethod
    def calc_recall(relevant_docs,retrieved_docs,rank):
        retrieved_docs = retrieved_docs[0:rank]
        retrieved_docs = set(retrieved_docs)
        if len(relevant_docs) < rank:
            raise Exception("recall rank larger than number of relevant docs")

        intersec = retrieved_docs.intersection(relevant_docs)
        return len(intersec) / len(relevant_docs)

    @staticmethod
    def calc_precision(relevant_docs,retrieved_docs,rank):
        retrieved_docs = retrieved_docs[0:rank]
        retrieved_docs = set(retrieved_docs)
        if len(relevant_docs) < rank:
            raise Exception("recall rank larger than number of relevant docs")

        intersec = retrieved_docs.intersection(relevant_docs)
        return len(intersec) / rank

    @staticmethod
    def calc_mrr(relevant_docs, retrieved_docs):
        for i,doc in enumerate(retrieved_docs):
            if doc in relevant_docs:
                return 1/(i+1)
        return 0

    def calc_avp(self,relevant_docs, retrieved_docs,rank):
        retrieved_docs = retrieved_docs[0:rank]
        pres = self.calc_precision(relevant_docs,retrieved_docs,rank)
        return pres*len(set(retrieved_docs).intersection(relevant_docs))/len(relevant_docs)


if __name__ == '__main__':
    result_path="/home/jonas/Documents/master/eval_results.jsonl"
    base_path="/media/jonas/archive/master/data/raffle_wiki/da/"

    ResultProcessor(result_path=result_path,base_path=base_path)