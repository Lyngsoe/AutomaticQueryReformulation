from tqdm import tqdm
import json
import numpy as np
import jsonlines



def eval(results):
    ranks = [3, 5, 10, 20, 30, 40]
    recall = [[] for i in range(len(ranks))]
    precision = [[] for i in range(len(ranks))]
    map = [[] for i in range(len(ranks))]


    mrr = []
    mean_rank = []


    for targets,retrieved_doc_id in results:

        relevant_docs_id = targets["paragraphs"]
        target_doc_id = targets["c_id"]

        mrr.append(calc_mrr(relevant_docs_id, retrieved_doc_id))
        mean_rank.append(calc_mrr([target_doc_id],retrieved_doc_id))

        for i,r in enumerate(ranks):

            recall[i].append(calc_recall(relevant_docs_id, retrieved_doc_id, r))
            precision[i].append(calc_precision(relevant_docs_id, retrieved_doc_id, r))
            map[i].append(calc_avp(relevant_docs_id, retrieved_doc_id, r))


    for i,r in enumerate(ranks):


        print("R@{}: {}".format(r, np.mean(recall[i])))

        print("P@{}: {}".format(r, np.mean(precision[i])))
        print("MAP@{}: {}".format(r,np.mean(map[i])))

    print("\n")
    print("MRR:",np.mean(mrr))
    print("mean_rank:",np.mean(mean_rank))


def calc_recall(relevant_docs,retrieved_docs,rank):
    retrieved_docs = retrieved_docs[:rank]
    retrieved_docs = set(retrieved_docs)
    intersec = retrieved_docs.intersection(relevant_docs)
    return len(intersec) / len(relevant_docs)


def calc_precision(relevant_docs,retrieved_docs,rank):
    retrieved_docs = retrieved_docs[:rank]
    retrieved_docs = set(retrieved_docs)
    intersec = retrieved_docs.intersection(relevant_docs)
    return len(intersec) / rank

def calc_mrr(relevant_docs, retrieved_docs):
    for i,doc in enumerate(retrieved_docs):
        if doc in relevant_docs:
            return 1/(i+1)
    return 0

def calc_avp(relevant_docs, retrieved_docs,rank):
    retrieved_docs = retrieved_docs[:rank]
    pres = calc_precision(relevant_docs,retrieved_docs,rank)
    return pres*len(set(retrieved_docs).intersection(relevant_docs))/len(relevant_docs)
