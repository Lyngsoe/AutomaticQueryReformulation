import numpy as np

class RecallReward:
    def __init__(self):
        pass
    def __call__(self,search_results,target,base_reward):
        rewards = []
        for i in range(len(search_results)):
            recall = recll_40(search_results[i],target[i])
            reward = recall - (base_reward[i]+0.0001)
            rewards.append(reward)
        return rewards

def get_documents_from_result(results):

    hits = results["hits"]["hits"]

    retrieved_documents = []
    for hit in hits:
        retrieved_documents.append(hit["_id"])
    return retrieved_documents

def recll_40(results,relevant_documents):


    retrieved_documents = get_documents_from_result(results)
    recall = len(set(relevant_documents).intersection(set(retrieved_documents)))/len(relevant_documents)
    return recall