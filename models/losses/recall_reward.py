import numpy as np

class RecallRewardMean:
    def __init__(self,base_line):
        self.base_line = base_line
        self.name = "RecallRewardMean"
    def __call__(self,search_results,target):
        batch_size = len(search_results)
        rewards = []
        for i in range(batch_size):
            recall = recll_40(search_results[i],target[i]["paragraphs"])
            rewards.append(recall)

        base_reward,normalized_reward = self.base_line(rewards)
        return np.mean(rewards),base_reward,normalized_reward


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