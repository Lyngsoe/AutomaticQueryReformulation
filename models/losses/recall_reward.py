import numpy as np
import torch

class RecallRewardMean:
    def __init__(self):
        pass
    def __call__(self,search_results,target,base_reward,length):
        mr = []
        mean_base = np.mean(base_reward)
        batch_size = len(search_results)
        for i in range(batch_size):
            recall = recll_40(search_results[i],target[i])
            reward = recall
            mr.append(reward)

        mr = np.mean(mr)
        rewards = torch.Tensor(length, batch_size).fill_(mr).cuda().double()
        base_reward = torch.Tensor(length, batch_size).fill_(np.mean(base_reward)).cuda().double()
        return rewards,base_reward,mr - mean_base


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