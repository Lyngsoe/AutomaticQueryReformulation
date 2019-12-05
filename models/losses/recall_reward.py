import numpy as np
import torch

class RecallRewardMean:
    def __init__(self):
        pass
    def __call__(self,search_results,target,base_reward,length):
        rewards = []
        mr = []
        batch_size = len(search_results)
        for i in range(batch_size):
            recall = recll_40(search_results[i],target[i])
            reward = recall - (base_reward[i]+0.0001)
            mr.append(reward)
            rewards.append(torch.Tensor(length, 1).fill_(reward / batch_size))

        rewards = torch.stack(rewards, dim=1).cuda().double()
        return rewards,np.mean(mr)


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