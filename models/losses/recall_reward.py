import numpy as np
import torch

class RecallRewardMean:
    def __init__(self):
        pass
    def __call__(self,search_results,target,base_reward,length):
        mr = []
        batch_size = len(search_results)
        rewards = []
        for i in range(batch_size):
            recall = recll_40(search_results[i],target[i]["paragraphs"])
            rewards.append(torch.Tensor(length).fill_(recall/length).cuda().double())
            mr.append(recall)

        mr = np.mean(mr)
        rewards_t = torch.stack(rewards,dim=1)
        base_reward_t = torch.Tensor(length, batch_size).fill_(base_reward).cuda().double()
        return rewards_t,base_reward_t,mr


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