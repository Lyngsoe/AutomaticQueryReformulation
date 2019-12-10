import numpy as np
import torch

class RankReward:
    def __init__(self):
        pass
    def __call__(self,search_results,target,base_reward,length):
        mr = []
        batch_size = len(search_results)
        for i in range(batch_size):
            prec = precision(search_results[i],target[i]["c_id"])
            reward = prec
            mr.append(reward)


        mr = np.sum(mr) / len(search_results)
        rewards_t = torch.Tensor(length, batch_size).fill_(mr).cuda().double()
        base_reward_t = torch.Tensor(length, batch_size).fill_(base_reward).cuda().double()
        return rewards_t,base_reward_t,mr

def get_corrent_document_rank(results,doc_id):

    hits = results["hits"]["hits"]
    pos = None
    for i,hit in enumerate(hits):
        if hit["_id"] == doc_id:
            pos = i
    return pos

def precision(results,doc_id):


    pos = get_corrent_document_rank(results,doc_id)

    if pos is None:
        return 0
    else:
        return 1/(pos+1)