import numpy as np

class RankReward:
    def __init__(self,base_line):
        self.base_line = base_line
        self.name = "RankReward"
    def __call__(self,search_results,target):
        batch_size = len(search_results)
        rewards = []
        for i in range(batch_size):
            prec = precision(search_results[i],target[i]["c_id"])
            rewards.append(prec)

        base_reward,normalized_reward = self.base_line(rewards,[q["q_id"] for q in target])
        return rewards,base_reward,normalized_reward

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