import json
import jsonlines
from tqdm import tqdm
import os
import random
from embedding_method.embedders import get_embedder
import hashlib
import copy



def read_squad(path):
    data = json.load(open(path, 'r'))
    qas_to_write = []
    paragraphs = []
    for wiki in data["data"]:
        title=wiki["title"]
        para_ids = [hashlib.sha3_256(para["context"].encode()).hexdigest() for para in wiki["paragraphs"]]

        for para in wiki["paragraphs"]:
            context = para["context"]
            c_id = hashlib.sha3_256(context.encode()).hexdigest()
            paragraphs.append({"paragraph": context, "c_id": c_id})
            for qa in para["qas"]:
                if not qa["is_impossible"]:
                    q = qa["question"]
                    q_id = qa["id"]

                    qas_to_write.append({"question":q,"c_id":c_id,"q_id":q_id,"title":title,"paragraphs":para_ids})


    return qas_to_write,paragraphs


def write_batch(writer,qas):
    for qa in qas:
        writer.write(qa)


#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"
os.makedirs(base_path,exist_ok=True)

dataset_path = base_path+"train-v2.0.json"
qa_writer = jsonlines.open(base_path+"qas.jsonl",'w',flush=True)
info = {}

qas,paragraphs = read_squad(dataset_path)
info.update({"qas":len(qas)})

i = 0
new_qas = []

bert = get_embedder("bertsub", "en")

for qa in tqdm(qas,desc="embedding questions"):

    new_qa = copy.deepcopy(qa)

    q_tokens,q_emb,q_token_ids = bert([new_qa["question"]])[0]

    new_qa.update({"question_tokens":q_tokens,"question_emb":[x.tolist() for x in q_emb],"question_token_ids":[int(x) for x in q_token_ids]})
    new_qas.append(new_qa)

    #if len(new_qas) > 10000:
        #random.shuffle(new_qas)
        #write_batch(qa_writer,new_qas)
        #new_qas = []

random.shuffle(new_qas)
write_batch(qa_writer,new_qas)
new_qas = []

dataset_path_eval = base_path+"dev-v2.0.json"
qa_writer_eval = jsonlines.open(base_path+"qas_eval.jsonl",'w',flush=True)
qas_eval,para_eval = read_squad(dataset_path_eval)
paragraphs.extend(para_eval)

para_writer = jsonlines.open(base_path+"paragraphs.jsonl",'w',flush=True)
write_batch(para_writer,paragraphs)
paragraphs = []

info.update({"qas_eval":len(qas_eval),"paragraphs":len(paragraphs)})

i = 0
for qa in tqdm(qas_eval,desc="embedding questions in eval"):
    new_qa = copy.deepcopy(qa)

    q_tokens, q_emb, q_token_ids = bert([new_qa["question"]])[0]

    new_qa.update({"question_tokens": q_tokens, "question_emb": [x.tolist() for x in q_emb],
                   "question_token_ids": [int(x) for x in q_token_ids]})
    new_qas.append(new_qa)

    # if len(new_qas) > 10000:
    # random.shuffle(new_qas)
    # write_batch(qa_writer,new_qas)
    # new_qas = []

random.shuffle(new_qas)
write_batch(qa_writer_eval,new_qas)
new_qas = []

json.dump(info,open(base_path+"info.json",'w'))

print("succes")


