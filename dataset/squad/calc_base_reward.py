from search_engine.elastic_search import ELSearch
import jsonlines
from models.losses.recall_reward import recll_40
from tqdm import tqdm


def write_batch(writer, qas):
    for qa in qas:
        writer.write(qa)

base_path = "/media/jonas/archive/master/data/rl_squad/"


search_engine = ELSearch("squad")

qas = []
for q in jsonlines.open(base_path+"qas.jsonl",'r'):
    qas.append(q)


for q in tqdm(qas,desc="base reward"):
    results = search_engine.search([q["question"]])
    recall = recll_40(results[0],q["paragraphs"])
    q.update({"base_reward":recall})


qa_writer = jsonlines.open(base_path+"qas.jsonl",'w',flush=True)
write_batch(qa_writer,qas)


qas = []
for q in jsonlines.open(base_path+"qas_eval.jsonl",'r'):
    qas.append(q)


for q in tqdm(qas,desc="base reward eval"):
    results = search_engine.search([q["question"]])
    recall = recll_40(results[0],q["paragraphs"])
    q.update({"base_reward":recall})


qa_writer = jsonlines.open(base_path+"qas_eval.jsonl",'w',flush=True)
write_batch(qa_writer,qas)