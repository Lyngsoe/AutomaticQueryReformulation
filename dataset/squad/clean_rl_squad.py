import jsonlines
from tqdm import tqdm


def write_batch(writer, qas):
    for qa in qas:
        writer.write(qa)

base_path = "/media/jonas/archive/master/data/rl_squad/"


qas = []
for q in jsonlines.open(base_path+"qas.jsonl",'r'):
    if len(q["question"]) > 2:
        qas.append(q)

qa_writer = jsonlines.open(base_path+"qas.jsonl",'w',flush=True)
write_batch(qa_writer,qas)


qas = []
for q in jsonlines.open(base_path+"qas_eval.jsonl",'r'):
    if len(q["question"]) > 2:
        qas.append(q)

qa_writer = jsonlines.open(base_path+"qas_eval.jsonl",'w',flush=True)
write_batch(qa_writer,qas)