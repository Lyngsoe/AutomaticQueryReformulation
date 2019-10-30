import os
import jsonlines
import random
import json
from tqdm import tqdm

def write_batch(writer,qas):
    for qa in qas:
        writer.write(qa)

def read_and_shuffle(reader,writer,max_count):
    qas = []
    pbar = tqdm(total=max_count,desc="reading qas")
    for qa in reader:
        qas.append(qa)
        pbar.update()
        if len(qas) > 86821/2+1:
            random.shuffle(qas)
            write_batch(writer,qas)
            qas = []

    random.shuffle(qas)
    write_batch(writer, qas)
    pbar.close()




#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/squad/"
os.makedirs(base_path,exist_ok=True)


info = json.load(open(base_path+"info.json"))

qa_reader = jsonlines.open(base_path+"qas.jsonl",'r')
qa_writer = jsonlines.open(base_path+"qas2.jsonl",'w',flush=True)

read_and_shuffle(qa_reader,qa_writer,info["qas"])

qa_reader = jsonlines.open(base_path+"qas_eval.jsonl",'r')
qa_writer = jsonlines.open(base_path+"qas_eval2.jsonl",'w',flush=True)

read_and_shuffle(qa_reader,qa_writer,info["qas_eval"])

