import json
import jsonlines
import spacy
from tqdm import tqdm
import os
import random
from embedding_method.embedders import get_embedder
import numpy as np

bert = get_embedder("berttoken", "en")

nlp = spacy.load("en_core_web_sm")


def read_squad(path):
    data = json.load(open(path, 'r'))
    qas_to_write = []
    for wiki in data["data"]:
        title=wiki["title"]
        for paragraphs in wiki["paragraphs"]:
            context = paragraphs["context"]
            for qa in paragraphs["qas"]:
                if not qa["is_impossible"]:
                    q = qa["question"]
                    q_id = qa["id"]
                    answer= qa["answers"][0]["text"]
                    context_answer = context+" * "+answer
                    qas_to_write.append({"question":q.lower(),"context":context_answer.lower(),"id":q_id,"title":title})

    return qas_to_write



def remove_stop_words(context):
    doc = nlp(context)
    tokens = [token.text for token in doc if not token.is_stop]
    return " ".join(tokens)

def replace_entities_in_text(text,cur_entities):
    doc = nlp(text)
    new_text = ""
    current_pos = 0
    for ent in doc.ents:
        entities = cur_entities.get(ent.label_, {})
        number = entities.get(ent.text)
        if number is None:
            number = str(len(entities.keys()))
            label = ent.label_
            entities.update({ent.text: number})
            cur_entities.update({ent.label_: entities})
        else:
            label = ent.label_
            number = str(number)
        new_text += text[current_pos:ent.start_char]
        new_text += " " + label + " " + number + " "
        current_pos = ent.end_char

    new_text += text[current_pos:]

    return new_text,cur_entities

def replace_entities_in_qa(context,question):
    cur_entities = {}
    context,cur_entities = replace_entities_in_text(context,cur_entities)
    question, cur_entities = replace_entities_in_text(question, cur_entities)
    return context,question


def write_batch(writer,qas):
    for qa in qas:
        writer.write(qa)


#base_path = "/home/jonas/data/squad2/"
base_path = "/media/jonas/archive/master/data/squad2/"
os.makedirs(base_path,exist_ok=True)


dataset_path = base_path+"train-v2.0.json"
qa_writer = jsonlines.open(base_path+"qas.jsonl",'w',flush=True)
info = {}

qas = read_squad(dataset_path)
info.update({"qas":len(qas)})

i = 0
new_qas = []
for qa in tqdm(qas,desc="Cleaning questions and context for train"):
    context = remove_stop_words(qa["context"])
    c,q = replace_entities_in_qa(context,qa["question"])

    q_token_ids = bert([q])
    c_token_ids = bert([c])

    new_qa = {"question":q,"context":c,"context_token_ids":[int(x) for x in c_token_ids],
               "question_token_ids":[int(x) for x in q_token_ids],"id":qa["id"],"title":qa["title"]}
    new_qas.append(new_qa)

    if len(new_qas) > 10000:
        random.shuffle(new_qas)
        write_batch(qa_writer,new_qas)
        new_qas = []

random.shuffle(new_qas)
write_batch(qa_writer,new_qas)
new_qas = []

dataset_path_eval = base_path+"dev-v2.0.json"
qa_writer_eval = jsonlines.open(base_path+"qas_eval.jsonl",'w',flush=True)
qas_eval = read_squad(dataset_path_eval)

info.update({"qas_eval":len(qas_eval)})

i = 0
for qa in tqdm(qas_eval,desc="Cleaning questions and context in eval"):
    context = remove_stop_words(qa["context"])
    c,q = replace_entities_in_qa(context,qa["question"])

    q_token_ids = bert([q])
    c_token_ids = bert([c])

    new_qa = {"question": q, "context": c, "context_token_ids": [int(x) for x in c_token_ids],
              "question_token_ids": [int(x) for x in q_token_ids], "id": qa["id"], "title": qa["title"]}
    new_qas.append(new_qa)

    if len(new_qas) > 10000:
        random.shuffle(new_qas)
        write_batch(qa_writer_eval, new_qas)
        new_qas = []

random.shuffle(new_qas)
write_batch(qa_writer_eval,new_qas)
new_qas = []

json.dump(info,open(base_path+"info.json",'w'))

print("succes")


