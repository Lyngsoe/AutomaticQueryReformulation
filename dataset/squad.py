import json
import jsonlines
import spacy
from tqdm import tqdm
import os
import random
from embedding_method.embedders import get_embedder
import numpy as np

bert = get_embedder("bertsub", "en")

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

def create_vocab(base_path,vocab):
    method = "laser"
    save_path = base_path + "{}/".format(method)
    embedder = get_embedder(method,"en")
    os.makedirs(save_path, exist_ok=True)
    create_vocab_for_method(method,save_path,embedder,vocab)

def create_vocab_for_method(method,save_path,embedder,vocab):

    writer = jsonlines.open(save_path + "word_emb.jsonl", "w")
    lookup = {}
    word_batch = []
    batch_size = 10
    pbar = tqdm(total=int(len(vocab) / batch_size), desc="embedding word batches for {}".format(method))

    for word in vocab:
        word_batch.append(word)

        if len(word_batch) >= batch_size:
            emb = embedder(word_batch)

            for i, e in enumerate(emb):
                lookup.update({word_batch[i]: writer._fp.tell()})
                writer.write({"word": word_batch[i], "emb": e.tolist()})

            word_batch = []
            pbar.update()

    pbar.close()

    if len(word_batch) != 0:
        emb = embedder(word_batch)

        for i, e in enumerate(emb):
            lookup.update({word_batch[i]: writer._fp.tell()})
            writer.write({"word": word_batch[i], "emb": e.tolist()})

    json.dump(lookup, open(save_path + "word2emb.json", 'w'))

base_path = "/home/jonas/data/squad/"
#base_path = "/media/jonas/archive/master/data/squad/"
os.makedirs(base_path,exist_ok=True)


dataset_path = base_path+"train-v2.0.json"
qa_writer = jsonlines.open(base_path+"qas.jsonl",'w')
info = {}

qas = read_squad(dataset_path)
info.update({"qas":len(qas)})

clean_qas = []
i = 0
for qa in tqdm(qas,desc="Cleaning questions and context for train"):
    context = remove_stop_words(qa["context"])
    c,q = replace_entities_in_qa(context,qa["question"])

    tokens,emb,token_ids = bert([q])[0]
    q_tokens,q_emb,q_token_ids = bert([q])[0]
    c_tokens, c_emb, c_token_ids = bert([c])[0]

    qa.update({"question":q,"context":c,"context_tokens":c_tokens,"context_emb":[x.tolist() for x in c_emb],"context_token_ids":[int(x) for x in c_token_ids],
               "question_tokens":q_tokens,"question_emb":[x.tolist() for x in q_emb],"question_token_ids":[int(x) for x in q_token_ids]})
    clean_qas.append(qa)
    if len(clean_qas) > 100:
        break


random.shuffle(clean_qas)

for qa in tqdm(clean_qas,desc="writing qas"):
    qa_writer.write(qa)

dataset_path_eval = base_path+"dev-v2.0.json"
qa_writer_eval = jsonlines.open(base_path+"qas_eval.jsonl",'w')
qas_eval = read_squad(dataset_path_eval)

info.update({"qas":len(qas_eval)})

clean_qas = []
i = 0
for qa in tqdm(qas_eval,desc="Cleaning questions and context in eval"):
    context = remove_stop_words(qa["context"])
    c,q = replace_entities_in_qa(context,qa["question"])

    q_tokens,q_emb,q_token_ids = bert([q])[0]
    c_tokens, c_emb, c_token_ids = bert([c])[0]

    qa.update({"question": q, "context": c, "context_tokens": c_tokens, "context_emb": [x.tolist() for x in c_emb],
               "context_token_ids": [int(x) for x in c_token_ids],
               "question_tokens": q_tokens, "question_emb": [x.tolist() for x in q_emb],
               "question_token_ids": [int(x) for x in q_token_ids]})
    clean_qas.append(qa)
    if len(clean_qas) > 100:
        break
for qa in tqdm(clean_qas,desc="writing qas eval"):
    qa_writer_eval.write(qa)

json.dump(info,open(base_path+"info.json",'w'))

print("succes")


