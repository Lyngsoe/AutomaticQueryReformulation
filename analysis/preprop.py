import json
import matplotlib.pyplot as plt
import numpy as np
import jsonlines
from embedding_method.embedders import get_embedder
from dataset.bertify import get_tokens
import spacy


nlp = spacy.load("en_core_web_sm")

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


base_path = "/home/jonas/data/squad/"

r = jsonlines.open(base_path+"qas_eval.jsonl",'r')
r2 = json.load(open(base_path+"dev-v2.0.json",'r'))
bert = get_embedder("bertsub", "en")

#8
#12
amount = 21

i = 0
for wiki in r2["data"]:
    title=wiki["title"]
    if i < amount:
        i+=1
        continue
    for paragraphs in wiki["paragraphs"]:
        context = paragraphs["context"]
        for qa in paragraphs["qas"]:
            if not qa["is_impossible"]:
                q = qa["question"]
                q_id = qa["id"]

                answer= qa["answers"][0]["text"]
                context_answer = context+" * "+answer
                print(q)
                print(answer)
                print(context)

                context = remove_stop_words(context)
                c, q = replace_entities_in_qa(context, q)

                tokens, emb, token_ids = bert([q])[0]
                q_tokens, q_emb, q_token_ids = bert([q])[0]
                c_tokens, c_emb, c_token_ids = bert([c])[0]

                b_tokens, _, _ = bert(["The unavoidable price of reliability is simplicity."])[0]
                print(" ".join(b_tokens))
                print(len(c_tokens))
                print(q)
                print(" ".join(q_tokens))
                print(c)
                print(" ".join(c_tokens))

                raise Exception

