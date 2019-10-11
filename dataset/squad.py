import json
import jsonlines
import spacy
from tqdm import tqdm
from laserembeddings import Laser
from embedding_method.embedders import get_embedder
import os

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
        if len(qas_to_write) > 10:
            break
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
dataset_path = base_path+"train-v2.0.json"
qa_writer = jsonlines.open(base_path+"qas.jsonl",'w')
info = {}

qas = read_squad(dataset_path)
info.update({"qas":len(qas)})

vocab = set()
tokenizer = Laser()._get_tokenizer("en")

clean_qas = []
for qa in tqdm(qas,desc="Cleaning questions and context and gathering vocab"):
    context = remove_stop_words(qa["context"])
    c,q = replace_entities_in_qa(context,qa["question"])

    tokens = tokenizer.tokenize(c+" "+q)
    tokens = tokens.split()
    vocab.update(set(tokens))

    qa.update({"question":q,"context":c})
    qa_writer.write(qa)

json.dump(info,open(base_path+"info.json",'w'))

create_vocab(base_path,vocab)

print("succes")


