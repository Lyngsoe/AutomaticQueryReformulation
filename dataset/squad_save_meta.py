from tqdm import tqdm
import jsonlines
import json

base_path = "/media/jonas/archive/master/data/squad/"
qa_reader = jsonlines.open(base_path+"qas.jsonl",'r')
info = json.load(open(base_path+"info.json"))
ctl = []
qtl = []
cwl = []

pbar = tqdm(total=info["qas"],desc="reading qas")
for qa in qa_reader:
    context_token_length = len(qa["context_token_ids"])
    question_token_length = len(qa["question_token_ids"])
    ctl.append(context_token_length)
    qtl.append(question_token_length)
    cwl.append(len(qa["context"].split(" ")))
    pbar.update()

pbar.close()

json.dump(ctl,open(base_path+"context_token_length.json",'w'))
json.dump(qtl,open(base_path+"question_token_length.json",'w'))
json.dump(cwl,open(base_path+"context_word_length.json",'w'))