import json
import matplotlib.pyplot as plt
import numpy as np
import jsonlines


base_path = "/media/jonas/archive/master/data/squad/"

r = jsonlines.open(base_path+"qas_eval.jsonl",'r')
r2 = json.load(open(base_path+"dev-v2.0.json",'r'))

data = r.read()
data = r.read()
data = r.read()
print("question:",data["question"])
cur_id = data["id"]


for wiki in r2["data"]:
    title=wiki["title"]
    for paragraphs in wiki["paragraphs"]:
        context = paragraphs["context"]
        for qa in paragraphs["qas"]:
            if not qa["is_impossible"]:
                q = qa["question"]
                q_id = qa["id"]

                answer= qa["answers"][0]["text"]
                context_answer = context+" * "+answer

                if q_id == cur_id:
                    print(q)
                    print(answer)
                    print(context)
