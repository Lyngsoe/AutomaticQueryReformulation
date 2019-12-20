import jsonlines
from tqdm import tqdm
import json
import numpy as np


base_path = "/media/jonas/archive/master/data/rl_squad/"

info = json.load(open(base_path+"info.json"))


qas = []
pbar = tqdm(total=info["qas"],desc="calc mean reward train")
for q in jsonlines.open(base_path+"qas.jsonl",'r'):
    qas.append(q["base_reward"])
    pbar.update()
pbar.close()

print("mean reward train",np.mean(qas))

qas = []
pbar = tqdm(total=info["qas"],desc="calc mean reward test")
for q in jsonlines.open(base_path+"qas_eval.jsonl",'r'):
    qas.append(q["base_reward"])
    pbar.update()
pbar.close()

print("mean reward train",np.mean(qas))