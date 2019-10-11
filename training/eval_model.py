from models.LSTM_auto_encoder_1 import LSTMAutoEncoder
from search_engine.elastic_search import ELSearch
import time
from training.dataloader_bpe import DataloaderBPE
from embedding_method.embedders import get_embedder
import torch
import jsonlines
import os
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"



drive_path = "/home/jonas/data/"
#drive_path = "/media/jonas/archive/master/data/"
debug = True
language = "da"
embedding_method = "laser"
oov_embedder = get_embedder(embedding_method,language)

model = LSTMAutoEncoder(drive_path=drive_path,language=language,device=device,debug=debug)
load_path = "/home/jonas/data/raffle_wiki/da/debug/experiments/LSTM_auto_encoder__10-10_14:19/"
model.load(load_path,train=False)

print(model.exp_name)
print("starting evaluation")
#search_engine = ELSearch(index=language)

eval_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder, embedding_method="laser", language="da",batch_size=1,debug=debug,fold=0,eval=True)
i_eval = 0
for eval_x, eval_y,queries,targets in iter(eval_data):
    eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
    eval_y = torch.tensor(eval_y, device=device).type(torch.float64)
    sentence,loss = model.predict(eval_x,eval_y)
    if i_eval < 3:
        print("test_sentence:", sentence[0]," loss:",loss)
    i_eval+=1
