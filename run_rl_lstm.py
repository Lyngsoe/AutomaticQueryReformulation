from training.rl_trainer import Trainer
from models.lstm_simple import LSTMSimple
from models.losses.recall_reward import RecallReward
from search_engine.elastic_search import ELSearch
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"

base_path = "/scratch/s134280/squad/"
#base_path = "/home/jonas/data/squad/"
#base_path = "/media/jonas/archive/master/data/squad/"


vocab_size = 30522
emb_size = 768 # embedding dimension
hidden_size = 1024
layers = 1
batch_size = 1
lr = 0.01
epochs = 200

reward_function = RecallReward()
search_engine = ELSearch("en")

model = LSTMSimple(base_path,reward_function,layers=layers,hidden_size=hidden_size,word_emb_size=emb_size,vocab_size=vocab_size,lr=lr,device=device)
trainer = Trainer(model=model,base_path=base_path,search_engine=search_engine,batch_size=batch_size,device=device,max_epoch=epochs)
trainer.train()
