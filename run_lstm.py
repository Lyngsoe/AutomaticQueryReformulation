from training.trainer import Trainer
from models.LSTM_auto_encoder_1 import LSTMAutoEncoder
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
encoder_layers = 1
decoder_layers = 4
batch_size = 64
lr = 0.01
epochs = 200


model = LSTMAutoEncoder(base_path,hidden_size=hidden_size,encoder_layers=encoder_layers,decoder_layers=decoder_layers,word_emb_size=emb_size,vocab_size=vocab_size,lr=lr,device=device)
trainer = Trainer(model=model,base_path=base_path,batch_size=batch_size,device=device,max_epoch=epochs)
trainer.train()
