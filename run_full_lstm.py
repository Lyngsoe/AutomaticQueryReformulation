from training.trainer import Trainer
from models.LSTM_full import LSTMAutoEncoder
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/squad/"
#base_path = "/scratch/s134280/squad/"

vocab_size = 30522
emb_size = 768 # embedding dimension
hidden_size_enc = 768 # the dimension
hidden_size_dec = 768
dropout = 0.2 # the dropout value
batch_size = 16
lr = 0.001
epochs = 250
encoder_layers = 5
decoder_layers = 1

specs = {
    "vocab_size":vocab_size,
    "emb_size":emb_size,
    "hidden_size_enc": hidden_size_enc,
    "hidden_size_dec": hidden_size_dec,
    "dropout": dropout,
    "lr": lr,
    "epochs": epochs,
    "decoder_layers":decoder_layers,
    "encoder_layers":encoder_layers
}

embeddings = np.load("/home/jonas/Documents/master/embedding_method/bert_embeddings.npy")
embeddings = torch.tensor(embeddings,device=device).double()

load = False

if load:
    load_path = "/media/jonas/archive/master/data/squad/experiments/LSTM__12-16_18:42/"
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size,embeddings=embeddings, vocab_size=vocab_size, device=device,dropout=dropout, hidden_size_enc=hidden_size_enc,hidden_size_dec=hidden_size_dec,decoder_layers=decoder_layers,encoder_layers=encoder_layers, lr=lr)
    epoch = model.load(load_path  + "/latest/", train=False)
    trainer = Trainer(model=model, base_path=base_path, batch_size=batch_size, device=device, epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size,embeddings=embeddings, vocab_size=vocab_size, device=device,dropout=dropout,decoder_layers=decoder_layers,encoder_layers=encoder_layers, hidden_size_enc=hidden_size_enc,hidden_size_dec=hidden_size_dec, lr=lr)
    trainer = Trainer(model=model, base_path=base_path, batch_size=batch_size, max_epoch=epochs, device=device,specs=specs)


trainer.train()
