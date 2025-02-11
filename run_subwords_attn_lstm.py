from training.trainer_subwords import TrainerSubwords
from models.LSTM_attention_subwords import LSTMAutoEncoder
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/squad2/"

vocab_size = 30522
emb_size = 1 # embedding dimension
hidden_size_enc = 64 # the dimension
hidden_size_dec = 256
dropout = 0.2 # the dropout value
batch_size = 16
lr = 0.0001
epochs = 250
encoder_layers = 1
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

load = False

if load:
    load_path = "/media/jonas/archive/master/data/squad/experiments/LSTM__12-16_18:42/"
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout, hidden_size_enc=hidden_size_enc,hidden_size_dec=hidden_size_dec,decoder_layers=decoder_layers,encoder_layers=encoder_layers, lr=lr)
    epoch = model.load(load_path  + "/latest/", train=False)
    trainer = TrainerSubwords(model=model, base_path=base_path, batch_size=batch_size, device=device, epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout,decoder_layers=decoder_layers,encoder_layers=encoder_layers, hidden_size_enc=hidden_size_enc,hidden_size_dec=hidden_size_dec, lr=lr)
    trainer = TrainerSubwords(model=model, base_path=base_path, batch_size=batch_size, max_epoch=epochs, device=device,specs=specs)


trainer.train()