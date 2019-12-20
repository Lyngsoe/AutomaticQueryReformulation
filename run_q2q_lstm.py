from training.trainer_q2q import TrainerQ2Q
from models.LSTM import LSTMAutoEncoder
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
print(device)
#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"


vocab_size = 30522
emb_size = 768 # embedding dimension
hidden_size = 512 # the dimension
dropout = 0.2 # the dropout value
batch_size = 8
lr = 0.00001
epochs = 250
l2 = 0
encoder_layers = 2
decoder_layers = 2


specs = {
    "vocab_size":vocab_size,
    "emb_size":emb_size,
    "hidden_size": hidden_size,
    "dropout": dropout,
    "lr": lr,
    "l2":l2,
    "epochs": epochs,
    "decoder_layers":decoder_layers,
    "encoder_layers":encoder_layers
}


load = False

if load:
    load_path = "/media/jonas/archive/master/data/rl_squad/experiments/LSTM__12-16_14:45"
    exp_name = "LSTM__12-16_14:45"
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout, hidden_size=hidden_size,decoder_layers=decoder_layers,encoder_layers=encoder_layers, lr=lr,l2=l2)
    epoch = model.load(load_path  + "/latest/", train=True)
    trainer = TrainerQ2Q(model=model, base_path=base_path, batch_size=batch_size, device=device, epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = LSTMAutoEncoder(base_path, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout,decoder_layers=decoder_layers,encoder_layers=encoder_layers, hidden_size=hidden_size, lr=lr,l2=l2)
    trainer = TrainerQ2Q(model=model, base_path=base_path, batch_size=batch_size, max_epoch=epochs, device=device,specs=specs)


trainer.train()
