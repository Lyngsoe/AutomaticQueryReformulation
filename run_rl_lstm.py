from training.rl_trainer import Trainer
from models.rl_LSTM import LSTMAutoEncoder
from models.losses.recall_reward import RecallRewardMean
from search_engine.elastic_search import ELSearch
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"

#base_path = "/scratch/s134280/squad/"
#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"


vocab_size = 30522
emb_size = 768 # embedding dimension
hidden_size = 768 # the dimension
dropout = 0.8 # the dropout value
batch_size = 64
lr = 0.000001
epochs = 250
encoder_layers = 2
decoder_layers = 2
l2 = 0.9

reward_function = RecallRewardMean()
search_engine = ELSearch("squad")

specs = {
    "vocab_size":vocab_size,
    "emb_size":emb_size,
    "hidden_size": hidden_size,
    "dropout": dropout,
    "lr": lr,
    "epochs": epochs,
    "decoder_layers":decoder_layers,
    "encoder_layers":encoder_layers,
    "l2":l2
}

load = False

if load:
    load_path = "/media/jonas/archive/master/data/squad/experiments/LSTM__12-09_10:59"
    model = LSTMAutoEncoder(base_path,reward_function, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout, hidden_size=hidden_size,decoder_layers=decoder_layers,encoder_layers=encoder_layers, lr=lr,l2=l2)
    epoch = model.load(load_path  + "/latest/", train=False)
    trainer = Trainer(model=model,search_engine=search_engine,base_path=base_path,batch_size=batch_size,device=device,epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = LSTMAutoEncoder(base_path,reward_function, word_emb_size=emb_size, vocab_size=vocab_size, device=device,dropout=dropout,decoder_layers=decoder_layers,encoder_layers=encoder_layers, hidden_size=hidden_size, lr=lr,l2=l2)
    trainer = Trainer(model=model,search_engine=search_engine,base_path=base_path,batch_size=batch_size,device=device,max_epoch=epochs,specs=specs)

trainer.train()