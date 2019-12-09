from training.trainer_subwords import TrainerSubwords
from models.my_transformer import MyTransformer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(device)
#device = "cpu"
#device = "cuda"

#base_path = "/scratch/s134280/squad2/"
#base_path = "/home/jonas/data/squad2/"
base_path = "/media/jonas/archive/master/data/squad2/"

vocab_size = 30522
emb_size = 1 # embedding dimension
d_model = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.5 # the dropout value
dff = 1024 # dimension of feed forward
batch_size = 8
lr = 0.0001
epochs = 200
l2 = 0

specs = {
    "vocab_size":vocab_size,
    "emb_size":emb_size,
    "d_model": d_model,
    "n_layers": n_layers,
    "nhead": nhead,
    "dropout": dropout,
    "dff": dff,
    "lr": lr,
    "l2": l2,
    "epochs": epochs,
}

load = False

if load:
    load_path = "/media/jonas/archive/master/data/squad/experiments/Transformer__12-05_19:36"
    model = MyTransformer(base_path, input_size=emb_size, output_size=vocab_size, device=device, nhead=nhead,dropout=dropout, d_model=d_model, dff=dff, lr=lr)
    epoch = model.load(load_path  + "/latest", train=False)
    trainer = TrainerSubwords(model=model, base_path=base_path, batch_size=batch_size, device=device, epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = MyTransformer(base_path, num_layers=n_layers, input_size=emb_size, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr, l2=l2)
    trainer = TrainerSubwords(model=model, base_path=base_path, batch_size=batch_size, max_epoch=epochs, device=device,specs=specs)


trainer.train()