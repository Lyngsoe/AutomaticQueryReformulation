from training.trainer_q2q import TrainerQ2Q
from models.my_transformer import MyTransformer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"
#base_path = "/scratch/s134280/squad/"

vocab_size = 30522
emb_size = 768 # embedding dimension
d_model = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
dff = 4*d_model # dimension of feed forward
batch_size = 8
lr = 0.0001
l2 = 0
epochs = 250

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
    "q2q":True
}

load = False

if load:
    load_path = "/media/jonas/archive/master/data/rl_squad/experiments/Transformer__12-16_16:51"
    exp_name = "Transformer__12-16_16:51"
    model = MyTransformer(base_path, input_size=emb_size, output_size=vocab_size, device=device, nhead=nhead,dropout=dropout, d_model=d_model, dff=dff, lr=lr,exp_name=exp_name)
    epoch = model.load(load_path  + "/latest", train=False)
    trainer = TrainerQ2Q(model=model, base_path=base_path, batch_size=batch_size, device=device, epoch=epoch,max_epoch=epochs,specs=specs)
else:
    model = MyTransformer(base_path, num_layers=n_layers, input_size=emb_size, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr, l2=l2)
    trainer = TrainerQ2Q(model=model, base_path=base_path, batch_size=batch_size, max_epoch=epochs, device=device,specs=specs)


trainer.train()
