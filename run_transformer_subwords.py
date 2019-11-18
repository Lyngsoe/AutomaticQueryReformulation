from training.trainer_subwords import TrainerSubwords
from models.my_transformer import MyTransformer
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#device = "cpu"
#device = "cuda"

base_path = "/scratch/s134280/squad2/"
#base_path = "/home/jonas/data/squad2/"
#base_path = "/media/jonas/archive/master/data/squad2/"

vocab_size = 30522
emb_size = 1 # embedding dimension
d_model = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
dff = 512 # dimension of feed forward
batch_size = 8
lr = 0.05
epochs = 200

#exp_name = "Transformer__11-06_11:39"
#model = MyTransformer(base_path,input_size=emb_size,output_size=vocab_size,device=device,nhead=nhead,dropout=dropout,d_model=d_model,dff=dff,lr=lr,exp_name=exp_name)
model = MyTransformer(base_path,input_size=emb_size,output_size=vocab_size,device=device,nhead=nhead,dropout=dropout,d_model=d_model,dff=dff,lr=lr)
#epoch = model.load(model.save_path+exp_name+"/latest",train=True)
#trainer = Trainer(model=model,base_path=base_path,batch_size=batch_size,device=device,epoch=epoch)
trainer = TrainerSubwords(model=model,base_path=base_path,batch_size=batch_size,device=device,max_epoch=epochs)
trainer.train()
