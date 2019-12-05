from training.rl_trainer import Trainer
from models.rl_transformer import RLTransformer
from models.losses.recall_reward import RecallRewardMean
from search_engine.elastic_search import ELSearch
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/rl_squad/"
#base_path = "/scratch/s134280/squad/"

vocab_size = 30522
emb_size = 768 # embedding dimension
d_model = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.8 # the dropout value
dff = 512 # dimension of feed forward
batch_size = 8
lr = 0.00001
epochs = 200
l2 = 0.001

reward_function = RecallRewardMean()
search_engine = ELSearch("squad")

load = False

if load:
    load_path = "/media/jonas/archive/master/data/cluster_exp/27_11_19/experiments/Transformer__11-13_16:49"
    model = RLTransformer(base_path, reward_function, input_size=emb_size, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr,l2=l2)
    epoch = model.load(load_path +"/latest",train=True)
    trainer = Trainer(model=model,search_engine=search_engine,base_path=base_path,batch_size=batch_size,device=device,epoch=epoch,max_epoch=epochs)
else:
    model = RLTransformer(base_path, reward_function, input_size=emb_size, output_size=vocab_size, device=device,nhead=nhead, dropout=dropout, d_model=d_model, dff=dff, lr=lr,l2=l2)
    trainer = Trainer(model=model,base_path=base_path,search_engine=search_engine,batch_size=batch_size,device=device,max_epoch=epochs)

trainer.train()



