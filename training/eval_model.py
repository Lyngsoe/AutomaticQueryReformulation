import torch
from models.my_transformer import MyTransformer
from dataset.bertify import construct_sentence,get_tokens
from training.dataloaders.squad_dataloader_2 import SquadDataloader2
from tqdm import tqdm
import jsonlines

def evaluate(model,device,save_path,max_seq_len=300):
    writer = jsonlines.open(save_path+"predictions.jsonl",'w',flush=True)
    eval_data = SquadDataloader2(base_path=base_path, batch_size=1, max_length=max_seq_len, eval=True)
    i_eval = 0
    test_loss = 0
    pbar = tqdm(total=5928, desc="evaluating")
    for eval_x, eval_y, queries, targets in iter(eval_data):
        eval_x = torch.tensor(eval_x, device=device).type(torch.float64).view(-1, eval_x.shape[0], eval_x.shape[2])
        eval_y = torch.tensor(eval_y, device=device).type(torch.float64).view(-1, eval_y.shape[0])

        loss, predictions = model.predict(eval_x, eval_y)

        test_loss += loss
        pbar.update()
        sentences = construct_sentence(predictions)
        if i_eval < 6:
            tqdm.write("#### EVAL")
            tqdm.write("query: {}".format(queries))
            tqdm.write("prediction: {}".format(sentences))
            tqdm.write("target: {} loss: {}".format(targets, loss))
        i_eval += 1
        to_save = {"query": queries[0],
                   "targets": targets[0],
                   "sentence":sentences}
        writer.write(to_save)


    test_loss = test_loss / i_eval
    tqdm.write("test_loss = {:.2f}".format(test_loss))
    pbar.close()


base_path = "/media/jonas/archive/master/data/squad/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab_size = 30522
emb_size = 768 # embedding dimension
d_model = 256 # the dimension of the feedforward network model in nn.TransformerEncoder
n_layers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 8 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
dff = 1024 # dimension of feed forward
batch_size = 8
lr = 0.01


model = MyTransformer(base_path,input_size=emb_size,output_size=vocab_size,device=device,nhead=nhead,dropout=dropout,d_model=d_model,dff=dff,lr=lr)
load_path = "/media/jonas/archive/master/data/cluster_exp/28_11_19/experiments/Transformer__11-13_16:49/latest"
model.load(load_path,train=False)
save_path = "/media/jonas/archive/master/data/cluster_exp/28_11_19/experiments/Transformer__11-13_16:49/"

evaluate(model,device,save_path=save_path)