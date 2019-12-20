import torch
from models.my_transformer import MyTransformer
from dataset.bertify import construct_sentence,get_tokens
from training.dataloaders.squad_dataloader_q2q import SquadDataloaderQ2Q
from tqdm import tqdm
import jsonlines
import numpy as np

def evaluate(model,device,save_path,max_seq_len=300):
    writer = jsonlines.open(save_path+"predictions_q2q.jsonl",'w',flush=True)
    eval_data = SquadDataloaderQ2Q(base_path=base_path, batch_size=1, max_length=max_seq_len, eval=True)
    i_eval = 0
    test_loss = 0
    pbar = tqdm(total=5928, desc="evaluating")
    for eval_x, eval_y, queries, targets, x_mask, y_mask, y_emb in iter(eval_data):
        eval_x = torch.tensor(np.transpose(eval_x, (1, 0, 2)), device=device).type(torch.float64)
        y_emb = torch.tensor(np.transpose(y_emb, (1, 0, 2)), device=device).type(torch.double)
        eval_y = torch.tensor(eval_y, device=device).type(torch.long)
        x_mask = torch.tensor(x_mask, device=device).type(torch.float64)
        y_mask = torch.tensor(y_mask, device=device).type(torch.float64)
        loss,predictions = model.predict(eval_x,eval_y,x_mask,y_mask,y_emb)

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
dff = 4*d_model # dimension of feed forward
batch_size = 8
lr = 0.0001


model = MyTransformer(base_path,input_size=emb_size,output_size=vocab_size,device=device,nhead=nhead,dropout=dropout,d_model=d_model,dff=dff,lr=lr)
load_path = "/media/jonas/archive/master/data/rl_squad/experiments/Transformer_q2q_medium/latest"
model.load(load_path,train=False)
save_path = "/media/jonas/archive/master/data/rl_squad/experiments/Transformer_q2q_medium/"

evaluate(model,device,save_path=save_path)