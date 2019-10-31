from models.LSTM_auto_encoder_1 import LSTMAutoEncoder
import time
from training.dataloaders.squad_dataloader_2 import SquadDataloader2
from embedding_method.embedders import get_embedder
import torch
import jsonlines
import os
from tqdm import tqdm
from dataset.bertify import construct_sentence


def evaluate(model,min_test_loss):
    eval_data = SquadDataloader2(base_path=base_path,batch_size=1,max_length=max_length,eval=True)
    i_eval = 0
    test_loss = 0
    pbar = tqdm(total=int(5928), desc="evaluating batches for epoch {}".format(epoch))
    for eval_x, eval_y,queries,targets in iter(eval_data):
        #eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
        #eval_y = torch.tensor(eval_y, device=device).type(torch.float64)

        eval_x = torch.tensor(eval_x, device=device).type(torch.float64).view(-1,eval_x.shape[0],emsize)
        eval_y = torch.tensor(eval_y, device=device).type(torch.float64).view(-1,eval_y.shape[0])

        loss,predictions = model.predict(eval_x,eval_y)

        test_loss+=loss
        pbar.update()
        if i_eval < 6:
            sentences = construct_sentence(predictions)
            tqdm.write("#### EVAL")
            tqdm.write("query: {}".format(queries))
            tqdm.write("prediction: {}".format(sentences))
            tqdm.write("target: {} loss: {}".format(targets,loss))
        #else:
            #break
        i_eval+=1

    test_loss=test_loss/i_eval
    pbar.close()

    epoch_summary = {
        "epoch":epoch,
        "test_loss":test_loss,
        "train_loss":mbl/train_iter
    }
    result_writer.write(epoch_summary)

    if min_test_loss is None or test_loss < min_test_loss:
        min_test_loss = test_loss
        model.save_best(epoch)

    model.save_latest(epoch)
    tqdm.write("model saved!")

    tqdm.write("epoch: {} train_loss: {}  test_loss: {}".format(epoch,mbl / train_iter,test_loss))

    return min_test_loss




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/squad/"
language = "en"
#embedding_method = "laser"
epochs = 200
batch_size = 32
ntokens = 119547 # the size of vocabulary
# bert size = 119547
# laser size = 73638
emsize = 768 # embedding dimension
nhid = 1024 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
max_length=50
exp_name = "LSTM_auto_encoder_1__10-28_11:23"
model = LSTMAutoEncoder(base_path,max_length=max_length,hidden_size=nhid,word_emb_size=emsize,lantent_space_size=nhid,vocab_size=ntokens,device=device,exp_name=exp_name)

tqdm.write(model.exp_name)
exp_path = model.save_path+model.exp_name
epoch = model.load(exp_path+"/latest/",train=True)
os.makedirs(exp_path+"/latest",exist_ok=True)
os.makedirs(exp_path+"/best",exist_ok=True)
result_writer = jsonlines.open(exp_path+"/resutls.jsonl",'a')
min_test_loss = None
max_batch = 100
tqdm.write("starting training")

while epoch < epochs:
    pbar = tqdm(total=int(86821/batch_size),desc="training batches for epoch {}".format(epoch))
    train_data = SquadDataloader2(base_path=base_path,batch_size=batch_size,max_length=max_length,eval=False)
    train_iter = 0
    mbl = 0
    start_time = time.time()
    for x,y in iter(train_data):

        x_tensor = torch.tensor(x, device=device).type(torch.float64).view(-1,x.shape[0],emsize)
        y_tensor = torch.tensor(y, device=device).type(torch.float64).view(-1,y.shape[0])

        train_iter+=1
        data_time = time.time() - start_time
        start_time = time.time()
        batch_loss = model.train(x_tensor, y_tensor)
        train_time =  time.time() - start_time
        mbl+=batch_loss
        pbar.set_description("training batches for epoch {} with training loss: {:.2f}".format(epoch, batch_loss))
        pbar.update()
        #print("data time:",data_time)
        #print("train time:",train_time)
        #print("ratio:",train_time/data_time)
        #if train_iter % max_batch == 0:
        break
    pbar.close()

    min_test_loss = evaluate(model,min_test_loss)
    epoch+=1



