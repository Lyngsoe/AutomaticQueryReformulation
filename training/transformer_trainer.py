from models.transformer import Transformer
from models.transformer_model2 import MyTransformer
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
    pbar = tqdm(total=int(293), desc="evaluating batches for epoch {}".format(epoch))
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




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

base_path = "/home/jonas/data/squad/"
#base_path = "/media/jonas/archive/master/data/squad/"
language = "en"
embedding_method = "laser"
epochs = 200
batch_size = 8
#oov_embedder = get_embedder(embedding_method,language)
ntokens = 30522 # the size of vocabulary
# bert size = 119547
# laser size = 73638
# small bert = 30522
emsize = 768 # embedding dimension
nhid = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 1 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
#model = Transformer(base_path,ntokens, emsize, nhead, nhid, nlayers,device, dropout)
#model = MyTransformer(base_path,output_size=ntokens, d_model=nhid, nhead=nhead, encoder_layers=nlayers,decoder_layers=nlayers,device=device)
model = MyTransformer(base_path,input_size=emsize,output_size=ntokens,device=device)

max_length=50
tqdm.write(model.exp_name)
exp_path = model.save_path+model.exp_name
os.makedirs(exp_path+"/latest",exist_ok=True)
os.makedirs(exp_path+"/best",exist_ok=True)
result_writer = jsonlines.open(exp_path+"/resutls.jsonl",'w')
min_test_loss = None
max_batch = 100
tqdm.write("starting training")

for epoch in range(epochs):
    pbar = tqdm(total=int(753/batch_size),desc="training batches for epoch {}".format(epoch))
    train_data = SquadDataloader2(base_path=base_path,batch_size=batch_size,max_length=max_length,eval=False)
    train_iter = 0
    mbl = 0
    start_time = time.time()
    for x,y in iter(train_data):

        x_tensor = torch.tensor(x, device=device).type(torch.double).view(-1,x.shape[0],emsize)
        y_tensor = torch.tensor(y, device=device).type(torch.double).view(-1,y.shape[0])

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
    pbar.close()

    min_test_loss = evaluate(model,min_test_loss)



