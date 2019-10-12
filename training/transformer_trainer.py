from models.transformer import Transformer
import time
from training.dataloaders.squad_dataloader import SquadDataloader
from embedding_method.embedders import get_embedder
import torch
import jsonlines
import os
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"



base_path = "/home/jonas/data/squad/"
#drive_path = "/media/jonas/archive/master/data/"
debug = True
language = "da"
embedding_method = "laser"
epochs = 200
batch_size = 8
oov_embedder = get_embedder(embedding_method,language)
ntokens = 73638 # the size of vocabulary
emsize = 1024 # embedding dimension
nhid = 128 # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 2 # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 2 # the number of heads in the multiheadattention models
dropout = 0.2 # the dropout value
model = Transformer(base_path,ntokens, emsize, nhead, nhid, nlayers,device, dropout)

exp_path = model.save_path+model.exp_name
os.makedirs(exp_path,exist_ok=True)
result_writer = jsonlines.open(exp_path+"/resutls.jsonl",'w')
min_test_loss = None
max_length = 20
max_batch = 10
print("starting training")


for epoch in range(epochs):
    train_data = SquadDataloader(base_path=base_path,language=language,batch_size=batch_size,max_length=max_length,eval=False)
    train_iter = 0
    mbl = 0
    for x,y in iter(train_data):
        x_tensor = torch.tensor(x, device=device).type(torch.float64)
        y_tensor = torch.tensor(y, device=device).type(torch.float64)
        train_iter+=1
        batch_loss = model.train(x_tensor, y_tensor)
        mbl+=batch_loss
        if train_iter > max_batch:
            break
    eval_data = SquadDataloader(base_path=base_path, language=language, batch_size=1, max_length=max_length, eval=True)
    i_eval = 0
    test_loss = 0
    test_sentences = []
    for eval_x, eval_y, queries, targets in iter(eval_data):
        eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
        eval_y = torch.tensor(eval_y, device=device).type(torch.float64)
        sentences, loss = model.predict(eval_x, eval_y)
        test_sentences.append(sentences[0])
        test_loss += loss

        if i_eval < 1:
            print("#### EVAL")
            print("query:", queries[0]["text"])
            print("prediction:", sentences[0], " loss:", loss)
            print("target:", targets[0])
        i_eval += 1
    test_loss = test_loss / i_eval

    epoch_summary = {
        "epoch": epoch,
        "test_loss": test_loss,
        "train_loss": mbl / train_iter
        # "test_sentences":test_sentences
    }
    result_writer.write(epoch_summary)

    if min_test_loss is None or test_loss < min_test_loss:
        min_test_loss = test_loss
        model.save(epoch)

    print("epoch:", epoch, "train_loss:", mbl / train_iter, " test_loss:", test_loss)