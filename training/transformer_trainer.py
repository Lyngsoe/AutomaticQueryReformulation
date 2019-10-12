from models.LSTM_auto_encoder_1 import LSTMAutoEncoder
import time
from training.dataloader_bpe import DataloaderBPE
from embedding_method.embedders import get_embedder
import torch
import jsonlines
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"



#drive_path = "/home/jonas/data/"
drive_path = "/media/jonas/archive/master/data/"
debug = True
language = "da"
embedding_method = "laser"
epochs = 200
batch_size = 8
oov_embedder = get_embedder(embedding_method,language)

model = LSTMAutoEncoder(drive_path=drive_path,language=language,device=device,debug=debug)

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every
print(model.exp_name)
exp_path = model.save_path+model.exp_name
os.makedirs(exp_path,exist_ok=True)
result_writer = jsonlines.open(exp_path+"/resutls.jsonl",'w')
min_test_loss = None
print("starting training")
for epoch in range(epochs):

    train_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder,embedding_method="laser",language="da",batch_size=batch_size,debug=debug)
    train_iter = 0
    mbl = 0
    for x,y in iter(train_data):
        #x_tensor = torch.stack([torch.FloatTensor(sample,device=device) for sample in x])
        #y_tensor = torch.stack([torch.FloatTensor(sample,device=device) for sample in y])
        #x_tensor = torch.from_numpy(x)
        #y_tensor = torch.from_numpy(y)
        x_tensor = torch.tensor(x, device=device).type(torch.float64)
        y_tensor = torch.tensor(y, device=device).type(torch.float64)
        #print(x_tensor.size())
        #print(y_tensor.size())
        train_iter+=1
        batch_loss = model.train(x_tensor, y_tensor)
        #print("batch loss",batch_loss)
        mbl+=batch_loss

    eval_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder, embedding_method="laser", language="da",batch_size=1,debug=debug,fold=3,eval=True)
    i_eval = 0
    test_loss = 0
    test_sentences = []
    for eval_x, eval_y,queries,targets in iter(eval_data):
        eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
        eval_y = torch.tensor(eval_y, device=device).type(torch.float64)
        sentences,loss = model.predict(eval_x,eval_y)
        test_sentences.append(sentences[0])
        test_loss+=loss

        if i_eval < 1:
            print("#### EVAL")
            print("query:", queries[0]["text"])
            print("prediction:", sentences[0], " loss:", loss)
            print("target:", targets[0])
        i_eval += 1


    test_loss=test_loss/i_eval


    epoch_summary = {
        "epoch":epoch,
        "test_loss":test_loss,
        "train_loss":mbl/train_iter,
        "test_sentences":test_sentences
    }
    result_writer.write(epoch_summary)

    if min_test_loss is None or test_loss < min_test_loss:
        min_test_loss = test_loss
        model.save(epoch)

    print("epoch:", epoch, "train_loss:", mbl / train_iter," test_loss:",test_loss)