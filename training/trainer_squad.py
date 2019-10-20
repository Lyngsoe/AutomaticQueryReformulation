from models.LSTM_auto_encoder_1 import LSTMAutoEncoder
import time
from training.dataloaders.squad_dataloader import SquadDataloader
import torch
import jsonlines
import os
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

def evaluate(model,min_test_loss):
    eval_data = SquadDataloader(base_path=base_path,language=language,batch_size=1,max_length=max_length,eval=True)
    i_eval = 0
    test_loss = 0
    test_sentences = []
    for eval_x, eval_y,queries,targets in iter(eval_data):
        eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
        eval_y = torch.tensor(eval_y, device=device).type(torch.float64)
        sentences,loss = model.predict(eval_x,eval_y)
        test_sentences.append(sentences[0])
        test_loss+=loss

        if i_eval < 3:
            print("#### EVAL")
            print("query:", queries)
            print("prediction:", sentences, " loss:", loss)
            print("target:", targets)
        else:
            break
        i_eval += 1

    test_loss=test_loss/i_eval


    epoch_summary = {
        "epoch":epoch,
        "test_loss":test_loss,
        "train_loss":mbl/train_iter
        #"test_sentences":test_sentences
    }
    result_writer.write(epoch_summary)

    if min_test_loss is None or test_loss < min_test_loss:
        min_test_loss = test_loss
        model.save(epoch)

    print("epoch:", epoch, "train_loss:", mbl / train_iter," test_loss:",test_loss)

    return min_test_loss



#base_path = "/home/jonas/data/squad/"
base_path = "/media/jonas/archive/master/data/squad/"
debug = False
language = "en"
embedding_method = "laser"
epochs = 100
batch_size = 32
max_length = 50

model = LSTMAutoEncoder(base_path=base_path,device=device,max_length=max_length)

print(model.exp_name)
exp_path = model.save_path+model.exp_name
os.makedirs(exp_path,exist_ok=True)
result_writer = jsonlines.open(exp_path+"/resutls.jsonl",'w')
min_test_loss = None
max_batch = 100
print("starting training")
for epoch in range(epochs):
    pbar = tqdm(total=int(86821/batch_size),desc="training batches for epoch {}".format(epoch))
    #pbar = tqdm(total=max_batch, desc="training batches for epoch {}".format(epoch))
    train_data = SquadDataloader(base_path=base_path,language=language,batch_size=batch_size,max_length=max_length,eval=False)
    train_iter = 0
    mbl = 0
    for x,y in iter(train_data):
        x_tensor = torch.tensor(x, device=device).type(torch.float64)
        y_tensor = torch.tensor(y, device=device).type(torch.float64)
        train_iter+=1
        batch_loss = model.train(x_tensor, y_tensor)
        mbl+=batch_loss
        pbar.set_description("training batches for epoch {} with training loss: {:.2f}".format(epoch, batch_loss))
        pbar.update()

        if train_iter % 100 == 0:
            min_test_loss = evaluate(model,min_test_loss)


    pbar.close()


