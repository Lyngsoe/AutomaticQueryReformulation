from models.LSTM_auto_encoder import LSTMAutoEncoder
import time
from training.dataloader_bpe import DataloaderBPE
from embedding_method.embedders import get_embedder
import torch
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"



#drive_path = "/home/jonas/data/"
drive_path = "/media/jonas/archive/master/data/"
debug = True
language = "da"
embedding_method = "laser"
epochs = 100
batch_size = 8
oov_embedder = get_embedder(embedding_method,language)

model = LSTMAutoEncoder(drive_path=drive_path,language=language,device=device,debug=debug)

start = time.time()
plot_losses = []
print_loss_total = 0  # Reset every print_every
plot_loss_total = 0  # Reset every plot_every



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
        train_iter+=1
        batch_loss = model.train(x_tensor, y_tensor)
        #print("batch loss",batch_loss)
        mbl+=batch_loss


    print("epoch:", epoch,"train_loss:",mbl/train_iter)

    eval_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder, embedding_method="laser", language="da",batch_size=1,debug=debug,fold=0,eval=True)
    i_eval = 0
    for eval_x, eval_y in iter(eval_data):
        eval_x = torch.tensor(eval_x, device=device).type(torch.float64)
        sentences = model.predict(eval_x)
        print("out_sentence:", sentences)
        i_eval+=1
        if i_eval > 3:
            break

