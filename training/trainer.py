from models.auto_encoder import AutoEncoder
from training.dataloader_bpe import DataloaderBPE
from embedding_method.embedders import get_embedder

drive_path = "/home/jonas/data/"
#drive_path = "/media/jonas/archive/master/data/"
debug = True
language = "da"
embedding_method = "laser"
epochs = 100
batch_size = 8


model = AutoEncoder(drive_path=drive_path,debug=debug,language=language,LSTM_size=128,latent_space_size=1024)
oov_embedder = get_embedder(embedding_method,language)


print("starting training")
for epoch in range(epochs):

    train_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder,embedding_method="laser",language="da",batch_size=batch_size,debug=debug)
    train_iter = 0
    mbl = 0
    for x,y in iter(train_data):
        train_iter+=1
        batch_loss = model.train_on_batch(x,y)
        mbl+=batch_loss

    print("epoch:", epoch,"train_loss:",mbl/train_iter)

    eval_data = DataloaderBPE(drive_path=drive_path, embedder=oov_embedder, embedding_method="laser", language="da",batch_size=1,debug=debug,fold=4,eval=True)
    i_eval = 0
    for eval_x, eval_y in iter(eval_data):
        sentences = model.predict_on_batch(eval_x)
        print("out_sentence:", sentences)
        i_eval+=1
        if i_eval > 3:
            break
