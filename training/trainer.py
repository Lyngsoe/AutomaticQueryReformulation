from models.auto_encoder import AutoEncoder
from training.dataloader import Dataloader

data_path = "/home/jonas/data/raffle_wiki/da/debug/"
epochs = 1

model = AutoEncoder(encoder_layers=1,decoder_layers=4,LSTM_size=10,latent_space_size=5,vocab_size=7015)

for epoch in range(epochs):
    train_data = Dataloader(data_base_path=data_path, embedding_method="laser",language="da")

    for x,y in iter(train_data):
        print(len(x))
        print(len(x[0]))
        print(type(x[0][0]))
        print("x-shape:",len(x),x[0].size)
        print(model.model.evaluate(x,y,10))
        raise Exception