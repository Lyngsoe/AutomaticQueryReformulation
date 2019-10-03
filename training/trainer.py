from models.auto_encoder import AutoEncoder
from training.dataloader import Dataloader

data_path = "/home/jonas/data/raffle_wiki/da/debug/"
epochs = 2

model = AutoEncoder(encoder_layers=2,decoder_layers=1,LSTM_size=10,latent_space_size=5)

for epoch in range(epochs):
    train = Dataloader(data_base_path=data_path, embedding_method="laser")