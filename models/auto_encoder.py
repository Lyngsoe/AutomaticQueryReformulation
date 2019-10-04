from tensorflow.keras.layers import Bidirectional,LSTM,Dense,TimeDistributed
from tensorflow.keras.models import Sequential

# The expected structure has the dimensions [samples, timesteps, features]


class AutoEncoder:
    def __init__(self,encoder_layers,decoder_layers,LSTM_size,latent_space_size,vocab_size):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.LSTM_size = LSTM_size
        self.latent_space_size = latent_space_size
        self.vocab_size = vocab_size

        self.model = Sequential()

        #ENCODER
        self.encoder()

        #LATENT SPACE
        self.model.add(Dense(self.latent_space_size))

        #DECODER
        self.decoder()

        #END GAME
        self.model.add(TimeDistributed(Dense(self.vocab_size, activation='softmax')))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    def encoder(self):
        for i in range(self.encoder_layers):
            self.model.add(Bidirectional(LSTM(self.LSTM_size, return_sequences=True),input_shape=(None, self.LSTM_size), name="encoder_{}".format(i)))

    def decoder(self):
        for i in range(self.encoder_layers):
            self.model.add(LSTM(self.LSTM_size, return_sequences=True,name="decoder_{}".format(i)))

    def train(self,X,y):
        self.model.fit(X, y, epochs=1, batch_size=1, verbose=2)

if __name__ == '__main__':

    model = AutoEncoder(encoder_layers=1,decoder_layers=4,LSTM_size=128,latent_space_size=128,vocab_size=73636)