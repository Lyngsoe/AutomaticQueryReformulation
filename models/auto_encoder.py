import tensorflow as tf

class AutoEncoder:
    def __init__(self,encoder_layers,decoder_layers,LSTM_size,latent_space_size,vocab_size):
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.LSTM_size = LSTM_size
        self.latent_space_size = latent_space_size
        self.vocab_size = vocab_size
        self.max_sentence_length = 20

        x = tf.keras.layers.Input(shape=(self.max_sentence_length, self.LSTM_size),dtype=tf.float32)
        #ENCODER
        #e1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.LSTM_size, return_sequences=True))(x)
        e1 = tf.keras.layers.LSTM(self.LSTM_size, return_sequences=True)(x)
        #LATENT SPACE
        latent_space = tf.keras.layers.Dense(self.latent_space_size)(e1)
        #DECODER
        d1 = tf.keras.layers.LSTM(self.LSTM_size, return_sequences=True)(latent_space)

        #END GAME
        predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))(d1)

        self.model = tf.keras.Model(inputs=x,outputs=predictions)


if __name__ == '__main__':
    model = AutoEncoder(encoder_layers=1,decoder_layers=4,LSTM_size=1024,latent_space_size=128,vocab_size=73636)
