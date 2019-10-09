import tensorflow as tf
import numpy as np
import json
from datetime import datetime


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class AutoEncoder:
    def __init__(self,drive_path,language,encoder_layers=1,decoder_layers=1,LSTM_size=128,latent_space_size=512,vocab_size=73637,emb_size=1024,debug=False,exp_name=None):
        self.debug = debug
        self.base_path = drive_path+"raffle_wiki/{}/debug/".format(language) if debug else drive_path+"raffle_wiki/{}/".format(language)
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.LSTM_size = LSTM_size
        self.latent_space_size = latent_space_size
        self.vocab_size = vocab_size
        self.max_sentence_length = 20
        self.mask_value = 0
        self.emb_size=emb_size
        self.id2bpe = json.load(open(self.base_path + "id2bpe.json", 'r'))
        self.model_name="LSTM_auto_encoder"

        self.exp_name = exp_name if exp_name is not None else self.get_exp_name()
        print(self.exp_name)

        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Masking(mask_value=self.mask_value, input_shape=(None, self.emb_size)),
                # tf.keras.Input(shape=(None,emb_size)),
                # tf.keras.layers.Embedding(input_dim=emb_size, output_dim=LSTM_size, mask_zero=True),
                tf.keras.layers.Dense(self.LSTM_size),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.LSTM_size, activation='sigmoid', return_sequences=True)),
                tf.keras.layers.Dense(self.latent_space_size),
                tf.keras.layers.LSTM(self.LSTM_size, activation='sigmoid', return_sequences=True),
                # tf.keras.layers.LSTM(LSTM_size,activation='sigmoid',stateful=True, return_sequences=True),
                # tf.keras.layers.LSTM(LSTM_size,activation='sigmoid',stateful=True, return_sequences=True),
                # tf.keras.layers.LSTM(LSTM_size,activation='sigmoid',stateful=True, return_sequences=True),
                tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.vocab_size, activation='softmax'))
            ])

        self.model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

    def train_on_batch(self,x,y):
        return self.model.train_on_batch(x,y)

    def predict_on_batch(self,x):
        seq_pred = self.model.predict_on_batch(x)
        bpe_tokens = self.get_tokens(seq_pred)
        sentences = self.compress(bpe_tokens)
        return sentences

    def save_model(self):
        tf.keras.models.save_model()
    def get_model(self):
        return self.model

    def compress(self,bpe_tokens):
        sentences = []
        for pred in bpe_tokens:
            tokens = " ".join(pred)
            tokens = tokens.replace("@@ ", "")
            tokens = tokens.replace("@@", "")
            sentences.append(tokens)
        return sentences

    def get_tokens(self,model_out):
        indices = np.argmax(model_out, axis=2)
        # print("index", indices.shape, type(indices))
        bpe_tokens = []
        for sample in list(indices):
            sent_bpe = []
            for ind in sample:
                token = self.id2bpe.get(str(ind))
                sent_bpe.append(token)
            bpe_tokens.append(sent_bpe)

        return bpe_tokens

    def get_exp_name(self):
        now = datetime.now()
        self.exp_name = self.model_name+"__"+now.strftime("%m-%d_%H:%M")

if __name__ == '__main__':
    model = AutoEncoder(encoder_layers=1,decoder_layers=4,LSTM_size=1024,latent_space_size=128,vocab_size=73636)
