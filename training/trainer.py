from models.auto_encoder import AutoEncoder
from training.dataloader import Dataloader
from embedding_method.embedders import get_embedder
import json
import tensorflow as tf
import numpy as np
from sklearn.metrics import log_loss

def loss(y_true,y_pred):
    return tf.keras.backend.mean(y_true)+0*tf.keras.backend.mean(y_pred)

def euclidean_distance_loss(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.square(y_pred - y_true), axis=-1)))

def one_hot_language(words):
    is_language = []
    for w in words:
        if lookup.get(w) is None:
            is_language.append(0)
        else:
            is_language.append(1)
    il = np.array(is_language)
    #print("is_language:", il.shape,il.dtype,type(il))
    predicted = np.ones_like(il)
    #print("predicted:", predicted.shape,predicted.dtype,type(predicted))
    lang_loss = log_loss(il,predicted,labels=[1,0])
    #print("language loss:", lang_loss)
    return lang_loss

def language_loss(sentences):
    sentence_language = []
    for sentence in sentences:
        words = sentence.split(" ")
        sentence_language.append(one_hot_language(words))

    #print("sentence_language:",len(sentence_language))
    stacked = np.stack(sentence_language)
    #print("stacked:", stacked.shape, stacked.dtype, type(stacked))
    mean_lang_loss = np.mean(stacked)
    #print("mean_lang_loss:", mean_lang_loss)
    return mean_lang_loss

def compress(bpe_tokens):
    sentences = []
    for pred in bpe_tokens:
        tokens = " ".join(pred)
        tokens = tokens.replace("@@ ", "")
        tokens = tokens.replace("@@", "")
        sentences.append(tokens)
    return sentences

def get_tokens(model_out):
    indices = np.argmax(model_out, axis=2)
    #print("index", indices.shape, type(indices))
    bpe_tokens = []
    for sample in list(indices):
        sent_bpe = []
        for ind in sample:
            token = lookup.get(str(ind))
            sent_bpe.append(token)
        bpe_tokens.append(sent_bpe)

    return bpe_tokens


# data_path = "/home/jonas/data/raffle_wiki/da/debug/"
data_path = "/media/jonas/archive/master/data/raffle_wiki/da/debug/"
epochs = 20
embedder = get_embedder(method="laser", language="da")
lookup = json.load(open(data_path + "id2word.json", 'r'))

encoder_layers = 1
decoder_layers = 1
LSTM_size = 1024
latent_space_size = 1024
vocab_size = 73636
max_sentence_length = 20

model = tf.keras.models.Sequential([
    tf.keras.layers.Input((max_sentence_length,LSTM_size)),
    #tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_size, return_sequences=True)),
    #tf.keras.layers.Dense(latent_space_size),
    tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])

#x = tf.keras.layers.Input((max_sentence_length,LSTM_size))
# ENCODER
# e1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(self.LSTM_size, return_sequences=True))(x)
#e1 = tf.keras.layers.LSTM(LSTM_size, return_sequences=True)
# LATENT SPACE
#latent_space = tf.keras.layers.Dense(latent_space_size)
# DECODER
#d1 = tf.keras.layers.LSTM(LSTM_size, return_sequences=True)

# END GAME
#predictions = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))



#model = tf.keras.Model(inputs=x, outputs=predictions)
model.compile(loss=loss,optimizer="adam")


for epoch in range(epochs):
    train_data = Dataloader(data_base_path=data_path, embedder=embedder,embedding_method="laser",language="da")

    for x,y in iter(train_data):
        #print("x:",x.shape,type(x))
        #print("y:",len(y),type(y))
        #print("x:", x.shape, type(x))
        seq_pred = model.predict_on_batch(x)

        bpe_tokens = get_tokens(seq_pred)

        sentences = compress(bpe_tokens)
        print(sentences[0])


        #print("out_sentence:",sentences[0])
        sentences_embeddings = embedder(sentences)
        sentences_embeddings = np.expand_dims(sentences_embeddings,1)

        #print("sentence_out:",sentences_embeddings.shape)
        #print("annotation:",y.shape)

        distance_loss = euclidean_distance_loss(sentences_embeddings, y)
        #print("loss:",distance_loss)
        l_loss = language_loss(sentences)
        #print("language loss:",l_loss)

        current_loss = distance_loss + l_loss*0
        #print("current loss:",current_loss)

        batch_size = len(sentences)
        #print(type(current_loss))
        full_loss = np.full((batch_size,20,73636),float(current_loss))
        #print("full_loss:",full_loss.shape)
        model.fit(x,full_loss)
        #print(model_loss)