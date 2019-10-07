from models.auto_encoder import AutoEncoder
from training.dataloader_simple import DataloaderSimple
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
        if id2bpe.get(w) is None:
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
            token = id2bpe.get(str(ind))
            sent_bpe.append(token)
        bpe_tokens.append(sent_bpe)

    return bpe_tokens


data_path = "/home/jonas/data/raffle_wiki/da/debug/"
#data_path = "/media/jonas/archive/master/data/raffle_wiki/da/debug/"
epochs = 20
embedder = get_embedder(method="laser", language="da")
id2bpe = json.load(open(data_path + "id2bpe.json", 'r'))

emb_size = 1024
encoder_layers = 1
decoder_layers = 1
LSTM_size = 1024
latent_space_size = 128
vocab_size = 73637
mask_value = 0
#
model = tf.keras.models.Sequential(
[
    #tf.keras.layers.Masking(mask_value=mask_value, input_shape=(None, emb_size)),
    tf.keras.Input(shape=(None,emb_size)),
    #tf.keras.layers.Embedding(input_dim=emb_size, output_dim=LSTM_size, mask_zero=True),
    tf.keras.layers.Dense(LSTM_size),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(LSTM_size, return_sequences=True)),
    tf.keras.layers.Dense(latent_space_size),
    tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    #tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    #tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    #tf.keras.layers.LSTM(LSTM_size, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))
])
model.compile(loss='categorical_crossentropy',optimizer="adam")

for epoch in range(epochs):

    eval_data = DataloaderSimple(data_base_path=data_path, embedder=embedder, embedding_method="laser", language="da")
    for x,y in iter(eval_data):
        #padded_inputs = tf.keras.preprocessing.sequence.pad_sequences(x,padding='post',value=mask_value)

        #print("padded_input:",padded_inputs.shape)
        #print(padded_inputs[0][5])
        print("epoch:",epoch)
        print("x",x.shape)
        print("y",y.shape)
        print("y",len(y[0]))
        print("y", len(y[0][0]))
        seq_pred = model.predict_on_batch(x)
        bpe_tokens = get_tokens(seq_pred)
        sentences = compress(bpe_tokens)
        print("out_sentence:",sentences[0])
        print("out_sentence:", sentences[1])
        break


    train_data = DataloaderSimple(data_base_path=data_path, embedder=embedder,embedding_method="laser",language="da")
    for x,y in iter(train_data):
        #padded_x = tf.keras.preprocessing.sequence.pad_sequences(x, padding='post',value=mask_value)
        #padded_y = tf.keras.preprocessing.sequence.pad_sequences(y, padding='post',maxlen=padded_x.shape[1],value=mask_value)
        #print("x",x.shape)
        #print("y",y.shape)
        model.fit(x,y)