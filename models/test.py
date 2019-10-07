import tensorflow as tf
import numpy as np

def loss(y_true,y_pred):
    l = tf.add(tf.reduce_mean(y_true),tf.multiply(tf.reduce_mean(y_pred),tf.constant(0.0)))
    print("loss:",l.shape)
    return l

dim = 1024
seq_len = 20
lantent_space_size = 1024
vocab_size=73000

x = tf.keras.Input(shape=(seq_len,dim))
encoder = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(dim, return_sequences=True),input_shape=(None, dim))(x)

latent_space = tf.keras.layers.Dense(lantent_space_size)(encoder)

decoder = tf.keras.layers.LSTM(dim, return_sequences=True)(latent_space)

output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))(decoder)

model = tf.keras.models.Model(inputs=x,outputs=output)

model.compile(loss=loss,optimizer="adam")

l = 0.5
y_true = np.full((1,20,73000),l)
print("y_true:",y_true.shape)
x_in = np.random.rand(1,20,1024)
print("x_in:",x_in.shape)

output = model.predict(x_in)
print("output:",output.shape)


eval_out = model.train_on_batch(x_in,y_true)
print(eval_out)