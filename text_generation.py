import tensorflow as tf
import os
import numpy as np

from data_preparation import *
from model import *

BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset, char2idx, idx2char, vocab = text2tf_data(filename='donald_tweets.txt',
                                                  seq_length=100,
                                                  buffer_size=BUFFER_SIZE,
                                                  batch_size=BATCH_SIZE)

VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNIT = 1024

model = build_model(
    vocab_size=VOCAB_SIZE,
    embedding_dim=EMBEDDING_DIM,
    rnn_units=RNN_UNIT,
    batch_size=BATCH_SIZE)
model.summary()

LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
OPTIMIZER = 'adam'
model.compile(optimizer=OPTIMIZER, loss=LOSS)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS = 10
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNIT, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))


def generate_text(model, start_string, num_generate):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []
    temperature = 1.0

    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a categorical distribution to predict the character returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # Pass the predicted character as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx2char[predicted_id])
    return start_string + ''.join(text_generated)


print(generate_text(model, u"@", 1000))
